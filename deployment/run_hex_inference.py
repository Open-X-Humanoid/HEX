from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zmq


HEX_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEAR_SONIC_ROOT = Path("/media/bsh/code/GR00T-WholeBodyControl")

for root in (HEX_ROOT, DEFAULT_GEAR_SONIC_ROOT):
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))

PROMPT_MSG_PREFIX = "prompt:"
HAND_BINARY_CLOSE_THRESHOLD = 0.5
DEFAULT_ZMQ_KEYBOARD_PORT = 5580


def _load_runtime_dependencies() -> None:
    global ComposedCameraClientSensor
    global instantiate_g1_robot_model
    global ZMQKeyboardSubscriber
    global ZMQStateSubscriber
    global Telemetry
    global compute_projected_gravity
    global LATENT_INITIAL_MOTION_TOKEN
    global calculate_latency_compensated_index
    global prepare_observation_for_eval
    global should_trigger_new_inference
    global G1GripperInverseKinematicsSolver
    global build_command_message
    global pack_pose_message

    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
    from gear_sonic.utils.data_collection.telemetry import Telemetry
    from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
    from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
    from gear_sonic.utils.inference.vla_utils import (
        calculate_latency_compensated_index,
        prepare_observation_for_eval,
        should_trigger_new_inference,
    )
    from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
        G1GripperInverseKinematicsSolver,
    )
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        pack_pose_message,
    )


@dataclass
class ClosedLoopConfig:
    # HEX ZMQ policy server.
    host: str = "127.0.0.1"
    port: int = 10093
    policy_timeout_ms: int = 60000

    # Optional checkpoint/stats path used for q99 state/action normalization.
    checkpoint_path: str | None = None
    stats_path: str | None = None
    embodiment_tag: str = "unitree_g1_sonic"
    clip_normalized_actions: bool = False

    # Runtime loop rates.
    action_publish_rate: int = 50
    action_horizon: int = 100
    inference_rate: float = 1 / 0.8

    # Camera server.
    camera_host: str = "localhost"
    camera_port: int = 5555

    # Robot state from gear_sonic_deploy output, topic g1_debug.
    state_zmq_host: str = "localhost"
    state_zmq_port: int = 5557

    # Action output to gear_sonic_deploy --input-type zmq_manager.
    action_zmq_host: str = "localhost"
    action_zmq_port: int = 5556

    # Keyboard control, compatible with /media/bsh/code/GR00T-WholeBodyControl/keyboard_pub.py.
    keyboard_zmq_host: str = "localhost"
    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT

    prompt: str = "pick up the cola"
    verbose_timing: bool = False


class ZMQPolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 60000):
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        if hasattr(self, "_socket"):
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    @staticmethod
    def _to_bytes(data: Any) -> bytes:
        import msgpack_numpy as mnp

        return mnp.packb(data, default=mnp.encode)

    @staticmethod
    def _from_bytes(data: bytes) -> Any:
        import msgpack_numpy as mnp

        return mnp.unpackb(data, object_hook=mnp.decode, raw=False)

    def call_endpoint(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data

        try:
            self._socket.send(self._to_bytes(request))
            message = self._socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise

        response = self._from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(response["error"])
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping")
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": options})
        return tuple(response)

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


def _parse_args() -> ClosedLoopConfig:
    parser = argparse.ArgumentParser(description="Run HEX -> SONIC closed-loop inference.")
    for field_name, field_def in ClosedLoopConfig.__dataclass_fields__.items():
        default = field_def.default
        arg_type = type(default) if default is not None else str
        arg_names = [f"--{field_name}"]
        kebab_name = f"--{field_name.replace('_', '-')}"
        if kebab_name not in arg_names:
            arg_names.append(kebab_name)
        if field_name == "host":
            arg_names.append("--policy-host")
        elif field_name == "port":
            arg_names.append("--policy-port")
        if isinstance(default, bool):
            parser.add_argument(*arg_names, action="store_true", default=default)
        else:
            parser.add_argument(*arg_names, type=arg_type, default=default)
    return ClosedLoopConfig(**vars(parser.parse_args()))


def _print_green(text: str) -> None:
    print(f"\033[92m{text}\033[0m", flush=True)


def _find_stats_path(config: ClosedLoopConfig) -> Path | None:
    if config.stats_path:
        path = Path(config.stats_path)
        if not path.exists():
            raise FileNotFoundError(f"stats_path does not exist: {path}")
        return path

    if not config.checkpoint_path:
        return None

    checkpoint = Path(config.checkpoint_path)
    for parent in (checkpoint.parent, *checkpoint.parents):
        candidate = parent / "dataset_statistics.json"
        if candidate.exists():
            return candidate
    return None


def _load_stats(config: ClosedLoopConfig) -> dict[str, Any] | None:
    stats_path = _find_stats_path(config)
    if stats_path is None:
        print(
            "WARNING: no dataset_statistics.json found. "
            "State/action normalization is disabled; pass --checkpoint_path or --stats_path.",
            flush=True,
        )
        return None

    with open(stats_path, "r") as f:
        all_stats = json.load(f)
    if config.embodiment_tag not in all_stats:
        raise KeyError(
            f"Embodiment tag '{config.embodiment_tag}' not found in {stats_path}. "
            f"Available: {list(all_stats.keys())}"
        )
    _print_green(f"Loaded HEX normalization stats: {stats_path}")
    return all_stats[config.embodiment_tag]


def _q99_normalize(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    denom = np.maximum(q99 - q01, 1e-6)
    return 2.0 * (x - q01) / denom - 1.0


def _q99_unnormalize(
    x: np.ndarray,
    stats: dict[str, Any],
    clip: bool = False,
) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)
    out = np.asarray(x, dtype=np.float32).copy()
    dim = min(out.shape[-1], q01.shape[0])
    normalized = np.clip(out[..., :dim], -1.0, 1.0) if clip else out[..., :dim]
    raw = 0.5 * (normalized + 1.0) * (q99[:dim] - q01[:dim]) + q01[:dim]
    out[..., :dim] = np.where(mask[:dim], raw, out[..., :dim])
    return out


def _as_bt(state_value: np.ndarray) -> np.ndarray:
    arr = np.asarray(state_value, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1)
    if arr.ndim == 2:
        return arr.reshape(1, *arr.shape)
    return arr


def _build_hex_state(observation: dict[str, Any], stats: dict[str, Any] | None) -> np.ndarray:
    state = observation["state"]
    parts = [
        state["left_leg"],
        state["right_leg"],
        state["waist"],
        state["left_arm"],
        state["left_hand"],
        state["right_arm"],
        state["right_hand"],
        # HEX sonic checkpoints use state.others as projected gravity.
        state["others"],
    ]
    flat_state = np.concatenate([_as_bt(part) for part in parts], axis=-1).astype(np.float32)

    if stats is not None:
        expected_dim = len(stats["state"]["q01"])
        if flat_state.shape[-1] != expected_dim:
            raise ValueError(f"HEX state dim mismatch: built {flat_state.shape[-1]}, stats expect {expected_dim}")
        flat_state = _q99_normalize(flat_state, stats["state"]).astype(np.float32)
    return flat_state


def _extract_prompt(observation: dict[str, Any]) -> str:
    value = observation["language"]["annotation.human.task_description"]
    return str(value[0][0])


def _extract_ego_image(observation: dict[str, Any]) -> np.ndarray:
    image = observation["video"]["ego_view"]
    if image.ndim == 5:
        image = image[0, 0]
    return np.asarray(image)


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
) -> dict[str, Any] | None:
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] waiting for camera message", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] waiting for robot state message", flush=True)
        return None

    cam_img = camera_msg["images"]["ego_view"]
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    observation = {
        "video": {"ego_view": cam_img[np.newaxis, np.newaxis]},
        "state": {},
        "language": {"annotation.human.task_description": [[language_prompt]]},
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }
    observation = prepare_observation_for_eval(robot_model, observation)

    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["others"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]
    return observation


class HexSonicClient:
    def __init__(self, config: ClosedLoopConfig, stats: dict[str, Any] | None):
        self._client = ZMQPolicyClient(config.host, config.port, config.policy_timeout_ms)
        self._stats = stats
        self._tag = config.embodiment_tag
        self._clip_actions = config.clip_normalized_actions

    def ping(self) -> bool:
        return self._client.ping()

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        hex_observation = {
            "batch_images": [[_extract_ego_image(observation)]],
            "instructions": [_extract_prompt(observation)],
            "state": _build_hex_state(observation, self._stats),
            "tags": [self._tag],
        }
        action, _info = self._client.get_action(hex_observation, options={"do_sample": False})
        actions = np.asarray(action["actions"], dtype=np.float32)
        if self._stats is not None:
            over_bound = np.abs(actions[..., :64]) > 1.0
            if np.any(over_bound):
                print(
                    "[HEX] normalized motion_token exceeds [-1, 1]: "
                    f"max_abs={np.abs(actions[..., :64]).max():.3f}, "
                    f"ratio={over_bound.mean():.2%}, "
                    f"clip={self._clip_actions}",
                    flush=True,
                )
            actions = _q99_unnormalize(
                actions,
                self._stats["action"],
                clip=self._clip_actions,
            ).astype(np.float32)

        if actions.shape[-1] < 64:
            raise ValueError(f"HEX action dim must be >= 64 for SONIC, got {actions.shape}")

        processed = {"motion_token": actions[..., :64]}
        if actions.shape[-1] >= 66:
            processed["hand_binary_action"] = actions[..., 64:66]
        return processed

    def close(self) -> None:
        self._client.close()


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
) -> bytes:
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)
    if frame_index.ndim == 0:
        frame_index = frame_index.reshape(1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index[:1],
    }
    if left_hand_joints is not None:
        pose_data["left_hand_joints"] = np.asarray(left_hand_joints, dtype=np.float32).reshape(1, 7)
    if right_hand_joints is not None:
        pose_data["right_hand_joints"] = np.asarray(right_hand_joints, dtype=np.float32).reshape(1, 7)
    return pack_pose_message(pose_data, topic="pose", version=4)


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def _binary_to_hand_joints(binary_value: float, side: str) -> np.ndarray:
    if binary_value >= HAND_BINARY_CLOSE_THRESHOLD:
        return _compute_closed_hand_joints(side)
    return np.zeros(7, dtype=np.float32)


def _select_timestep(value: np.ndarray, index: int, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[min(index, arr.shape[0] - 1)]
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise ValueError(f"{name} must resolve to shape [{dim}], got {arr.shape}")
    return arr.astype(np.float32)


def _extract_hand_joint_actions(processed_action: dict[str, np.ndarray], index: int):
    if "left_hand_joints" in processed_action and "right_hand_joints" in processed_action:
        return (
            _select_timestep(processed_action["left_hand_joints"], index, 7, "left_hand_joints"),
            _select_timestep(processed_action["right_hand_joints"], index, 7, "right_hand_joints"),
        )

    hand_binary = _select_timestep(
        processed_action.get("hand_binary_action", np.zeros(2, dtype=np.float32)),
        index,
        2,
        "hand_binary_action",
    )
    return (
        _binary_to_hand_joints(float(hand_binary[0]), "L"),
        _binary_to_hand_joints(float(hand_binary[1]), "R"),
    )


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    while not stop_event.is_set():
        try:
            inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        busy_event.set()
        try:
            observation = prepare_obs_fn()
            if observation is None:
                continue
            t0 = time.monotonic()
            processed_action = inference_fn(observation)
            if processed_action is not None:
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        pass
                result_queue.put_nowait((processed_action, t0))
        except Exception as exc:
            print(f"Error in HEX inference worker: {exc}", flush=True)
            import traceback

            traceback.print_exc()
        finally:
            busy_event.clear()


def _sleep_remaining(t_start: float, period: float) -> None:
    remaining = period - (time.monotonic() - t_start)
    if remaining > 0:
        time.sleep(remaining)


def main(config: ClosedLoopConfig) -> None:
    _load_runtime_dependencies()

    stats = _load_stats(config)
    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    policy = HexSonicClient(config, stats)

    print(f"Connecting to HEX ZMQ server at {config.host}:{config.port}...")
    if policy.ping():
        _print_green("HEX server is reachable.")
    else:
        print("WARNING: HEX server ping failed. Inference may still fail until server is ready.")

    state_subscriber = ZMQStateSubscriber(host=config.state_zmq_host, port=config.state_zmq_port)
    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host,
        port=config.camera_port,
    )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    _print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port,
        host=config.keyboard_zmq_host,
    )
    telemetry = Telemetry(window_size=100)

    pause_loop = True
    cpp_loop_running = False
    cpp_mode = "OFF"
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.inference_rate
    zmq_frame_counter = 0
    language_prompt_ref = [config.prompt]

    def publish_initial_pose() -> None:
        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        zmq_socket.send(
            pack_latent_action_message(
                motion_token=LATENT_INITIAL_MOTION_TOKEN,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
        )
        _print_green("Sent latent initial pose via ZMQ")

    def send_cpp_control_command(start: bool, planner: bool = False) -> bool:
        nonlocal cpp_loop_running, cpp_mode
        try:
            zmq_socket.send(build_command_message(start=start, stop=not start, planner=planner))
            time.sleep(0.01)
            cpp_loop_running = start
            cpp_mode = "PLANNER" if start and planner else ("POSE" if start else "OFF")
            _print_green(f"Sent command: {'start' if start else 'stop'} ({cpp_mode})")
            return True
        except Exception as exc:
            print(f"Failed to send C++ command: {exc}", flush=True)
            return False

    def check_keyboard_input() -> None:
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                _print_green(f'Prompt changed: "{old_prompt}" -> "{new_prompt}"')
            return

        if key == "i":
            zmq_frame_counter = 0
            publish_initial_pose()
            cached_action_chunk = None
            action_chunk_index = 0
            last_inference_time = 0.0
            if cpp_loop_running and cpp_mode == "PLANNER":
                send_cpp_control_command(start=True, planner=False)
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} HEX policy loop", flush=True)
        elif key == "k":
            if cpp_loop_running:
                send_cpp_control_command(start=False, planner=(cpp_mode == "PLANNER"))
            else:
                send_cpp_control_command(start=True, planner=True)
                print("Press 'i' to send initial pose and switch to POSE mode.", flush=True)
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(f"Initial left hand: {'closed' if initial_pose_left_hand_closed else 'open'}")
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(f"Initial right hand: {'closed' if initial_pose_right_hand_closed else 'open'}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    busy_event = threading.Event()

    worker = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            stop_event,
            busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber,
                state_subscriber,
                robot_model,
                language_prompt_ref[0],
                log_errors=True,
            ),
            policy.get_action,
        ),
        daemon=True,
    )
    worker.start()

    loop_period = 1.0 / config.action_publish_rate
    _print_green(f'Starting HEX -> SONIC loop with prompt: "{language_prompt_ref[0]}"')

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                delay = time.monotonic() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    delay,
                    config.action_publish_rate,
                    config.action_horizon,
                )
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                _print_green(f"New HEX action chunk, latency={delay:.3f}s")
            except queue.Empty:
                pass

            if should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=busy_event.is_set(),
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            ):
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if pause_loop:
                time.sleep(0.2)
                continue

            with telemetry.timer("hex_sonic_loop"):
                if cached_action_chunk is None:
                    _sleep_remaining(t_start, loop_period)
                    continue

                motion_token = np.asarray(cached_action_chunk["motion_token"], dtype=np.float32)
                if motion_token.ndim == 3:
                    motion_token = motion_token[0]
                horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                current_idx = min(action_chunk_index, horizon - 1)
                if motion_token.ndim == 2:
                    motion_token = motion_token[current_idx]

                left_hand, right_hand = _extract_hand_joint_actions(cached_action_chunk, current_idx)
                frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                zmq_frame_counter += 1

                zmq_socket.send(
                    pack_latent_action_message(
                        motion_token=motion_token,
                        frame_index=frame_index,
                        left_hand_joints=left_hand,
                        right_hand_joints=right_hand,
                    )
                )
                if zmq_frame_counter % 50 == 0:
                    _print_green(
                        f"ZMQ sent frame={frame_index[0]}, token_shape={motion_token.shape}"
                    )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            if config.verbose_timing:
                telemetry.log_timing_info(context="HEX SONIC loop", threshold=0.0)
            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("HEX -> SONIC loop terminated by user")
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        policy.close()
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main(_parse_args())
