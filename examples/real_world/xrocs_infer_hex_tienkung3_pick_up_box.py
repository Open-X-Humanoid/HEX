#!/usr/bin/env python3
"""Standalone XROCS client for the TianGong3 pick-up-box task using a HEX ZMQ policy.

This client packs raw robot observations and executes returned physical actions.
HEX.predict_action handles image processing, normalization and tensor operations;
no local checkpoint or torch installation is needed on the client.

Use --step-by-step to inspect and send one action per Enter press. Each inference
executes only the first --execute-step-count actions (20 by default).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Optional
from pathlib import Path

import numpy as np
import zmq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.model_server.tools import zmq_numpy as mnp

DEFAULT_INSTRUCTION = "Pick up the box"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 10093
REQUEST_TIMEOUT = 120.0
ROBOT_ACTION_DIM = 34
CONFIG_PATH = "/home/eai/Documents/configuration.toml"

EXECUTE_START_INDEX = 0
EXECUTE_STEP_COUNT = 20
AUTO_INTERVAL = 0.0
ARM_INTERPOLATION_TARGET_INDEX = 40

PROPRIO_HISTORY_LEN = 64
PROPRIO_HZ = 30.0
PROPRIO_WARMUP_MIN = 16

WALK_SETTLE_SECONDS = 0.2
WALK_VEL_SEND_MIN = 0.05
WALK_VEL_SEND_MINEST = 0.01
WALK_TRIGGER_COUNT = 20

MODEL_PROPRIO_DIM = 86
MODEL_ACTION_DIM = 34
CAMERA_KEY_IN_ROBOT_OBS = "head"
TASK_DESC = "Pick up a box, turn around and go forward, and place it in the box."


HOME_JOINTS = [
    0.508651, 0.015933, 0.077883, -2.033218, 0.113979, -0.061594, 0.012368,
    0.386156, -0.023887, 0.119413, -1.906566, -0.149266, -0.126736, -0.058374,
]
HOME_HEAD = [0.005754, 0.798733]
KEY_HEAD_Q_DEFAULT = [0.005754, 0.798733]
KEY_HEAD_W_DEFAULT = [0.006233, 0.798559]

LEFT_ARM_SLICE = slice(0, 7)
LEFT_HAND_IDX = 7
RIGHT_ARM_SLICE = slice(8, 15)
RIGHT_HAND_IDX = 15
HEAD_SLICE = slice(16, 18)
STAND_POSE_SLICE = slice(18, 24)
WAIST_SLICE = slice(24, 27)
MOVE_VEL_SLICE = slice(27, 33)
SKY_IDX = 33

PROPRIO_LEFT_ARM_SLICE = slice(0, 7)
PROPRIO_RIGHT_ARM_SLICE = slice(13, 20)
PROPRIO_HEAD_SLICE = slice(26, 28)
PROPRIO_STAND_SLICE = slice(40, 46)

CMD_OBS_COMPARE_GROUPS = (
    ("Larm", LEFT_ARM_SLICE, PROPRIO_LEFT_ARM_SLICE),
    ("Rarm", RIGHT_ARM_SLICE, PROPRIO_RIGHT_ARM_SLICE),
    ("head", HEAD_SLICE, PROPRIO_HEAD_SLICE),
    ("stand", STAND_POSE_SLICE, PROPRIO_STAND_SLICE),
)

GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0
WALK_MODE_NAME = "gotoHBWALK"
STAND_MODE_NAME = "gotoSTAND"

MISSING_STATE_WARNED = set()


class KeyHeadSwitcher:
    """Keyboard head switcher."""

    def __init__(self, pose_q: np.ndarray, pose_w: np.ndarray):
        self.pose_q = np.asarray(pose_q, dtype=np.float32).reshape(2)
        self.pose_w = np.asarray(pose_w, dtype=np.float32).reshape(2)
        self._lock = threading.Lock()
        self._mode = "model"
        self._stop = threading.Event()
        self._thread = None
        self._old_term = None

    def current(self) -> tuple[Optional[np.ndarray], str]:
        with self._lock:
            if self._mode == "q":
                return self.pose_q.copy(), "q"
            if self._mode == "w":
                return self.pose_w.copy(), "w"
            return None, "model"

    def start(self) -> None:
        if not sys.stdin.isatty():
            print("[WARN] stdin is not a TTY; keyboard head switch (a/q/w) disabled.")
            return
        self._thread = threading.Thread(target=self._loop, name="key-head", daemon=True)
        self._thread.start()
        print(
            "[client] key-head ON (start=MODEL head):\n"
            f"  a -> model head\n"
            f"  q -> {self.pose_q.tolist()}\n"
            f"  w -> {self.pose_w.tolist()}"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._restore_term()

    def _restore_term(self) -> None:
        if self._old_term is None:
            return
        try:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term)
        except Exception:
            pass
        self._old_term = None

    def _loop(self) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        try:
            self._old_term = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception as e:
            print(f"[WARN] key-head termios setup failed: {e}")
            return

        try:
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                key = ch.lower()
                if key == "a":
                    with self._lock:
                        self._mode = "model"
                    print("\n[key-head] a -> MODEL head", flush=True)
                elif key == "q":
                    with self._lock:
                        self._mode = "q"
                    print(f"\n[key-head] q -> {self.pose_q.tolist()}", flush=True)
                elif key == "w":
                    with self._lock:
                        self._mode = "w"
                    print(f"\n[key-head] w -> {self.pose_w.tolist()}", flush=True)
        finally:
            self._restore_term()


def _warn_missing_once(name, default_shape):
    if name not in MISSING_STATE_WARNED:
        print(f"[WARN] Missing state field: {name}. Use zeros with shape {default_shape}.")
        MISSING_STATE_WARNED.add(name)


def get_by_path(obs, path):
    if isinstance(path, str):
        if path in obs:
            return obs[path]
        cur = obs
        for k in path.split("/"):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                raise KeyError(path)
        return cur

    cur = obs
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            raise KeyError(path)
    return cur


def get_obs_array_by_candidates(obs, candidates, default_shape, name):
    for path in candidates:
        try:
            arr = get_by_path(obs, path)
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            pass

    _warn_missing_once(name, default_shape)
    return np.zeros(default_shape, dtype=np.float32)


def flatten_to_dim(arr, dim, name):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] == dim:
        return arr
    if arr.shape[0] > dim:
        print(f"[WARN] {name} dim {arr.shape[0]} > {dim}, truncate.")
        return arr[:dim]
    print(f"[WARN] {name} dim {arr.shape[0]} < {dim}, pad zeros.")
    out = np.zeros((dim,), dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def _state_field(obs, base_key, dim, name, extra_candidates=()):
    candidates = [
        f"puppet/{base_key}_align",
        ("puppet", f"{base_key}_align"),
        f"puppet/{base_key}",
        ("puppet", base_key),
        *extra_candidates,
    ]
    return flatten_to_dim(
        get_obs_array_by_candidates(obs, candidates, (dim,), f"puppet/{base_key}"),
        dim,
        name,
    )


def build_state_from_obs(obs):
    """Pack the 86 raw XROCS values; hand scaling/normalization happens in HEX."""
    left_arm = np.asarray(obs["arm_joints"]["left"], dtype=np.float32).reshape(7)
    left_hand = np.asarray(obs["hand_joints"]["left"], dtype=np.float32).reshape(6)
    right_arm = np.asarray(obs["arm_joints"]["right"], dtype=np.float32).reshape(7)
    right_hand = np.asarray(obs["hand_joints"]["right"], dtype=np.float32).reshape(6)
    # left_touch / right_touch are available in obs but intentionally unused here.
    head = np.asarray(obs["head_joints"], dtype=np.float32).reshape(2)
    leg_left = np.asarray(obs["leg_joints"]["left"], dtype=np.float32).reshape(6)
    leg_right = np.asarray(obs["leg_joints"]["right"], dtype=np.float32).reshape(6)
    stand_status = np.asarray(obs["stand_status"], dtype=np.float32).reshape(6)
    body_imu = np.asarray(obs["body_imu_data"], dtype=np.float32).reshape(40)

    state = np.concatenate(
        [
            left_arm,
            left_hand,
            right_arm,
            right_hand,
            head,
            leg_left,
            leg_right,
            stand_status,
            body_imu,
        ],
        axis=0,
    ).astype(np.float32)
    if state.shape[0] != MODEL_PROPRIO_DIM:
        raise RuntimeError(f"State dim mismatch: got {state.shape[0]}, expected {MODEL_PROPRIO_DIM}")
    return state


def get_camera_image_from_obs(obs):
    """Read the camera payload without decoding, resizing or changing colors."""
    if "images" not in obs:
        raise KeyError("obs does not contain 'images'.")
    if CAMERA_KEY_IN_ROBOT_OBS not in obs["images"]:
        raise KeyError(
            f"CAMERA_KEY_IN_ROBOT_OBS='{CAMERA_KEY_IN_ROBOT_OBS}' not found. "
            f"Available image keys: {list(obs['images'].keys())}"
        )
    image = obs["images"][CAMERA_KEY_IN_ROBOT_OBS]
    if isinstance(image, bytearray):
        return bytes(image)
    if isinstance(image, (bytes, np.ndarray)):
        return image
    raise TypeError(f"Expected encoded bytes or a camera array, got {type(image).__name__}")


class ProprioHistoryBuffer:
    def __init__(self, maxlen: int = PROPRIO_HISTORY_LEN):
        self.maxlen = int(maxlen)
        self._buf: Deque[np.ndarray] = deque(maxlen=self.maxlen)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()

    def push(self, proprio: np.ndarray) -> None:
        q = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if q.size != MODEL_PROPRIO_DIM:
            raise ValueError(f"proprio must be {MODEL_PROPRIO_DIM}-D, got {q.size}")
        with self._lock:
            self._buf.append(q.copy())

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def as_array(self) -> np.ndarray:
        with self._lock:
            if not self._buf:
                raise RuntimeError("proprio history empty")
            return np.stack(list(self._buf), axis=0)


def start_proprio_sampler(
    get_obs_fn: Callable[[], Any],
    buffer: ProprioHistoryBuffer,
    obs_lock: threading.Lock,
    stop_event: threading.Event,
    hz: float = PROPRIO_HZ,
) -> threading.Thread:
    def _loop() -> None:
        dt = 1.0 / float(hz)
        while not stop_event.is_set():
            t0 = time.perf_counter()
            try:
                with obs_lock:
                    obs = get_obs_fn()
                    state = build_state_from_obs(obs)
                buffer.push(state)
            except Exception as exc:
                if not stop_event.is_set():
                    print(f"[proprio-sampler] warn: {exc}")
                    time.sleep(dt)
                    continue
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    thread = threading.Thread(target=_loop, name="xrocs-proprio-sampler", daemon=True)
    thread.start()
    return thread


def warmup_proprio_buffer(buffer: ProprioHistoryBuffer, need: int, timeout_s: float) -> None:
    need = max(1, int(need))
    t0 = time.time()
    while len(buffer) < need:
        if time.time() - t0 > timeout_s:
            break
        time.sleep(0.01)
    print(f"[proprio] warmup done: buffer={len(buffer)} (need>={need})")


def build_model_obs_from_robot_obs(obs, proprio_history: Optional[np.ndarray] = None):
    primary_img = get_camera_image_from_obs(obs)
    state = build_state_from_obs(obs)
    model_obs = {
        "task_description": TASK_DESC,
        "primary_image": primary_img,
        "proprio": state.astype(np.float32),
    }
    if proprio_history is not None:
        hist = np.asarray(proprio_history, dtype=np.float32)
        if hist.ndim != 2 or hist.shape[-1] != MODEL_PROPRIO_DIM:
            raise ValueError(f"proprio_history must be [T,{MODEL_PROPRIO_DIM}], got {hist.shape}")
        model_obs["proprio_history"] = hist
        # History is diagnostic only for this history0 policy. Keep the current
        # state paired with the image from the same robot observation.
    return model_obs, state


def dump_obs_tree(obs, prefix="", depth=0, max_depth=5):
    if depth > max_depth:
        return
    if isinstance(obs, dict):
        for k, v in obs.items():
            p = f"{prefix}/{k}" if prefix else str(k)
            if isinstance(v, dict):
                print(f"  {p}/")
                dump_obs_tree(v, p, depth + 1, max_depth)
            elif isinstance(v, (bytes, bytearray)):
                print(f"  {p}  <bytes len={len(v)}>")
            else:
                try:
                    arr = np.asarray(v)
                    if arr.dtype == object:
                        print(f"  {p}  type={type(v).__name__}")
                    else:
                        print(
                            f"  {p}  shape={arr.shape} dtype={arr.dtype} "
                            f"vals={np.round(arr.reshape(-1)[:8].astype(np.float64), 4)}"
                        )
                except Exception:
                    print(f"  {p}  type={type(v).__name__}")


def print_proprio_debug(state, name="xrocs proprio/state"):
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    print("=" * 60)
    print(f"{name} dim={state.shape} dtype={state.dtype}")
    print(
        f"min/max/mean/std: {state.min():.6f} / {state.max():.6f} / "
        f"{state.mean():.6f} / {state.std():.6f}"
    )
    print("arm_left   [0:7]   :", np.round(state[0:7], 4))
    print("hand_left  [7:13]  :", np.round(state[7:13], 4))
    print("arm_right  [13:20] :", np.round(state[13:20], 4))
    print("hand_right [20:26] :", np.round(state[20:26], 4))
    print("head       [26:28] :", np.round(state[26:28], 4))
    print("leg_left   [28:34] :", np.round(state[28:34], 4))
    print("leg_right  [34:40] :", np.round(state[34:40], 4))
    print("stand      [40:46] :", np.round(state[40:46], 4))
    print("body_imu   [46:86] :", np.round(state[46:86], 4))
    print("=" * 60)


def print_cmd_obs_diff(action_34: np.ndarray, proprio: np.ndarray, *, index: int, fixed_head: Optional[np.ndarray] = None) -> None:
    cmd = np.asarray(action_34, dtype=np.float32).copy().reshape(-1)
    if cmd.shape[0] != MODEL_ACTION_DIM:
        print(f"[cmd-obs {index}] skip: action dim={cmd.shape[0]}, expected {MODEL_ACTION_DIM}")
        return
    if fixed_head is not None:
        cmd[HEAD_SLICE] = np.asarray(fixed_head, dtype=np.float32).reshape(2)

    obs = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if obs.shape[0] != MODEL_PROPRIO_DIM:
        print(f"[cmd-obs {index}] skip: proprio dim={obs.shape[0]}, expected {MODEL_PROPRIO_DIM}")
        return

    parts: list[str] = []
    total_sq = 0.0
    for name, a_sl, p_sl in CMD_OBS_COMPARE_GROUPS:
        delta = cmd[a_sl] - obs[p_sl]
        abs_delta = np.abs(delta)
        l2 = float(np.linalg.norm(delta))
        max_i = int(np.argmax(abs_delta))
        max_abs = float(abs_delta[max_i])
        total_sq += float(np.dot(delta, delta))
        parts.append(
            f"{name}:L2={l2:.4f} max={max_abs:.4f}@i{max_i}"
            f"(cmd={float(cmd[a_sl][max_i]):.4f},obs={float(obs[p_sl][max_i]):.4f})"
        )

    print(f"[cmd-obs {index}] total_L2={total_sq ** 0.5:.4f} | " + " ".join(parts))


def decode_sky_raw(action_34) -> float:
    return float(np.asarray(action_34, dtype=np.float32).reshape(-1)[SKY_IDX])


def move_velocity_norm(action_34) -> float:
    v = np.asarray(action_34, dtype=np.float32).reshape(-1)[MOVE_VEL_SLICE]
    return float(np.linalg.norm(v))


def interpolate_arm_actions(
    previous_action,
    actions,
    *,
    start_index: int,
    step_count: int,
    target_index: int = ARM_INTERPOLATION_TARGET_INDEX,
) -> np.ndarray:
    """Interpolate only both arms; keep every other model action unchanged."""
    all_actions = np.asarray(actions, dtype=np.float32)
    previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if all_actions.ndim != 2 or all_actions.shape[1] != MODEL_ACTION_DIM:
        raise ValueError(f"Expected actions with shape [T, {MODEL_ACTION_DIM}], got {all_actions.shape}")
    if previous.shape != (MODEL_ACTION_DIM,):
        raise ValueError(f"Expected previous action shape ({MODEL_ACTION_DIM},), got {previous.shape}")

    end_index = min(start_index + step_count, len(all_actions))
    result = all_actions[start_index:end_index].copy()
    if not len(result):
        return result

    target_index = min(max(int(target_index), 0), len(all_actions) - 1)
    target = all_actions[target_index]
    # Match the reference interpolation: execute intermediate points and leave
    # the exact target for the following prediction/execution cycle.
    weights = np.linspace(0.0, 1.0, len(result) + 2, dtype=np.float32)[1:-1, None]
    for arm_slice in (LEFT_ARM_SLICE, RIGHT_ARM_SLICE):
        result[:, arm_slice] = (
            previous[arm_slice][None, :]
            + weights * (target[arm_slice][None, :] - previous[arm_slice][None, :])
        )
    return result


def update_walk_trigger_streak(streak: int, velocity_norm: float, threshold: float) -> int:
    """Count consecutive actions above the WALK velocity threshold."""
    return streak + 1 if velocity_norm > threshold else 0


def action_to_robot_action_dict(
    action_34,
    *,
    walking: bool = True,
    fixed_head=None,
    send_head: bool = True,
    walk_vel_send_min: float = WALK_VEL_SEND_MINEST,
):
    a = np.asarray(action_34, dtype=np.float32).copy().reshape(-1)
    if a.shape[0] != MODEL_ACTION_DIM:
        raise ValueError(f"Action dim mismatch: got {a.shape[0]}, expected {MODEL_ACTION_DIM}")
    if fixed_head is not None:
        a[HEAD_SLICE] = np.asarray(fixed_head, dtype=np.float32).reshape(2)

    left_gripper = np.clip(a[LEFT_HAND_IDX], GRIPPER_MIN, GRIPPER_MAX)
    right_gripper = np.clip(a[RIGHT_HAND_IDX], GRIPPER_MIN, GRIPPER_MAX)
    action_dict = {
        "arm": {
            "position": {
                "left": np.asarray(a[LEFT_ARM_SLICE], dtype=np.float32),
                "right": np.asarray(a[RIGHT_ARM_SLICE], dtype=np.float32),
            }
        },
        "hand": {
            "position": {
                "left": np.asarray([left_gripper], dtype=np.float32),
                "right": np.asarray([right_gripper], dtype=np.float32),
            }
        },
        "locomotion": {},
    }
    if send_head:
        action_dict["head"] = {"position": np.asarray(a[HEAD_SLICE], dtype=np.float32)}
    if walking:
        v = np.asarray(a[MOVE_VEL_SLICE], dtype=np.float32)
        if float(np.linalg.norm(v)) >= float(walk_vel_send_min):
            action_dict["locomotion"]["move_velocity"] = v
        else:
            action_dict["locomotion"]["move_velocity"] = np.zeros_like(v)
        action_dict["waist"] = {"position": {31: float(a[WAIST_SLICE][0])}}
    else:
        action_dict["locomotion"]["stand_pose"] = np.asarray(a[STAND_POSE_SLICE], dtype=np.float32)
    return action_dict


class JointInference:
    def __init__(self, robot_config=CONFIG_PATH):
        from xrocs.core.config_loader import ConfigLoader
        from xrocs.core.station_loader import StationLoader
        from xrocs.utils.logger.logger_loader import logger

        cfg_loader = ConfigLoader(robot_config)
        self.cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(self.cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()
        self.robot = self.robot_station.get_robot_handle()["robot"]
        self.hb_walk_enabled = False
        self._logger = logger
        print("init done")

    def prepare(self, init_head: bool = False, home_joints=None, home_head=None):
        from xrocs.common.data_type import Joints

        joints = HOME_JOINTS if home_joints is None else list(np.asarray(home_joints, dtype=np.float32).reshape(-1))
        head = HOME_HEAD if home_head is None else list(np.asarray(home_head, dtype=np.float32).reshape(-1))
        both_home = Joints(joints, num_of_dofs=14)
        self.robot.reach_target_joint(both_home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        time.sleep(2)
        if init_head:
            try:
                self.robot_station.step({"head": {"position": np.asarray(head, dtype=np.float32)}})
                print(f"head init -> {np.round(np.asarray(head, dtype=np.float32), 4).tolist()}")
                time.sleep(1)
            except Exception as e:
                print(f"[WARN] head init failed (continuing): {e}")
        self._logger.success("Resetting to data-start pose success!")

    def start_walk(self, dry_run: bool = False, settle: float = WALK_SETTLE_SECONDS):
        if self.hb_walk_enabled:
            return
        print("[mode] Switching to", WALK_MODE_NAME)
        if not dry_run:
            self.robot.locomotion.set_mode(WALK_MODE_NAME)
            time.sleep(max(float(settle), 0.0))
        self.hb_walk_enabled = True

    def start_stand(self):
        print("[mode] Switching to", STAND_MODE_NAME)
        self.robot.locomotion.set_mode(STAND_MODE_NAME)
        time.sleep(2.0)
        self.hb_walk_enabled = False

    def stop_robot(self, *, goto_stand: bool = True):
        if goto_stand:
            print("[mode] Switching to", STAND_MODE_NAME)
            try:
                self.robot.locomotion.set_mode(STAND_MODE_NAME)
                time.sleep(2.0)
            except Exception as e:
                print(f"[WARN] set_mode({STAND_MODE_NAME}) failed: {e}")
        else:
            print("[mode] shutdown: stay WALK (no gotoSTAND)")
        self.hb_walk_enabled = False


class HexZMQPolicy:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        ckpt_path: Optional[str] = None,
        instruction: str,
        unnorm_key: Optional[str],
        tag: Optional[str],
        timeout: float = REQUEST_TIMEOUT,
    ):
        # Kept for CLI/API compatibility. Statistics come from the server's
        # loaded checkpoint, so this path never needs to exist on the client.
        del ckpt_path
        self.instruction = [instruction]
        self.state_dim = MODEL_PROPRIO_DIM
        self.action_dim = MODEL_ACTION_DIM
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
        self.socket.connect(f"tcp://{host}:{port}")
        try:
            metadata = self._request({"endpoint": "metadata"})
            if "xrocs_tiangong3" not in metadata.get("observation_formats", []):
                raise RuntimeError("Server does not support raw XROCS observations; restart the updated HEX server")
            keys = metadata["norm_stats_keys"]
            if unnorm_key is None:
                if tag in keys:
                    unnorm_key = tag
                elif len(keys) == 1:
                    unnorm_key = keys[0]
                else:
                    raise ValueError(f"Pass --unnorm-key to select server statistics from {keys}")
            if unnorm_key not in keys:
                raise ValueError(f"--unnorm-key '{unnorm_key}' not in server statistics {keys}")
            self.unnorm_key = unnorm_key
            self.tags = [tag if tag is not None else unnorm_key]
            if self.tags[0] not in metadata["embodiment_tags"]:
                raise ValueError(f"Unknown server embodiment tag: {self.tags[0]}")
            if metadata["action_dims"][self.tags[0]] != self.action_dim:
                raise ValueError(f"Robot requires {self.action_dim}-D actions")
        except Exception:
            self.close()
            raise
        print(f"connected to HEX ZMQ server: {metadata}")
        print(
            f"unnorm_key={self.unnorm_key}, tag={self.tags[0]}, "
            f"raw_state_dim={self.state_dim}, action_dim={self.action_dim}; "
            "pre/postprocessing runs in HEX.predict_action"
        )

    def _request(self, payload: dict) -> object:
        self.socket.send(mnp.packb(payload))
        response = mnp.unpackb(self.socket.recv(), raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(response["error"])
        return response

    def request_actions(self, model_obs: dict, state: np.ndarray, timeout: float = REQUEST_TIMEOUT) -> np.ndarray:
        del timeout
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"Expected a finite {self.state_dim}-D raw state, got {state.shape}")
        request = {
            "endpoint": "get_action",
            "data": {
                "observation": {
                    "batch_images": [[model_obs["primary_image"]]],
                    "instructions": self.instruction,
                    "state": state,
                    "tags": self.tags,
                    "observation_format": "xrocs_tiangong3",
                    "unnorm_key": self.unnorm_key,
                }
            },
        }

        t0 = time.time()
        response = self._request(request)
        latency = time.time() - t0
        if not isinstance(response, (list, tuple)) or not response:
            raise RuntimeError(f"unexpected server response: {response}")

        if response[0].get("actions_are_normalized") is not False:
            raise RuntimeError("Expected physical actions from server, received normalized or unmarked actions")
        actions = np.asarray(response[0]["actions"], dtype=np.float32)
        if actions.ndim != 2 or not len(actions) or actions.shape[-1] != self.action_dim:
            raise RuntimeError(f"Server returned invalid action shape: {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("Server returned non-finite actions")

        print(f"request latency: {latency * 1000:.0f} ms, actions {actions.shape}")
        return np.asarray(actions, dtype=np.float32)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="HEX ZMQ server host; do not use 0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--ckpt-path", default=None, help="Compatibility option; preprocessing uses the checkpoint loaded by the server.")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--robot-config", default=CONFIG_PATH)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true", help="Only infer and print; skip robot reset, mode changes and action sends.")
    parser.add_argument("--step-by-step", action="store_true", help="Wait for Enter before each action; q + Enter exits.")
    parser.add_argument("--once", action="store_true", help="Run only one action chunk.")
    parser.add_argument("--execute-start-index", type=int, default=EXECUTE_START_INDEX)
    parser.add_argument("--execute-step-count", type=int, default=EXECUTE_STEP_COUNT, help="Number of time steps per prediction chunk (each action remains 34-D).")
    parser.add_argument(
        "--arm-interpolation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Interpolate only the left/right arm joints toward prediction index "
            f"{ARM_INTERPOLATION_TARGET_INDEX}; all other action fields remain unchanged."
        ),
    )
    parser.add_argument("--auto-interval", type=float, default=AUTO_INTERVAL)
    parser.add_argument("--print-proprio", action="store_true")
    parser.add_argument("--print-cmd-obs-diff", action="store_true")
    parser.add_argument("--no-reset", action="store_true", help="Skip the initial robot pose reset and policy reset.")
    parser.add_argument("--init-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--send-head", action="store_true", help="Kept for compatibility; model head is sent by default.")
    parser.add_argument("--lock-start-head", action="store_true")
    parser.add_argument("--fixed-head", type=float, nargs=2, default=None, metavar=("H0", "H1"))
    parser.add_argument("--key-head", action="store_true")
    parser.add_argument("--key-head-w", type=float, nargs=2, default=KEY_HEAD_W_DEFAULT, metavar=("H0", "H1"))
    parser.add_argument("--dump-obs", action="store_true")
    parser.add_argument("--proprio-history-len", type=int, default=PROPRIO_HISTORY_LEN)
    parser.add_argument("--proprio-hz", type=float, default=PROPRIO_HZ)
    parser.add_argument("--proprio-warmup-min", type=int, default=PROPRIO_WARMUP_MIN)
    parser.add_argument("--no-proprio-sampler", action="store_true")
    parser.add_argument("--no-auto-walk", action="store_true")
    parser.add_argument("--no-shutdown-stand", action="store_true")
    parser.add_argument("--walk-settle", type=float, default=WALK_SETTLE_SECONDS)
    parser.add_argument("--walk-vel-send-min", type=float, default=WALK_VEL_SEND_MIN)
    return parser


def _resolve_head_control(args):
    fixed_head = None
    key_head = None
    if args.key_head:
        pose_q = np.asarray(
            args.fixed_head if args.fixed_head is not None else KEY_HEAD_Q_DEFAULT,
            dtype=np.float32,
        ).reshape(2)
        pose_w = np.asarray(args.key_head_w, dtype=np.float32).reshape(2)
        key_head = KeyHeadSwitcher(pose_q, pose_w)
        print(f"[client] key-head: start=MODEL; q={pose_q.tolist()} w={pose_w.tolist()} a=MODEL")
    elif args.fixed_head is not None:
        fixed_head = np.asarray(args.fixed_head, dtype=np.float32).reshape(2)
        print(f"[client] FIXED head -> {fixed_head.tolist()} (model head ignored at execute)")
    elif args.lock_start_head:
        fixed_head = np.asarray(HOME_HEAD, dtype=np.float32).reshape(2)
        print(f"[client] execute head = state[0] {fixed_head.tolist()} (locked)")
    else:
        print("[client] execute head = model action")
    return fixed_head, key_head


def wait_for_action_enter(index: int) -> bool:
    while True:
        command = input(f"[step] Press ENTER to send action {index}, or q + ENTER to quit: ").strip().lower()
        if command == "":
            return True
        if command == "q":
            return False
        print("Action not sent. Press Enter to send, or q + Enter to quit.")


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.step_by_step and args.key_head:
        parser.error("--step-by-step and --key-head both read stdin; use --fixed-head for step debugging")
    if args.execute_start_index < 0 or args.execute_step_count <= 0:
        parser.error("--execute-start-index must be >= 0 and --execute-step-count must be > 0")
    if args.proprio_history_len <= 0 or args.proprio_hz <= 0:
        parser.error("--proprio-history-len and --proprio-hz must be > 0")

    robot_client = JointInference(robot_config=args.robot_config)
    if args.dump_obs:
        print("\n================ get_obs() structure ================")
        obs = robot_client.robot_station.get_obs()
        dump_obs_tree(obs)
        print("=====================================================")
        print("(--dump-obs) done, exiting without moving the robot.")
        return

    policy = HexZMQPolicy(
        host=args.host,
        port=args.port,
        ckpt_path=args.ckpt_path,
        instruction=args.instruction,
        unnorm_key=args.unnorm_key,
        tag=args.tag,
        timeout=args.timeout,
    )

    fixed_head, key_head = _resolve_head_control(args)

    proprio_buf = ProprioHistoryBuffer(maxlen=args.proprio_history_len)
    obs_lock = threading.Lock()
    stop_sampler = threading.Event()
    sampler_thread = None
    warmup_need = min(args.proprio_history_len, max(args.proprio_warmup_min, 1))
    auto_walk_enabled = not bool(args.no_auto_walk)
    walking = False
    walk_trigger_streak = 0
    robot_commands_started = False
    previous_executed_action = None

    try:
        # Validate observations before any robot motion.
        initial_obs = robot_client.robot_station.get_obs()
        build_model_obs_from_robot_obs(initial_obs)
        if args.dry_run:
            print("[dry-run] skipping robot reset, mode changes and action sends.")
        else:
            operation = "start inference" if args.no_reset else "reset robot pose and start inference"
            input(f"Press ENTER to {operation}...")
            robot_commands_started = True
            robot_client.start_stand()
            if not args.no_reset:
                print(f"Reset robot to data-start pose (arms{' + head' if args.init_head else ' only'}) ...")
                robot_client.prepare(init_head=args.init_head, home_head=fixed_head)

        if not args.no_reset:
            policy.reset()
        if args.step_by_step:
            print("[step] Every predicted action requires Enter; q + Enter exits.")
        if key_head is not None:
            key_head.start()

        if not args.no_proprio_sampler:
            sampler_thread = start_proprio_sampler(
                get_obs_fn=robot_client.robot_station.get_obs,
                buffer=proprio_buf,
                obs_lock=obs_lock,
                stop_event=stop_sampler,
                hz=args.proprio_hz,
            )
            warmup_timeout = warmup_need / max(args.proprio_hz, 1e-3) + 2.0
            print(f"[proprio] sampler {args.proprio_hz} Hz, ring={args.proprio_history_len}, warmup>={warmup_need}")
            warmup_proprio_buffer(proprio_buf, need=warmup_need, timeout_s=warmup_timeout)
        else:
            print(f"[proprio] no background sampler; main-thread warmup {warmup_need} frames @ {args.proprio_hz} Hz")
            dt = 1.0 / max(args.proprio_hz, 1e-3)
            while len(proprio_buf) < warmup_need:
                with obs_lock:
                    obs = robot_client.robot_station.get_obs()
                    proprio_buf.push(build_state_from_obs(obs))
                time.sleep(dt)
            print(f"[proprio] warmup done: buffer={len(proprio_buf)}")

        if auto_walk_enabled:
            print(
                "[client] STAND: switch to gotoHBWALK after "
                f"{WALK_TRIGGER_COUNT} consecutive actions with "
                f"||move_velocity|| > {args.walk_vel_send_min:g}; sky unused"
            )
        else:
            print("[client] --no-auto-walk: stay STAND, send stand_pose only")

        while True:
            loop_t0 = time.time()
            obs_t0 = time.time()
            with obs_lock:
                obs = robot_client.robot_station.get_obs()
                state_now = build_state_from_obs(obs)
            proprio_buf.push(state_now)

            hist = proprio_buf.as_array()
            model_obs, state = build_model_obs_from_robot_obs(obs, proprio_history=hist)

            if args.print_proprio:
                print_proprio_debug(state, name="HEX proprio")
                print(f"[proprio] history shape={hist.shape}")

            obs_ms = (time.time() - obs_t0) * 1000.0
            print("\n================ Send obs to HEX server ================")
            print("obs keys:", list(obs.keys()))
            print("state shape:", state.shape)
            print("proprio_history shape:", model_obs["proprio_history"].shape)
            print("primary_image (raw):", type(model_obs["primary_image"]).__name__, getattr(model_obs["primary_image"], "shape", None))
            print("instruction:", args.instruction)

            try:
                req_t0 = time.time()
                actions = policy.request_actions(model_obs, state, timeout=args.timeout)
                req_ms = (time.time() - req_t0) * 1000.0
            except Exception as e:
                print("Failed to request actions:", e)
                raise

            total_ms = (time.time() - loop_t0) * 1000.0
            print(
                f"Timing: get_obs+build_obs={obs_ms:.1f} ms | "
                f"zmq_roundtrip+server={req_ms:.1f} ms | "
                f"client_total={total_ms:.1f} ms | proprio_buf={len(proprio_buf)}"
            )

            max_len = len(actions)
            if args.execute_start_index >= max_len:
                raise ValueError(f"--execute-start-index {args.execute_start_index} >= returned chunk length {max_len}")
            end_idx = min(args.execute_start_index + args.execute_step_count, max_len)
            print(f"Executing time steps [{args.execute_start_index}:{end_idx}] of {max_len}; each action is 34-D.")

            execution_actions = np.asarray(actions[args.execute_start_index:end_idx], dtype=np.float32).copy()
            if args.arm_interpolation:
                if previous_executed_action is None:
                    previous_executed_action = np.asarray(actions[args.execute_start_index], dtype=np.float32).copy()
                target_idx = min(ARM_INTERPOLATION_TARGET_INDEX, max_len - 1)
                execution_actions = interpolate_arm_actions(
                    previous_executed_action,
                    actions,
                    start_index=args.execute_start_index,
                    step_count=end_idx - args.execute_start_index,
                    target_index=target_idx,
                )
                print(
                    "[arm-interpolation] ON: "
                    f"arms interpolate toward action[{target_idx}]; all non-arm fields use their original time steps"
                )

            for i, action in enumerate(execution_actions, start=args.execute_start_index):
                action_34 = np.asarray(action, dtype=np.float32).reshape(-1)

                head_override = None
                head_tag = ""
                if key_head is not None:
                    head_override, lab = key_head.current()
                    head_tag = f"(KEY {lab})"
                elif fixed_head is not None:
                    head_override = fixed_head
                    head_tag = "(state[0] locked)" if args.fixed_head is None else "(FIXED)"

                print("\n================ Execute action ================")
                print("action index:", i)
                print("left arm     :", np.round(action_34[LEFT_ARM_SLICE], 4))
                print("left gripper :", round(float(action_34[LEFT_HAND_IDX]), 4))
                print("right arm    :", np.round(action_34[RIGHT_ARM_SLICE], 4))
                print("right gripper:", round(float(action_34[RIGHT_HAND_IDX]), 4))
                print("head model   :", np.round(action_34[HEAD_SLICE], 4))
                if head_override is not None:
                    print("head sent    :", np.round(head_override, 4), head_tag)
                elif key_head is not None:
                    print("head sent    :", np.round(action_34[HEAD_SLICE], 4), "(KEY model)")
                else:
                    print("head sent    :", np.round(action_34[HEAD_SLICE], 4), "(model)")

                sky_raw = decode_sky_raw(action_34)
                move_n = move_velocity_norm(action_34)
                switch_to_walk = False
                if auto_walk_enabled and not walking:
                    walk_trigger_streak = update_walk_trigger_streak(
                        walk_trigger_streak, move_n, float(args.walk_vel_send_min)
                    )
                    if walk_trigger_streak >= WALK_TRIGGER_COUNT:
                        walking = True
                        switch_to_walk = True
                print("sky[33]      :", round(sky_raw, 3), "(info only; unused)")
                print(f"||v||={move_n:.4f}  mode={'WALK' if walking else 'STAND'}")
                if auto_walk_enabled and not walking:
                    print(f"walk trigger : {walk_trigger_streak}/{WALK_TRIGGER_COUNT} consecutive")
                print("stand_pose   :", np.round(action_34[STAND_POSE_SLICE], 4), "(NOT sent)" if walking else "(SENT)")
                print(
                    "waist        :",
                    np.round(action_34[WAIST_SLICE], 4),
                    "(SENT {31: waist0})" if walking else "(NOT sent in STAND)",
                )
                send_walk = walking and move_n >= float(args.walk_vel_send_min)
                if not walking:
                    vel_tag = "(not sent in STAND)"
                elif send_walk:
                    vel_tag = "(SENT)"
                else:
                    vel_tag = f"(FILTERED ||v||<{args.walk_vel_send_min:g} -> zeros; stay WALK)"
                print("move_velocity:", np.round(action_34[MOVE_VEL_SLICE], 4), vel_tag)

                action_dict = action_to_robot_action_dict(
                    action_34,
                    walking=walking,
                    fixed_head=head_override,
                    walk_vel_send_min=float(args.walk_vel_send_min),
                )
                print("action_dict keys:", action_dict.keys())

                if args.print_cmd_obs_diff:
                    try:
                        with obs_lock:
                            obs_now = robot_client.robot_station.get_obs()
                            proprio_now = build_state_from_obs(obs_now)
                        print_cmd_obs_diff(action_34, proprio_now, index=i, fixed_head=head_override)
                    except Exception as e:
                        print(f"[cmd-obs {i}] failed: {e}")

                if args.dry_run:
                    print("[dry-run] not sending robot_station.step()")
                else:
                    if args.step_by_step and not wait_for_action_enter(i):
                        print("Stopped before sending the pending action.")
                        return
                    robot_commands_started = True
                    with obs_lock:
                        if switch_to_walk:
                            robot_client.start_walk(settle=float(args.walk_settle))
                        robot_client.robot_station.step(action_dict)
                    print(f"executed action index={i}")

                if args.arm_interpolation:
                    previous_executed_action = action_34.copy()

                if args.auto_interval > 0:
                    time.sleep(args.auto_interval)

            print("\nAction chunk finished.")
            if args.once:
                break

    except (KeyboardInterrupt, EOFError):
        print("Input interrupted; stopping client.")

    finally:
        if key_head is not None:
            key_head.stop()
        stop_sampler.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=2.0)
        policy.close()

        if not args.dry_run and robot_commands_started:
            try:
                robot_client.stop_robot(goto_stand=not bool(args.no_shutdown_stand))
            except Exception as e:
                print("Failed to stop robot cleanly:", e)

        print("Client closed.")


if __name__ == "__main__":
    main()
