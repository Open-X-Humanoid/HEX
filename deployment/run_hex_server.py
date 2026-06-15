from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import socket
import sys
from typing import Any

import msgpack_numpy as mnp
import numpy as np
import torch
import zmq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL_SERVER_PORT = 10093


@dataclass
class ServerConfig:
    """Configuration for running the HEX inference server."""

    model_path: str
    """Path to the HEX checkpoint, usually a pytorch_model.pt file."""

    host: str = "0.0.0.0"
    """Host address for the ZMQ server."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the ZMQ server."""

    device: str = "cuda"
    """Device used for model inference, e.g. cuda or cpu."""

    dtype: str = "bfloat16"
    """Inference dtype: float32, bfloat16, or float16."""

    metadata_env: str = "hex"
    """Small metadata tag sent to clients when they connect."""

    idle_timeout: int = -1
    """Idle timeout in seconds. Use -1 to keep the server alive indefinitely."""


def _parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Run a HEX model as a ZMQ policy server.")
    parser.add_argument(
        "--model_path",
        "--model-path",
        "--ckpt_path",
        "--ckpt-path",
        required=True,
        help="Path to HEX checkpoint",
    )
    parser.add_argument("--host", default=ServerConfig.host)
    parser.add_argument("--port", type=int, default=ServerConfig.port)
    parser.add_argument("--device", default=ServerConfig.device)
    parser.add_argument(
        "--dtype",
        default=ServerConfig.dtype,
        choices=["float32", "bfloat16", "float16"],
        help="Model inference dtype",
    )
    parser.add_argument("--metadata_env", default=ServerConfig.metadata_env)
    parser.add_argument("--idle_timeout", type=int, default=ServerConfig.idle_timeout)
    return ServerConfig(**vars(parser.parse_args()))


class ZMQHEXPolicyServer:
    def __init__(
        self,
        policy,
        metadata: dict[str, Any],
        host: str,
        port: int,
        idle_timeout: int = -1,
    ):
        self._policy = policy
        self._metadata = metadata
        self._host = host
        self._port = port
        self._idle_timeout = idle_timeout
        self._running = True
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://{host}:{port}")

    @staticmethod
    def _to_bytes(data: Any) -> bytes:
        return mnp.packb(data, default=mnp.encode)

    @staticmethod
    def _from_bytes(data: bytes) -> Any:
        return mnp.unpackb(data, object_hook=mnp.decode, raw=False)

    def _handle_get_action(self, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        observation = data.get("observation")
        if observation is None:
            raise ValueError("get_action request missing data['observation']")
        if not observation.get("tags"):
            raise ValueError("HEX get_action requires observation['tags']")

        output = self._policy.predict_action(
            **observation,
            **dict(data.get("options") or {}),
        )
        normalized_actions = np.asarray(output["normalized_actions"], dtype=np.float32)
        return {"actions": normalized_actions[0, :, :66]}, {"metadata": self._metadata}

    def _route(self, request: dict[str, Any]) -> Any:
        endpoint = request.get("endpoint", "get_action")
        if endpoint == "ping":
            return {"status": "ok", "message": "Server is running", "metadata": self._metadata}
        if endpoint == "metadata":
            return self._metadata
        if endpoint == "kill":
            self._running = False
            return {"status": "ok", "message": "Server stopping"}
        if endpoint == "get_action":
            return self._handle_get_action(request.get("data", {}))
        raise ValueError(f"Unknown endpoint: {endpoint}")

    def serve_forever(self) -> None:
        logging.info("ZMQ server listening on tcp://%s:%s", self._host, self._port)
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        timeout_ms = 1000 if self._idle_timeout > 0 else None
        idle_start = None

        while self._running:
            events = dict(poller.poll(timeout_ms))
            if self._socket not in events:
                if self._idle_timeout > 0:
                    import time

                    now = time.monotonic()
                    idle_start = now if idle_start is None else idle_start
                    if now - idle_start >= self._idle_timeout:
                        logging.info("Idle timeout (%ss) reached, shutting down server.", self._idle_timeout)
                        break
                continue

            idle_start = None
            try:
                request = self._from_bytes(self._socket.recv())
                response = self._route(request)
                self._socket.send(self._to_bytes(response))
            except Exception as exc:
                logging.exception("Error handling ZMQ request")
                self._socket.send(self._to_bytes({"error": str(exc)}))

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


def _resolve_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "float32":
        return None
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_policy(config: ServerConfig):
    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    logging.info("Loading HEX model from %s", model_path)
    from hex.model.framework.base_framework import baseframework

    policy = baseframework.from_pretrained(str(model_path))

    # Keep dtype selection explicit so the caller can trade memory for precision.
    torch_dtype = _resolve_dtype(config.dtype)
    if torch_dtype is not None:
        policy = policy.to(torch_dtype)

    return policy.to(config.device).eval()


def main(config: ServerConfig) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    logging.info("Starting HEX inference server")
    logging.info("  Model path: %s", config.model_path)
    logging.info("  Device: %s", config.device)
    logging.info("  Dtype: %s", config.dtype)
    logging.info("  Host: %s", config.host)
    logging.info("  Port: %s", config.port)
    logging.info("  Hostname/IP: %s / %s", hostname, local_ip)

    policy = load_policy(config)
    server = ZMQHEXPolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
        metadata={
            "env": config.metadata_env,
            "model_path": config.model_path,
            "device": config.device,
            "dtype": config.dtype,
            "action_dim": 66,
            "actions_are_normalized": True,
            "requires_tags": True,
        },
        idle_timeout=config.idle_timeout,
    )

    logging.info("Server ready; listening on %s:%s", config.host, config.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down server")
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(_parse_args())
