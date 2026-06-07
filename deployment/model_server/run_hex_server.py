from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import socket
import sys

import torch

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
    """Host address for the websocket server."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the websocket server."""

    device: str = "cuda"
    """Device used for model inference, e.g. cuda or cpu."""

    dtype: str = "bfloat16"
    """Inference dtype: float32, bfloat16, or float16."""

    metadata_env: str = "hex"
    """Small metadata tag sent to clients when they connect."""


def _parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Run a HEX model as a websocket policy server.")
    parser.add_argument("--model_path", "--ckpt_path", required=True, help="Path to HEX checkpoint")
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
    return ServerConfig(**vars(parser.parse_args()))


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

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    policy = load_policy(config)
    server = WebsocketPolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
        metadata={
            "env": config.metadata_env,
            "model_path": config.model_path,
            "device": config.device,
            "dtype": config.dtype,
        },
    )

    logging.info("Server ready; listening on %s:%s", config.host, config.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down server")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(_parse_args())
