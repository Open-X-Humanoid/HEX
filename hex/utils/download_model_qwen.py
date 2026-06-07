import os
import argparse
from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Qwen3-VL model checkpoints from Hugging Face."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="./pretrained_models",
        help="Directory where the downloaded Qwen models will be saved.",
    )

    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="https://huggingface.co",
        help='Hugging Face endpoint, e.g., "https://huggingface.co" or "https://hf-mirror.com".',
    )

    parser.add_argument(
        "--proxy",
        type=str,
        default="",
        help="HTTP/HTTPS proxy. Set to an empty string to disable proxy.",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel download workers.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

    if args.proxy:
        os.environ["https_proxy"] = args.proxy
        os.environ["http_proxy"] = args.proxy
    else:
        os.environ.pop("https_proxy", None)
        os.environ.pop("http_proxy", None)

    qwen_models = {
        "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
        # "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    }

    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)

    for name, repo_id in qwen_models.items():
        local_dir = os.path.join(base_dir, name)
        print(f"Downloading {name} to {local_dir} ...")

        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            max_workers=args.max_workers,
            resume_download=True,
            token=os.environ.get("HF_TOKEN", None),
        )

    print("All Qwen models downloaded successfully.")


if __name__ == "__main__":
    main()