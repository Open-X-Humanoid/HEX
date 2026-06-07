import os
import time
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from requests.exceptions import SSLError, ConnectionError, ReadTimeout


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download remaining HEX training datasets from Hugging Face."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="your/dataset/path",
        help="Directory where the downloaded datasets will be saved.",
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

    parser.add_argument(
        "--num_retries",
        type=int,
        default=10,
        help="Number of retries for each dataset if network errors occur.",
    )

    return parser.parse_args()


def setup_env(args):
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1600"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "160"

    if args.proxy:
        os.environ["https_proxy"] = args.proxy
        os.environ["http_proxy"] = args.proxy
    else:
        os.environ.pop("https_proxy", None)
        os.environ.pop("http_proxy", None)


def download_with_retry(repo_id, local_dir, max_workers=8, num_retries=10):
    for attempt in range(num_retries):
        try:
            print(f"Downloading {repo_id} to {local_dir} ...")

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                max_workers=max_workers,
                resume_download=True,
                token=os.environ.get("HF_TOKEN", None),
            )

            print(f"[OK] Finished downloading {repo_id}")
            return

        except (SSLError, ConnectionError, ReadTimeout) as e:
            wait_time = min(60, 5 * (attempt + 1))
            print(f"[WARN] Attempt {attempt + 1}/{num_retries} failed for {repo_id}: {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to download {repo_id} after {num_retries} attempts.")


def main():
    args = parse_args()
    setup_env(args)

    datasets = {
        "eai_real_world": "Cognition2ActionLab/eai_real_world", # Cognition2ActionLab/eai_real_world contains our processed Humanoid-Everyday dataset
        # "humanoid_everyday_h1": "USC-PSI-Lab/Humanoid-Everyday-H1",
        # "humanoid_everyday_g1": "USC-PSI-Lab/Humanoid-Everyday-G1",       
        "leju_robot_box_storage_parcel": "RoboCOIN/leju_robot_box_storage_parcel",
        "leju_robot_hotel_services": "RoboCOIN/leju_robot_hotel_services_a",
        "leju_robot_moving_parts": "RoboCOIN/leju_robot_moving_parts_a",
        "leju_robot_part_placement": "RoboCOIN/leju_robot_part_placement",
        "leju_robot_pass_the_cleaner": "RoboCOIN/leju_robot_pass_the_cleaner_a",
    }

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for name, repo_id in datasets.items():
        local_dir = base_dir / name
        download_with_retry(
            repo_id=repo_id,
            local_dir=local_dir,
            max_workers=args.max_workers,
            num_retries=args.num_retries,
        )

    print("All remaining datasets downloaded successfully.")


if __name__ == "__main__":
    main()