import os
import time
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from requests.exceptions import SSLError, ConnectionError, ReadTimeout


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Agibot2UnitreeG1Retarget dataset from Hugging Face."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="your/dataset/path",
        help="Directory where the dataset will be saved.",
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
        help="Number of parallel download workers. Use 1 or 2 to avoid rate limits.",
    )

    parser.add_argument(
        "--num_retries",
        type=int,
        default=20,
        help="Number of retries when network errors or rate limits occur.",
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


def download_with_retry(repo_id, local_dir, max_workers=1, num_retries=20):
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

        except HfHubHTTPError as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                wait_time = 360
                print(f"[RATE LIMIT] Attempt {attempt + 1}/{num_retries}: {e}")
                print(f"Sleeping {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise

        except LocalEntryNotFoundError as e:
            wait_time = 360
            print(f"[CACHE MISS / NETWORK ERROR] Attempt {attempt + 1}/{num_retries}: {e}")
            print(f"Sleeping {wait_time} seconds before retrying...")
            time.sleep(wait_time)

        except (SSLError, ConnectionError, ReadTimeout) as e:
            wait_time = min(300, 30 * (attempt + 1))
            print(f"[NETWORK ERROR] Attempt {attempt + 1}/{num_retries}: {e}")
            print(f"Sleeping {wait_time} seconds before retrying...")
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to download {repo_id} after {num_retries} attempts.")


def main():
    args = parse_args()
    setup_env(args)

    repo_id = "l2aggle/Agibot2UnitreeG1Retarget"
    dataset_name = "Agibot2UnitreeG1Retarget"

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    local_dir = base_dir / dataset_name

    download_with_retry(
        repo_id=repo_id,
        local_dir=local_dir,
        max_workers=args.max_workers,
        num_retries=args.num_retries,
    )

    print("Dataset downloaded successfully.")


if __name__ == "__main__":
    main()