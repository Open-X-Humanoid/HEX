import re
import json
import shutil
import argparse
from pathlib import Path


AGIBOT_DIR_NAME = "Agibot2UnitreeG1Retarget"
A2UG1_DATASET_DIR_NAME = "A2UG1_dataset"

EGO_KEY = "observation.images.ego_view"
WRIST_LEFT_KEY = "observation.images.wrist_left"
WRIST_RIGHT_KEY = "observation.images.wrist_right"

WRIST_VIDEO_DIRS = {
    WRIST_LEFT_KEY,
    WRIST_RIGHT_KEY,
}


def slugify_task(name: str) -> str:
    """Convert a raw task name to a normalized g1_xxx directory name."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return f"g1_{s}"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def infer_src_root_from_base_dir(base_dir: Path) -> Path:
    """
    Infer the extracted A2UG1 dataset directory from the download base directory.

    Expected layout:
        base_dir/
          Agibot2UnitreeG1Retarget/
            A2UG1_dataset.tar.gz.*
            A2UG1_dataset/
              <task folders>
    """
    return base_dir / AGIBOT_DIR_NAME / A2UG1_DATASET_DIR_NAME


def edit_modality_json(modality_path: Path, dry_run: bool = False) -> bool:
    """
    Patch meta/modality.json:
    1. Rename action.base_motion to action.others.
    2. Remove video.wrist_left and video.wrist_right.
    """
    if not modality_path.exists():
        print(f"[WARN] modality.json not found: {modality_path}")
        return False

    data = read_json(modality_path)
    changed = False

    # Rename action.base_motion to action.others.
    action = data.get("action")
    if isinstance(action, dict) and "base_motion" in action:
        if "others" not in action:
            action["others"] = action["base_motion"]
        action.pop("base_motion", None)
        data["action"] = action
        changed = True

    # Remove wrist camera entries from video modality.
    video = data.get("video")
    if isinstance(video, dict):
        for key in ["wrist_left", "wrist_right"]:
            if key in video:
                video.pop(key, None)
                changed = True
        data["video"] = video

    if changed:
        if dry_run:
            print(f"[DRY] would edit modality.json: {modality_path}")
        else:
            write_json(modality_path, data)
            print(f"[OK] edited modality.json: {modality_path}")

    return changed


def edit_info_json(info_path: Path, dry_run: bool = False) -> bool:
    """
    Patch meta/info.json:
    1. Convert video_path to a {video_key} template based on ego-view.
    2. Remove wrist camera features.
    """
    if not info_path.exists():
        print(f"[WARN] info.json not found: {info_path}")
        return False

    data = read_json(info_path)
    changed = False

    # Patch video_path.
    video_path = data.get("video_path")

    if isinstance(video_path, dict):
        ego_path = None

        # Common formats:
        #   "ego_view": "videos/.../observation.images.ego_view/..."
        #   "observation.images.ego_view": "videos/.../observation.images.ego_view/..."
        for key in ["ego_view", EGO_KEY]:
            if key in video_path and isinstance(video_path[key], str):
                ego_path = video_path[key]
                break

        if ego_path is not None:
            new_video_path = ego_path.replace(EGO_KEY, "{video_key}")
            data["video_path"] = new_video_path
            changed = True
        else:
            # Fallback: keep dict format, but remove wrist entries.
            for key in ["wrist_left", "wrist_right", WRIST_LEFT_KEY, WRIST_RIGHT_KEY]:
                if key in video_path:
                    video_path.pop(key, None)
                    changed = True
            data["video_path"] = video_path

    elif isinstance(video_path, str):
        new_video_path = video_path.replace(EGO_KEY, "{video_key}")
        if new_video_path != video_path:
            data["video_path"] = new_video_path
            changed = True

    # Remove wrist camera features.
    features = data.get("features")
    if isinstance(features, dict):
        for key in [WRIST_LEFT_KEY, WRIST_RIGHT_KEY, "wrist_left", "wrist_right"]:
            if key in features:
                features.pop(key, None)
                changed = True
        data["features"] = features

    if changed:
        if dry_run:
            print(f"[DRY] would edit info.json: {info_path}")
        else:
            write_json(info_path, data)
            print(f"[OK] edited info.json: {info_path}")

    return changed


def remove_wrist_videos(task_dir: Path, dry_run: bool = False) -> int:
    """Remove wrist camera video directories under one task directory."""
    videos_dir = task_dir / "videos"
    if not videos_dir.exists():
        return 0

    removed = 0

    # Use list(...) because directories may be deleted while iterating.
    for path in list(videos_dir.rglob("*")):
        if path.is_dir() and path.name in WRIST_VIDEO_DIRS:
            if dry_run:
                print(f"[DRY] would remove dir: {path}")
            else:
                shutil.rmtree(path)
                print(f"[OK] removed dir: {path}")
            removed += 1

    return removed


def process_task(task_dir: Path, dry_run: bool = False) -> dict:
    """
    Patch one task directory:
    1. meta/modality.json
    2. meta/info.json
    3. wrist video folders
    """
    stats = {
        "modality_changed": False,
        "info_changed": False,
        "wrist_dirs_removed": 0,
    }

    modality_path = task_dir / "meta" / "modality.json"
    info_path = task_dir / "meta" / "info.json"

    stats["modality_changed"] = edit_modality_json(modality_path, dry_run=dry_run)
    stats["info_changed"] = edit_info_json(info_path, dry_run=dry_run)
    stats["wrist_dirs_removed"] = remove_wrist_videos(task_dir, dry_run=dry_run)

    return stats


def import_a2ug1_tasks(
    src_root: Path,
    dst_root: Path,
    staging_root: Path,
    dry_run: bool = False,
    overwrite: bool = False,
):
    """
    Import A2UG1 tasks into the final eai_real_world directory.

    Workflow:
    1. Copy each source task to staging_root/g1_xxx.
    2. Patch metadata and remove wrist videos in the staging copy.
    3. Move the cleaned task to dst_root/g1_xxx.
    """
    task_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()])

    copied = 0
    processed = 0
    moved = 0
    skipped = 0

    if not dry_run:
        staging_root.mkdir(parents=True, exist_ok=True)
        dst_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(task_dirs)} A2UG1 task directories.")

    for task_src in task_dirs:
        new_name = slugify_task(task_src.name)
        task_tmp = staging_root / new_name
        task_dst = dst_root / new_name

        print(f"\n=== Processing {task_src.name} -> {new_name} ===")

        if task_dst.exists() and not overwrite:
            print(f"[SKIP] destination already exists: {task_dst}")
            skipped += 1
            continue

        if task_tmp.exists():
            if dry_run:
                print(f"[DRY] would remove existing staging dir: {task_tmp}")
            else:
                shutil.rmtree(task_tmp)
                print(f"[OK] removed existing staging dir: {task_tmp}")

        # Step 1: copy source task to staging directory.
        if dry_run:
            print(f"[DRY] would copy to staging: {task_src} -> {task_tmp}")
            print("[DRY] preview cleanup on source without modifying it:")
            process_task(task_src, dry_run=True)
            processed += 1
            continue

        shutil.copytree(task_src, task_tmp)
        copied += 1
        print(f"[OK] copied to staging: {task_src} -> {task_tmp}")

        # Step 2: patch metadata and remove wrist videos in staging copy.
        process_task(task_tmp, dry_run=False)
        processed += 1

        # Step 3: move cleaned staging copy to final destination.
        if task_dst.exists():
            if overwrite:
                shutil.rmtree(task_dst)
                print(f"[OK] removed existing destination: {task_dst}")
            else:
                print(f"[WARN] destination exists, keep cleaned staging copy at: {task_tmp}")
                skipped += 1
                continue

        shutil.move(str(task_tmp), str(task_dst))
        moved += 1
        print(f"[OK] moved to final destination: {task_dst}")

    # Remove empty staging root if possible.
    if not dry_run and staging_root.exists():
        try:
            staging_root.rmdir()
            print(f"\n[OK] removed empty staging root: {staging_root}")
        except OSError:
            print(f"\n[WARN] staging root is not empty, kept at: {staging_root}")

    return {
        "copied": copied,
        "processed": processed,
        "moved": moved,
        "skipped": skipped,
        "total": len(task_dirs),
    }


def patch_existing_g1_tasks(dst_root: Path, dry_run: bool = False):
    """
    Patch existing g1_* tasks under dst_root in place.
    This is useful if some tasks have already been imported before.
    """
    task_dirs = sorted(
        [p for p in dst_root.iterdir() if p.is_dir() and p.name.startswith("g1_")]
    )

    print(f"Found {len(task_dirs)} existing g1_* tasks under {dst_root}")

    processed = 0
    for task_dir in task_dirs:
        print(f"\n=== Patching existing task: {task_dir.name} ===")
        process_task(task_dir, dry_run=dry_run)
        processed += 1

    return processed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import and normalize Agibot2UnitreeG1Retarget A2UG1 tasks."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help=(
            "Base directory used by download_dataset_agibot2g1.py. "
            "The script expects the extracted dataset at "
            "base_dir/Agibot2UnitreeG1Retarget/A2UG1_dataset."
        ),
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default=None,
        help=(
            "Source A2UG1 dataset root directory. "
            "If not set, it is inferred from --base_dir."
        ),
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        required=True,
        help="Destination eai_real_world dataset root directory.",
    )
    parser.add_argument(
        "--staging_root",
        type=str,
        default=None,
        help="Temporary staging directory. Default: dst_root/.tmp_a2ug1_import.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print actions without copying, editing, deleting, or moving files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination task directories.",
    )
    parser.add_argument(
        "--no_import",
        action="store_true",
        help="Do not import A2UG1 tasks.",
    )
    parser.add_argument(
        "--patch_existing_g1",
        action="store_true",
        help="Patch existing g1_* tasks under dst_root in place.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir)
    src_root = Path(args.src_root) if args.src_root else infer_src_root_from_base_dir(base_dir)
    dst_root = Path(args.dst_root)

    if args.staging_root is None:
        staging_root = dst_root / ".tmp_a2ug1_import"
    else:
        staging_root = Path(args.staging_root)

    print("Configuration:")
    print(f"  base_dir:          {base_dir}")
    print(f"  src_root:          {src_root}")
    print(f"  dst_root:          {dst_root}")
    print(f"  staging_root:      {staging_root}")
    print(f"  dry_run:           {args.dry_run}")
    print(f"  overwrite:         {args.overwrite}")
    print(f"  no_import:         {args.no_import}")
    print(f"  patch_existing_g1: {args.patch_existing_g1}")

    if not args.no_import:
        if not src_root.exists():
            raise FileNotFoundError(
                f"src_root not found: {src_root}\n\n"
                f"Please first download and extract the dataset with:\n"
                f"  base_dir={base_dir}\n"
                f"  python hex/utils/download_dataset_agibot2g1.py --base_dir ${{base_dir}}\n"
                f"  cd ${{base_dir}}/Agibot2UnitreeG1Retarget\n"
                f"  cat A2UG1_dataset.tar.gz.* | tar -xzf -"
            )

        stats = import_a2ug1_tasks(
            src_root=src_root,
            dst_root=dst_root,
            staging_root=staging_root,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

        print("\nImport summary:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    if args.patch_existing_g1:
        if not dst_root.exists():
            raise FileNotFoundError(f"dst_root not found: {dst_root}")

        processed = patch_existing_g1_tasks(
            dst_root=dst_root,
            dry_run=args.dry_run,
        )

        print("\nExisting g1_* patch summary:")
        print(f"  processed: {processed}")

    print("\nDone.")


if __name__ == "__main__":
    main()