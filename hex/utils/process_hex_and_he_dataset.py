import argparse
import shutil
from pathlib import Path


def is_task_dir(path: Path) -> bool:
    """Return True if the path looks like a dataset task directory."""
    if not path.is_dir():
        return False

    # Skip hidden/cache folders.
    if path.name.startswith("."):
        return False

    return True


def move_nested_eai(parent_dir: Path, nested_name: str = "eai_real_world", dry_run: bool = False):
    nested_dir = parent_dir / nested_name

    if not nested_dir.exists():
        raise FileNotFoundError(f"Nested directory not found: {nested_dir}")

    if not nested_dir.is_dir():
        raise NotADirectoryError(f"Nested path is not a directory: {nested_dir}")

    print(f"Parent directory: {parent_dir}")
    print(f"Nested directory: {nested_dir}")
    print(f"Dry run: {dry_run}")
    print()

    moved_count = 0

    for item in sorted(nested_dir.iterdir()):
        if not is_task_dir(item):
            print(f"[SKIP] non-task item: {item}")
            continue

        target = parent_dir / item.name

        if target.exists():
            raise FileExistsError(
                f"Target already exists: {target}\n"
                f"Please handle the conflict manually before moving."
            )

        print(f"[MOVE] {item} -> {target}")

        if not dry_run:
            shutil.move(str(item), str(target))

        moved_count += 1

    print()
    print(f"Moved task directories: {moved_count}")

    if dry_run:
        print(f"[DRY] would remove nested directory: {nested_dir}")
    else:
        print(f"[REMOVE] {nested_dir}")
        shutil.rmtree(nested_dir)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Move task folders from nested eai_real_world to its parent directory."
    )

    parser.add_argument(
        "--parent_dir",
        type=str,
        default="your/dataset/path",
        help="Parent directory that contains the nested eai_real_world folder.",
    )

    parser.add_argument(
        "--nested_name",
        type=str,
        default="eai_real_world",
        help="Name of the nested directory to flatten.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print actions without moving or deleting files.",
    )

    args = parser.parse_args()

    move_nested_eai(
        parent_dir=Path(args.parent_dir),
        nested_name=args.nested_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()