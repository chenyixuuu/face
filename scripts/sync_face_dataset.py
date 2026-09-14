#!/usr/bin/env python3
"""
Sync a flat folder of face photos into a by-person dataset folder.

This script is designed for the current learning workflow:
  - You keep adding photos into one source folder, e.g. ~/Desktop/人脸测试.
  - Photo filenames use a person prefix, e.g. zxy1.jpg, zxy2.jpg, hb1.jpg.
  - The script copies them into grouped folders, e.g. data/face_test_by_person/zxy/.
  - It does not move or delete the original photos.
  - It can be run repeatedly; existing copied files are skipped.

Usage:
  python scripts/sync_face_dataset.py \
    --source /Users/chenxi/Desktop/人脸测试 \
    --dest data/face_test_by_person
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

from project_paths import DEFAULT_GROUPED_DATASET, user_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}


def person_key_from_filename(path: Path) -> str:
    """
    Extract leading letters/Chinese chars as the person key.

    Examples:
      zxy1.jpg -> zxy
      hb4.jpg -> hb
      张三2.jpg -> 张三
    """
    match = re.match(r"([A-Za-z\u4e00-\u9fff]+)", path.stem)
    if match:
        return match.group(1)
    return "unknown"


def iter_image_files(source: Path):
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def copy_without_overwrite(source_file: Path, target_dir: Path) -> tuple[Path, bool]:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / source_file.name
    if target_file.exists():
        return target_file, False
    shutil.copy2(source_file, target_file)
    return target_file, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="flat source folder with photos")
    parser.add_argument("--dest", default=str(DEFAULT_GROUPED_DATASET), help="grouped dataset destination")
    args = parser.parse_args()

    source = user_path(args.source)
    dest = user_path(args.dest)

    if not source.exists():
        raise FileNotFoundError(f"source folder not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"source is not a folder: {source}")

    copied_count = 0
    skipped_count = 0
    grouped = defaultdict(list)

    for image_file in iter_image_files(source):
        person_key = person_key_from_filename(image_file)
        target_file, copied = copy_without_overwrite(image_file, dest / person_key)
        grouped[person_key].append(target_file.name)
        if copied:
            copied_count += 1
        else:
            skipped_count += 1

    print(f"source: {source}")
    print(f"dest: {dest}")
    print(f"copied: {copied_count}")
    print(f"skipped_existing: {skipped_count}")
    print("groups:")
    for person_key in sorted(grouped):
        print(f"- {person_key}: {len(grouped[person_key])} files")


if __name__ == "__main__":
    main()
