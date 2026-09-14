#!/usr/bin/env python3
"""Register one person from one or more photos, each containing exactly one face."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from project_paths import DEFAULT_DATA_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_registry(registry_path: Path) -> dict:
    if not registry_path.exists():
        return {"people": {}}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def save_registry(registry_path: Path, registry: dict) -> None:
    """Atomic replacement prevents failed writes from truncating the registry."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=registry_path.parent,
            prefix=".registry-", suffix=".tmp", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(registry, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, registry_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def next_person_id(registry: dict) -> str:
    max_number = 0
    for person_id in registry.get("people", {}):
        if person_id.startswith("person_"):
            try:
                max_number = max(max_number, int(person_id.split("_", 1)[1]))
            except ValueError:
                pass
    return f"person_{max_number + 1:03d}"


def register_images(name, image_paths, data_dir, *, app=None, analyzer=None):
    """Validate before writing; app/analyzer injection supports offline tests.

    Single-writer CLI: do not register concurrently with another CLI or the web app.
    """
    name = name.strip()
    if not name:
        raise ValueError("name must not be blank")
    data_dir = Path(data_dir).expanduser().resolve()
    registry_path = data_dir / "registry.json"
    registry = load_registry(registry_path)
    accepted, rejected, seen = [], [], set()
    for value in image_paths:
        source = Path(value).expanduser().resolve()
        reason = None
        if source in seen:
            reason = "duplicate image path"
        elif not source.is_file():
            reason = "image not found or not a regular file"
        elif source.suffix.lower() not in IMAGE_EXTENSIONS:
            reason = "unsupported image extension"
        seen.add(source)
        if reason is None:
            if analyzer is None:
                from face_demo import analyze_image, build_app
                analyzer = analyze_image
                if app is None:
                    app = build_app()
            try:
                result = analyzer(app, str(source))
            except (FileNotFoundError, ValueError) as error:
                reason = f"unreadable image: {error}"
            else:
                if result["face_count"] != 1 or result["face"] is None:
                    reason = f"expected exactly one face; detected {result['face_count']}"
                else:
                    embedding = np.asarray(result["face"]["embedding"], dtype=np.float32)
                    if (embedding.shape != (512,) or not np.isfinite(embedding).all()
                            or float(np.linalg.norm(embedding)) == 0):
                        reason = "invalid 512-dimensional embedding"
                    else:
                        accepted.append((source, embedding))
        if reason is not None:
            rejected.append({"image": str(source), "reason": reason})
    if not accepted:
        return {"person_id": None, "accepted": 0, "rejected": rejected}

    person_id = next_person_id(registry)
    person_dir = data_dir / "people" / person_id
    person_dir.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never overwrites an existing or orphaned person folder.
    person_dir.mkdir(exist_ok=False)
    try:
        raw_dir = person_dir / "raw"
        raw_dir.mkdir()
        photos = []
        for index, (source, _) in enumerate(accepted, start=1):
            target = raw_dir / f"{index:03d}_{source.name}"
            shutil.copy2(source, target)
            photos.append(str(target))
        embedding_path = person_dir / "embeddings.npy"
        np.save(embedding_path, np.stack([embedding for _, embedding in accepted]))
        registry.setdefault("people", {})[person_id] = {
            "name": name, "folder": str(person_dir), "photos": photos,
            "embedding_file": str(embedding_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": "insightface/buffalo_l",
        }
        save_registry(registry_path, registry)
    except Exception:
        # This exact folder was created exclusively by the current attempt.
        shutil.rmtree(person_dir)
        raise
    return {"person_id": person_id, "accepted": len(accepted), "rejected": rejected,
            "embedding_file": str(embedding_path), "registry": str(registry_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="display name, e.g. 张三")
    parser.add_argument("--image", required=True, nargs="+", action="extend", help="image paths; repeatable")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="data directory")
    args = parser.parse_args()
    result = register_images(args.name, args.image, args.data_dir)
    for rejection in result["rejected"]:
        print(f"rejected: {rejection['image']} — {rejection['reason']}")
    if result["person_id"] is None:
        print("registration failed: no valid single-face photos; registry unchanged")
        return 1
    print(f"registered: {result['person_id']}")
    print(f"name: {args.name.strip()}")
    print(f"accepted_photos: {result['accepted']}")
    print(f"rejected_photos: {len(result['rejected'])}")
    print("embedding_dim: 512")
    print(f"embedding_file: {result['embedding_file']}")
    print(f"registry: {result['registry']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
