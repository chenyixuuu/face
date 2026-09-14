#!/usr/bin/env python3
"""Render high-confidence validation mistakes with true and predicted galleries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--errors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--thumb-size", type=int, default=180)
    parser.add_argument("--gallery-size", type=int, default=3)
    return parser.parse_args()


def open_thumb(path: Path, size: int) -> Image.Image:
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    except Exception:
        missing = Image.new("RGB", (size, size), "#3a1f1f")
        ImageDraw.Draw(missing).text((8, 8), "LOAD ERROR", fill="white")
        return missing


def identity_images(data_root: Path, identity: str, limit: int) -> list[Path]:
    directory = data_root / "validation" / identity
    return sorted(path for path in directory.iterdir() if path.is_file())[:limit]


def main() -> int:
    args = parse_args()
    with args.errors.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[: args.rows]
    if not rows:
        raise SystemExit("No error rows found")

    columns = 1 + args.gallery_size * 2
    label_height = 48
    header_height = 42
    cell = args.thumb_size
    canvas = Image.new(
        "RGB",
        (columns * cell, header_height + len(rows) * (cell + label_height)),
        "#111827",
    )
    draw = ImageDraw.Draw(canvas)
    headers = ["QUERY"] + [f"TRUE G{i + 1}" for i in range(args.gallery_size)] + [
        f"WRONG G{i + 1}" for i in range(args.gallery_size)
    ]
    for column, header in enumerate(headers):
        draw.text((column * cell + 8, 14), header, fill="#f9fafb")

    for row_index, row in enumerate(rows):
        y = header_height + row_index * (cell + label_height)
        query_path = args.data_root / row["query_path"]
        true_paths = identity_images(args.data_root, row["true_identity"], args.gallery_size)
        wrong_paths = identity_images(args.data_root, row["predicted_identity"], args.gallery_size)
        paths = [query_path, *true_paths, *wrong_paths]
        while len(paths) < columns:
            paths.append(Path("/missing"))
        for column, path in enumerate(paths[:columns]):
            canvas.paste(open_thumb(path, cell), (column * cell, y))
        label = (
            f"true={row['true_identity']}  predicted={row['predicted_identity']}  "
            f"sim={float(row['top1_similarity']):.3f}  margin={float(row['margin']):.3f}  "
            f"query={Path(row['query_path']).name}"
        )
        draw.rectangle((0, y + cell, columns * cell, y + cell + label_height), fill="#111827")
        draw.text((8, y + cell + 13), label, fill="#f9fafb")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    print(f"saved={args.output} rows={len(rows)} size={canvas.width}x{canvas.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
