#!/usr/bin/env python3
"""Analyze validation retrieval errors without consulting the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch

from analyze_gallery_sizes import adapt_all, build_model
from train_metric_adapter import load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gallery-size", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hardest", type=int, default=200)
    return parser.parse_args()


def sample_bin(size: int) -> str:
    if size <= 4:
        return "4"
    if size == 5:
        return "5"
    if size == 6:
        return "6"
    return "7+"


def main() -> int:
    args = parse_args()
    if args.gallery_size < 1:
        raise SystemExit("--gallery-size must be positive")
    device = torch.device(args.device)
    vectors, identities, paths = load_split(args.embeddings, "validation")
    adapted = adapt_all(build_model(args.checkpoint, device), vectors, device, args.batch_size)

    gallery_vectors: list[np.ndarray] = []
    gallery_ids: list[str] = []
    query_indices: list[int] = []
    identity_sizes: dict[str, int] = {}
    for identity in np.unique(identities):
        indices = np.where(identities == identity)[0]
        indices = indices[np.argsort(paths[indices])]
        if len(indices) <= args.gallery_size:
            continue
        centroid = adapted[indices[: args.gallery_size]].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        gallery_vectors.append(centroid.astype(np.float32))
        gallery_ids.append(str(identity))
        identity_sizes[str(identity)] = int(len(indices))
        query_indices.extend(int(index) for index in indices[args.gallery_size :])

    gallery = np.stack(gallery_vectors)
    gallery_ids_array = np.asarray(gallery_ids)
    query = adapted[query_indices]
    query_ids = identities[query_indices].astype(str)
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    similarities, neighbours = index.search(query, 3)
    predicted = gallery_ids_array[neighbours[:, 0]]
    correct = predicted == query_ids
    margins = similarities[:, 0] - similarities[:, 1]

    by_bin: dict[str, list[int]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    for position, query_index in enumerate(query_indices):
        identity = str(query_ids[position])
        count = identity_sizes[identity]
        by_bin[sample_bin(count)].append(position)
        rows.append(
            {
                "query_path": str(paths[query_index]),
                "true_identity": identity,
                "predicted_identity": str(predicted[position]),
                "identity_images": count,
                "top1_similarity": float(similarities[position, 0]),
                "top2_similarity": float(similarities[position, 1]),
                "margin": float(margins[position]),
                "correct": bool(correct[position]),
            }
        )

    bin_metrics = {}
    for label in ("4", "5", "6", "7+"):
        positions = np.asarray(by_bin[label], dtype=np.int64)
        if not len(positions):
            continue
        bin_metrics[label] = {
            "queries": int(len(positions)),
            "top1": float(correct[positions].mean()),
            "mean_top1_similarity": float(similarities[positions, 0].mean()),
            "mean_margin": float(margins[positions].mean()),
        }

    wrong_identity_counts = Counter(row["true_identity"] for row in rows if not row["correct"])
    summary = {
        "split": "validation",
        "gallery_size": args.gallery_size,
        "gallery_identities": len(gallery_ids),
        "queries": len(rows),
        "top1": float(correct.mean()),
        "errors": int((~correct).sum()),
        "sample_count_bins": bin_metrics,
        "identities_with_at_least_one_error": len(wrong_identity_counts),
        "identities_with_all_queries_wrong": int(
            sum(
                wrong_count == identity_sizes[identity] - args.gallery_size
                for identity, wrong_count in wrong_identity_counts.items()
            )
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    wrong_rows = sorted(
        (row for row in rows if not row["correct"]),
        key=lambda row: (float(row["margin"]), -float(row["top1_similarity"])),
        reverse=True,
    )[: args.hardest]
    with (args.output_dir / "validation_hard_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(wrong_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
