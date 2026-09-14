#!/usr/bin/env python3
"""Compare multi-image galleries on validation embeddings only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_metric_adapter import ResidualMetricAdapter, load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gallery-sizes", type=int, nargs="+", default=(1, 2, 3))
    return parser.parse_args()


@torch.inference_mode()
def adapt_all(
    model: torch.nn.Module,
    vectors: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(vectors), batch_size):
        batch = torch.from_numpy(vectors[start : start + batch_size]).to(device)
        chunks.append(model(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def build_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    first = state["project.0.weight"]
    last = state["project.3.weight"]
    model = ResidualMetricAdapter(first.shape[1], first.shape[0], last.shape[0]).to(device)
    model.load_state_dict(state)
    return model


def evaluate_gallery_size(
    vectors: np.ndarray,
    identities: np.ndarray,
    paths: np.ndarray,
    gallery_size: int,
    query_start: int | None = None,
) -> dict[str, float | int]:
    import faiss

    gallery_vectors: list[np.ndarray] = []
    gallery_ids: list[str] = []
    query_indices: list[int] = []
    eligible_sizes: list[int] = []
    query_start = gallery_size if query_start is None else query_start
    if query_start < gallery_size:
        raise ValueError("query_start cannot be smaller than gallery_size")
    for identity in np.unique(identities):
        indices = np.where(identities == identity)[0]
        indices = indices[np.argsort(paths[indices])]
        if len(indices) <= query_start:
            continue
        centroid = vectors[indices[:gallery_size]].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        gallery_vectors.append(centroid.astype(np.float32))
        gallery_ids.append(str(identity))
        query_indices.extend(int(index) for index in indices[query_start:])
        eligible_sizes.append(int(len(indices)))

    gallery = np.stack(gallery_vectors)
    gallery_ids_array = np.asarray(gallery_ids)
    query = vectors[query_indices]
    query_ids = identities[query_indices].astype(str)
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    similarities, neighbours = index.search(query, min(3, len(gallery)))
    candidates = gallery_ids_array[neighbours]
    return {
        "gallery_size": int(gallery_size),
        "query_start": int(query_start),
        "gallery_identities": int(len(gallery)),
        "queries": int(len(query)),
        "top1": float(np.mean(candidates[:, 0] == query_ids)),
        "top3": float(np.mean(np.any(candidates == query_ids[:, None], axis=1))),
        "mean_top1_similarity": float(similarities[:, 0].mean()),
        "median_images_per_eligible_identity": float(np.median(eligible_sizes)),
    }


def main() -> int:
    args = parse_args()
    if any(size < 1 for size in args.gallery_sizes):
        raise SystemExit("Gallery sizes must be positive")
    device = torch.device(args.device)
    vectors, identities, paths = load_split(args.embeddings, "validation")
    model = build_model(args.checkpoint, device)
    adapted = adapt_all(model, vectors, device, args.batch_size)
    standard_results = [
        evaluate_gallery_size(adapted, identities, paths, gallery_size)
        for gallery_size in args.gallery_sizes
    ]
    fixed_query_start = max(args.gallery_sizes)
    matched_cohort_results = [
        evaluate_gallery_size(adapted, identities, paths, gallery_size, fixed_query_start)
        for gallery_size in args.gallery_sizes
    ]
    report = {
        "split": "validation",
        "standard_protocol": "first K path-sorted images form a normalized centroid; all remaining images are queries",
        "standard_results": standard_results,
        "matched_cohort_protocol": "all K values use identities with more than max(K) images and the same queries after max(K)",
        "matched_cohort_results": matched_cohort_results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
