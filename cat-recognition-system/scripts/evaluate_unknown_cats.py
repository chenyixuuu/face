#!/usr/bin/env python3
"""Calibrate and audit unknown-cat rejection using validation identities only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_metric_adapter import build_adapter_from_checkpoint, load_split


def partition_identities(identities: np.ndarray, seed: int) -> dict[str, list[str]]:
    """Place each eligible identity in exactly one of four deterministic roles."""
    ordered = sorted(
        set(map(str, identities)),
        key=lambda identity: hashlib.sha256(f"{seed}:{identity}".encode()).digest(),
    )
    names = ("calibration_known", "calibration_unknown", "evaluation_known", "evaluation_unknown")
    return {name: ordered[offset::4] for offset, name in enumerate(names)}


def choose_threshold(unknown_scores: np.ndarray, max_false_accept_rate: float) -> float:
    """Lowest threshold attaining the predeclared unknown false-accept target."""
    if len(unknown_scores) == 0 or not 0 <= max_false_accept_rate < 1:
        raise ValueError("Need unknown scores and a target in [0, 1)")
    sorted_scores = np.sort(unknown_scores.astype(np.float32))
    allowed = int(np.floor(max_false_accept_rate * len(sorted_scores)))
    cutoff = sorted_scores[len(sorted_scores) - allowed - 1]
    return float(np.nextafter(cutoff, np.float32(np.inf)))


def evaluate_role(
    vectors: np.ndarray,
    identities: np.ndarray,
    paths: np.ndarray,
    known_ids: list[str],
    unknown_ids: list[str],
    gallery_size: int,
) -> dict[str, object]:
    import faiss

    groups: dict[str, np.ndarray] = {}
    for identity in sorted(set(known_ids + unknown_ids)):
        indices = np.flatnonzero(identities == identity)
        groups[identity] = indices[np.argsort(paths[indices])]

    gallery = []
    gallery_ids = []
    known_query_indices = []
    unknown_query_indices = []
    for identity in known_ids:
        indices = groups[identity]
        if len(indices) <= gallery_size:
            raise ValueError(f"Known identity lacks query photos: {identity}")
        center = vectors[indices[:gallery_size]].mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        gallery.append(center.astype(np.float32))
        gallery_ids.append(identity)
        known_query_indices.extend(indices[gallery_size:].tolist())
    for identity in unknown_ids:
        unknown_query_indices.extend(groups[identity].tolist())

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.stack(gallery))

    def search(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        scores, neighbors = index.search(vectors[indices], 1)
        predictions = np.asarray(gallery_ids)[neighbors[:, 0]]
        return scores[:, 0], predictions

    known_scores, known_predictions = search(known_query_indices)
    unknown_scores, _ = search(unknown_query_indices)
    known_truth = identities[known_query_indices]
    return {
        "gallery_identities": len(gallery_ids),
        "known_queries": len(known_query_indices),
        "unknown_queries": len(unknown_query_indices),
        "known_scores": known_scores,
        "unknown_scores": unknown_scores,
        "known_correct": known_predictions == known_truth,
    }


def summarize(role: dict[str, object], threshold: float) -> dict[str, float | int]:
    known_scores = role["known_scores"]
    unknown_scores = role["unknown_scores"]
    correct = role["known_correct"]
    assert isinstance(known_scores, np.ndarray)
    assert isinstance(unknown_scores, np.ndarray)
    assert isinstance(correct, np.ndarray)
    accepted_known = known_scores >= threshold
    return {
        "gallery_identities": int(role["gallery_identities"]),
        "known_queries": int(role["known_queries"]),
        "unknown_queries": int(role["unknown_queries"]),
        "known_top1_without_rejection": float(np.mean(correct)),
        "known_correct_accept_rate": float(np.mean(correct & accepted_known)),
        "known_reject_rate": float(np.mean(~accepted_known)),
        "unknown_false_accept_rate": float(np.mean(unknown_scores >= threshold)),
        "unknown_reject_rate": float(np.mean(unknown_scores < threshold)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gallery-size", type=int, default=2)
    parser.add_argument("--max-unknown-far", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.gallery_size < 1:
        raise SystemExit("--gallery-size must be positive")

    vectors, identities, paths = load_split(args.embeddings, "validation")
    excluded = set(json.loads(args.exclusions.read_text())["excluded_identities"])
    counts = dict(zip(*np.unique(identities, return_counts=True)))
    eligible = {str(identity) for identity, count in counts.items() if count > args.gallery_size and str(identity) not in excluded}
    roles = partition_identities(np.asarray(sorted(eligible)), args.seed)
    if any(not group for group in roles.values()):
        raise ValueError("Not enough identities for four disjoint roles")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_adapter_from_checkpoint(checkpoint).to(args.device).eval()
    adapted = []
    with torch.inference_mode():
        for start in range(0, len(vectors), args.batch_size):
            batch = torch.from_numpy(vectors[start:start + args.batch_size]).to(args.device)
            adapted.append(model(batch).cpu().numpy().astype(np.float32))
    features = np.concatenate(adapted)

    calibration = evaluate_role(features, identities, paths, roles["calibration_known"], roles["calibration_unknown"], args.gallery_size)
    threshold = choose_threshold(calibration["unknown_scores"], args.max_unknown_far)
    evaluation = evaluate_role(features, identities, paths, roles["evaluation_known"], roles["evaluation_unknown"], args.gallery_size)
    report = {
        "split": "validation_only",
        "protocol": "Identity-disjoint calibration and evaluation roles; known gallery uses first K path-sorted photos; remaining known photos and all unknown photos are queries.",
        "seed": args.seed,
        "gallery_size": args.gallery_size,
        "excluded_conflict_identities": len(excluded),
        "eligible_identities": len(eligible),
        "role_identity_counts": {name: len(group) for name, group in roles.items()},
        "target_max_unknown_false_accept_rate": args.max_unknown_far,
        "selected_threshold": threshold,
        "calibration": summarize(calibration, threshold),
        "evaluation": summarize(evaluation, threshold),
        "limitations": "Internal validation simulation only; not a campus-photo or final-test estimate. Candidate matches still require human confirmation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
