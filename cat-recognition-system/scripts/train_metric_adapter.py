#!/usr/bin/env python3
"""Train a small supervised metric adapter on frozen DINOv2 embeddings.

The adapter is deliberately lightweight: it learns from train identities while the
backbone stays frozen, and model selection uses validation identities only.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_split(directory: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefixes = sorted(directory.glob(f"dinov2_vitb14_reg_{split}_rank*_embeddings.npy"))
    if not prefixes:
        raise FileNotFoundError(f"No embedding shards for split={split} in {directory}")
    vectors, identities, paths = [], [], []
    for embedding_path in prefixes:
        prefix = str(embedding_path).removesuffix("_embeddings.npy")
        vectors.append(np.load(embedding_path).astype(np.float32))
        identities.append(np.load(prefix + "_identities.npy"))
        paths.append(np.load(prefix + "_paths.npy"))
    return np.concatenate(vectors), np.concatenate(identities), np.concatenate(paths)


class ResidualMetricAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.project(vectors), dim=1)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    denominator = torch.logsumexp(logits.masked_fill(self_mask, float("-inf")), dim=1)
    positive_mean = (logits * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return (denominator - positive_mean).mean()


def make_groups(labels: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.astype(str)):
        groups[label].append(index)
    eligible = sorted(label for label, indices in groups.items() if len(indices) >= 2)
    return eligible, {label: np.asarray(groups[label], dtype=np.int64) for label in eligible}


def sample_balanced_batch(
    vectors: np.ndarray,
    eligible: list[str],
    groups: dict[str, np.ndarray],
    identities_per_batch: int,
    images_per_identity: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = rng.choice(eligible, size=identities_per_batch, replace=False)
    vector_indices: list[int] = []
    numeric_labels: list[int] = []
    for numeric_label, identity in enumerate(chosen):
        indices = rng.choice(groups[str(identity)], size=images_per_identity, replace=False)
        vector_indices.extend(indices.tolist())
        numeric_labels.extend([numeric_label] * images_per_identity)
    return torch.from_numpy(vectors[vector_indices]), torch.tensor(numeric_labels, dtype=torch.long)


@torch.inference_mode()
def retrieval_metrics(
    model: nn.Module,
    vectors: np.ndarray,
    identities: np.ndarray,
    paths: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    import faiss

    outputs = []
    model.eval()
    for start in range(0, len(vectors), batch_size):
        batch = torch.from_numpy(vectors[start : start + batch_size]).to(device)
        outputs.append(model(batch).cpu().numpy().astype(np.float32))
    adapted = np.concatenate(outputs)

    gallery_indices: list[int] = []
    query_indices: list[int] = []
    for identity in np.unique(identities):
        indices = np.where(identities == identity)[0]
        indices = indices[np.argsort(paths[indices])]
        if len(indices) >= 2:
            gallery_indices.append(int(indices[0]))
            query_indices.extend(int(index) for index in indices[1:])
    gallery = adapted[gallery_indices]
    gallery_ids = identities[gallery_indices].astype(str)
    query = adapted[query_indices]
    query_ids = identities[query_indices].astype(str)
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    _, neighbours = index.search(query, min(3, len(gallery)))
    candidates = gallery_ids[neighbours]
    return {
        "gallery_identities": int(len(gallery)),
        "queries": int(len(query)),
        "top1": float(np.mean(candidates[:, 0] == query_ids)),
        "top3": float(np.mean(np.any(candidates == query_ids[:, None], axis=1))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--identities-per-batch", type=int, default=128)
    parser.add_argument("--images-per-identity", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_vectors, train_ids, _ = load_split(args.embeddings, "train")
    val_vectors, val_ids, val_paths = load_split(args.embeddings, "validation")
    eligible, groups = make_groups(train_ids)
    if len(eligible) < args.identities_per_batch:
        raise ValueError("Not enough train identities with at least two images")

    model = ResidualMetricAdapter(train_vectors.shape[1], args.hidden_dim, args.output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    history_path = args.output / "history.jsonl"
    best_top1 = -1.0

    with history_path.open("w", encoding="utf-8") as history:
        for step in range(1, args.steps + 1):
            model.train()
            batch, labels = sample_balanced_batch(
                train_vectors,
                eligible,
                groups,
                args.identities_per_batch,
                args.images_per_identity,
                rng,
            )
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_contrastive_loss(model(batch), labels, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            if step == 1 or step % 50 == 0:
                print(f"step={step} loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.8f}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics = retrieval_metrics(model, val_vectors, val_ids, val_paths, device, args.eval_batch_size)
                record = {"step": step, "loss": float(loss.item()), **metrics}
                history.write(json.dumps(record) + "\n")
                history.flush()
                print(json.dumps(record), flush=True)
                torch.save({"model": model.state_dict(), "config": vars(args), "metrics": metrics}, args.output / "last.pt")
                if float(metrics["top1"]) > best_top1:
                    best_top1 = float(metrics["top1"])
                    torch.save({"model": model.state_dict(), "config": vars(args), "metrics": metrics}, args.output / "best.pt")

    print(f"best_validation_top1={best_top1:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
