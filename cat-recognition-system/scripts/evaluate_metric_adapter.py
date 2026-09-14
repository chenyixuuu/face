#!/usr/bin/env python3
"""Evaluate a trained metric adapter with the fixed one-image gallery protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_metric_adapter import ResidualMetricAdapter, load_split, retrieval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required acknowledgement for the final held-out test evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing test evaluation without --allow-test")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    first_weight = state["project.0.weight"]
    last_weight = state["project.3.weight"]
    model = ResidualMetricAdapter(
        input_dim=int(first_weight.shape[1]),
        hidden_dim=int(first_weight.shape[0]),
        output_dim=int(last_weight.shape[0]),
    ).to(device)
    model.load_state_dict(state)

    vectors, identities, paths = load_split(args.embeddings, args.split)
    metrics = retrieval_metrics(model, vectors, identities, paths, device, args.batch_size)
    result = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_validation_metrics": checkpoint.get("metrics"),
        **metrics,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
