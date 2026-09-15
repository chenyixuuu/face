#!/usr/bin/env python3
"""Build review candidates and a conservative validation exclusion list."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_components(pairs: list[dict]) -> list[tuple[str, ...]]:
    parent: dict[str, str] = {}

    def find(identity: str) -> str:
        parent.setdefault(identity, identity)
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        union(str(pair["left_identity"]), str(pair["right_identity"]))

    groups: dict[str, list[str]] = {}
    for identity in parent:
        groups.setdefault(find(identity), []).append(identity)
    return sorted(tuple(sorted(group)) for group in groups.values())


def candidate_action(distance: int) -> str:
    return "exclude_pending_review" if distance <= 2 else "manual_review"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    pairs = list(audit["pairs"])
    components = build_components(pairs)
    component_by_identity = {
        identity: component_index
        for component_index, component in enumerate(components, start=1)
        for identity in component
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = args.output_dir / "validation_merge_candidates.csv"
    with candidates_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "component_id", "distance", "recommended_action", "left_identity",
            "left_path", "right_identity", "right_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for pair in sorted(
            pairs,
            key=lambda row: (
                int(row["distance"]), str(row["left_identity"]), str(row["right_identity"]),
                str(row["left_path"]), str(row["right_path"]),
            ),
        ):
            writer.writerow({
                "component_id": component_by_identity[str(pair["left_identity"])],
                "distance": int(pair["distance"]),
                "recommended_action": candidate_action(int(pair["distance"])),
                "left_identity": pair["left_identity"],
                "left_path": pair["left_path"],
                "right_identity": pair["right_identity"],
                "right_path": pair["right_path"],
            })

    excluded_identities = sorted(component_by_identity)
    exclusions_path = args.output_dir / "validation_excluded_identities.json"
    exclusions_path.write_text(json.dumps({
        "policy": (
            "Sensitivity analysis only: exclude every identity in a cross-identity dHash "
            "candidate component; do not delete images or automatically merge labels."
        ),
        "source_audit": str(args.audit),
        "candidate_pairs": len(pairs),
        "components": len(components),
        "excluded_identities": excluded_identities,
        "component_members": [list(component) for component in components],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_pairs": len(pairs),
        "components": len(components),
        "excluded_identities": len(excluded_identities),
        "candidates": str(candidates_path),
        "exclusions": str(exclusions_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
