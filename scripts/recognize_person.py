#!/usr/bin/env python3
"""
Recognize an unknown face against the local face-recognition registry.

Usage:
  python scripts/recognize_person.py --image data/samples/person1_b.jpg --threshold 0.55

What it does:
  1. Loads data/registry.json.
  2. Extracts an embedding from the unknown image.
  3. Compares it with registered embeddings.
  4. Prints the best match and a threshold-based decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import DEFAULT_DATA_DIR, registry_reference, user_path


def load_registry(registry_path: Path) -> dict:
    if not registry_path.exists():
        raise FileNotFoundError(f"registry not found: {registry_path}")
    return json.loads(registry_path.read_text(encoding="utf-8"))


def compare_with_person(query_embedding: np.ndarray, embedding_file: Path) -> float:
    registered_embeddings = np.load(embedding_file)
    similarities = [
        cosine_similarity(query_embedding, registered_embedding)
        for registered_embedding in registered_embeddings
    ]
    return max(similarities)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denominator = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="unknown image path")
    parser.add_argument("--threshold", type=float, default=0.55, help="minimum similarity to accept a match")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="data directory")
    args = parser.parse_args()

    # Model dependencies are loaded after argument parsing, so --help works in
    # lightweight environments and path/storage code remains testable offline.
    from face_demo import analyze_image, build_app

    data_dir = user_path(args.data_dir)
    registry_path = data_dir / "registry.json"
    registry = load_registry(registry_path)
    people = registry.get("people", {})
    if not people:
        raise RuntimeError("registry is empty; register a person first")

    app = build_app()
    image_path = user_path(args.image)
    query_result = analyze_image(app, str(image_path))
    if query_result["face"] is None:
        raise RuntimeError(f"recognition failed: no face found in {args.image}")

    query_face = query_result["face"]
    query_embedding = query_face["embedding"].astype(np.float32)

    best_match = None
    all_matches = []
    for person_id, person_info in people.items():
        embedding_file = registry_reference(person_info["embedding_file"], registry_path)
        similarity = compare_with_person(query_embedding, embedding_file)
        match = {
            "person_id": person_id,
            "name": person_info.get("name", ""),
            "similarity": similarity,
        }
        all_matches.append(match)
        if best_match is None or similarity > best_match["similarity"]:
            best_match = match

    assert best_match is not None
    decision = "matched" if best_match["similarity"] >= args.threshold else "unknown_person"

    print(f"image: {image_path}")
    print(f"faces_detected: {query_result['face_count']}")
    if query_result["face_count"] > 1:
        print("warning: multiple faces detected; recognized the largest face")
    print(f"bbox: {query_face['bbox']}")
    print(f"det_score: {query_face['det_score']}")
    print(f"embedding_dim: {query_face['embedding_dim']}")
    print(f"best_match_id: {best_match['person_id']}")
    print(f"best_match_name: {best_match['name']}")
    print(f"similarity: {best_match['similarity']:.6f}")
    print(f"threshold: {args.threshold:.6f}")
    print(f"decision: {decision}")

    print("top_matches:")
    for match in sorted(all_matches, key=lambda item: item["similarity"], reverse=True)[:5]:
        print(f"- {match['person_id']} {match['name']} {match['similarity']:.6f}")


if __name__ == "__main__":
    main()
