#!/usr/bin/env python3
"""Evaluate closed-set or open-set face identification on grouped images.

Known groups contribute their first N images to enrollment and the rest to
queries. Groups named with ``--unknown-group`` never participate in enrollment;
all of their images are independent stranger queries. When stranger queries are
present, a threshold sweep report is also written.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path

import numpy as np

from project_paths import DEFAULT_GROUPED_DATASET, DEFAULT_REPORT, user_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_FIELDS = [
    "query_type", "true_person", "query_image", "pred_person", "best_person",
    "similarity", "threshold", "decision", "correct", "query_face_count",
    "query_det_score", "enroll_image",
]
INVALID_FACE_DECISIONS = {"no_face", "multiple_faces"}
SWEEP_FIELDS = [
    "threshold", "known_detected", "known_correct_accepts", "known_success_rate",
    "unknown_detected", "unknown_false_accepts", "false_accept_rate",
    "unknown_rejection_rate", "balanced_accuracy",
]


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.startswith(".") and not path.name.startswith("._")
    )


def load_grouped_dataset(dataset_dir: Path, excluded: set[str], min_images: int) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir() or person_dir.name in excluded:
            continue
        images = list_images(person_dir)
        if len(images) >= min_images:
            groups[person_dir.name] = images
    return groups


def load_unknown_groups(dataset_dir: Path, names: set[str]) -> dict[str, list[Path]]:
    groups = {}
    for name in sorted(names):
        folder = dataset_dir / name
        if not folder.is_dir():
            raise ValueError(f"unknown group does not exist: {name}")
        images = list_images(folder)
        if not images:
            raise ValueError(f"unknown group has no supported images: {name}")
        groups[name] = images
    return groups


def get_embedding(app, image_path: Path, analyzer=None):
    if analyzer is None:
        from face_demo import analyze_image
        analyzer = analyze_image
    result = analyzer(app, str(image_path))
    if result["face"] is None:
        return None, result
    return result["face"]["embedding"].astype(np.float32), result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denominator = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denominator)


def threshold_values(start: float, end: float, step: float) -> list[float]:
    if not (0 <= start <= end <= 1):
        raise ValueError("sweep range must satisfy 0 <= start <= end <= 1")
    if step <= 0:
        raise ValueError("sweep step must be positive")
    current, stop, increment = Decimal(str(start)), Decimal(str(end)), Decimal(str(step))
    values = []
    while current <= stop:
        values.append(float(current))
        current += increment
    return values


def summarize_threshold(rows: list[dict], threshold: float) -> dict:
    valid = [row for row in rows if row["decision"] not in INVALID_FACE_DECISIONS]
    known = [row for row in valid if row["query_type"] == "known"]
    unknown = [row for row in valid if row["query_type"] == "unknown"]
    known_correct = sum(
        row["best_person"] == row["true_person"] and float(row["similarity"]) >= threshold
        for row in known
    )
    false_accepts = sum(float(row["similarity"]) >= threshold for row in unknown)
    known_rate = known_correct / len(known) if known else None
    false_accept_rate = false_accepts / len(unknown) if unknown else None
    rejection_rate = 1 - false_accept_rate if false_accept_rate is not None else None
    balanced = ((known_rate + rejection_rate) / 2
                if known_rate is not None and rejection_rate is not None else None)
    return {
        "threshold": f"{threshold:.6f}",
        "known_detected": len(known),
        "known_correct_accepts": known_correct,
        "known_success_rate": "" if known_rate is None else f"{known_rate:.6f}",
        "unknown_detected": len(unknown),
        "unknown_false_accepts": false_accepts,
        "false_accept_rate": "" if false_accept_rate is None else f"{false_accept_rate:.6f}",
        "unknown_rejection_rate": "" if rejection_rate is None else f"{rejection_rate:.6f}",
        "balanced_accuracy": "" if balanced is None else f"{balanced:.6f}",
    }


def choose_recommended_threshold(summaries: list[dict]) -> dict | None:
    eligible = [row for row in summaries if row["balanced_accuracy"] != ""]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (
        float(row["balanced_accuracy"]),
        -float(row["false_accept_rate"]),
        float(row["threshold"]),
    ))


def best_match(query_embedding, enrolled: dict[str, dict]) -> tuple[str, float, str]:
    best_person, best_similarity, best_enroll_image = "", -1.0, ""
    for person_name, info in enrolled.items():
        for enroll_image, registered_embedding in zip(info["images"], info["embeddings"]):
            similarity = cosine_similarity(query_embedding, registered_embedding)
            if similarity > best_similarity:
                best_person, best_similarity = person_name, similarity
                best_enroll_image = str(enroll_image)
    return best_person, best_similarity, best_enroll_image


def evaluate_query(app, query_type: str, true_person: str, query_image: Path,
                   enrolled: dict[str, dict], threshold: float,
                   strict_single_face: bool = False) -> dict:
    query_embedding, result = get_embedding(app, query_image)
    invalid_decision = None
    if query_embedding is None:
        invalid_decision = "no_face"
    elif strict_single_face and result["face_count"] != 1:
        invalid_decision = "multiple_faces"
    if invalid_decision:
        return {
            "query_type": query_type, "true_person": true_person,
            "query_image": str(query_image), "pred_person": "", "best_person": "",
            "similarity": "", "threshold": f"{threshold:.6f}", "decision": invalid_decision,
            "correct": "false", "query_face_count": result["face_count"],
            "query_det_score": "", "enroll_image": "",
        }
    person, similarity, enroll_image = best_match(query_embedding, enrolled)
    decision = "matched" if similarity >= threshold else "unknown_person"
    predicted = person if decision == "matched" else ""
    correct = predicted == true_person if query_type == "known" else decision == "unknown_person"
    return {
        "query_type": query_type, "true_person": true_person,
        "query_image": str(query_image), "pred_person": predicted, "best_person": person,
        "similarity": f"{similarity:.6f}", "threshold": f"{threshold:.6f}",
        "decision": decision, "correct": str(correct).lower(),
        "query_face_count": result["face_count"], "query_det_score": result["face"]["det_score"],
        "enroll_image": enroll_image,
    }


def prepare_enrollment(app, person_name: str, images: list[Path], enroll_count: int,
                       strict_single_face: bool = False) -> tuple[dict | None, list[dict], list[Path]]:
    """Collect enrollment embeddings and return the remaining query images."""
    embeddings, enroll_images, enrollment_rows = [], [], []
    candidates = images if strict_single_face else images[:enroll_count]
    attempted_count = 0
    for enroll_image in candidates:
        if len(embeddings) >= enroll_count:
            break
        attempted_count += 1
        embedding, result = get_embedding(app, enroll_image)
        status = "ok"
        if embedding is None:
            status = "no_face"
        elif strict_single_face and result["face_count"] != 1:
            status = "multiple_faces"
        enrollment_rows.append({
            "person": person_name, "enroll_image": str(enroll_image),
            "status": status, "face_count": result["face_count"],
            "det_score": "" if status != "ok" else result["face"]["det_score"],
        })
        if status == "ok":
            embeddings.append(embedding)
            enroll_images.append(enroll_image)
    query_start = attempted_count if strict_single_face else enroll_count
    info = ({"images": enroll_images, "embeddings": embeddings} if embeddings else None)
    return info, enrollment_rows, images[query_start:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_GROUPED_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_REPORT), help="per-query CSV report path")
    parser.add_argument("--threshold", type=float, default=0.55, help="threshold used in the per-query report")
    parser.add_argument("--enroll-count", type=int, default=1)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--unknown-group", action="append", default=[],
                        help="stranger group used only as queries; can be repeated")
    parser.add_argument("--unknown-dataset",
                        help="separate grouped dataset whose non-empty folders are all stranger queries")
    parser.add_argument("--strict-single-face", action="store_true",
                        help="exclude no-face and multi-face images from threshold metrics")
    parser.add_argument("--sweep-output", help="threshold sweep CSV; derived from --output by default")
    parser.add_argument("--sweep-start", type=float, default=0.30)
    parser.add_argument("--sweep-end", type=float, default=0.80)
    parser.add_argument("--sweep-step", type=float, default=0.01)
    args = parser.parse_args()

    if args.enroll_count < 1 or args.min_images < 1:
        raise ValueError("enroll-count and min-images must be positive")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    from face_demo import build_app

    dataset_dir, output_path = user_path(args.dataset), user_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unknown_names, excluded = set(args.unknown_group), set(args.exclude)
    overlap = unknown_names & excluded
    if overlap:
        raise ValueError(f"groups cannot be both unknown and excluded: {sorted(overlap)}")
    known_groups = load_grouped_dataset(
        dataset_dir, excluded=excluded | unknown_names, min_images=args.min_images
    )
    unknown_groups = load_unknown_groups(dataset_dir, unknown_names)
    if args.unknown_dataset:
        external_unknown = load_grouped_dataset(
            user_path(args.unknown_dataset), excluded=set(), min_images=1
        )
        duplicate_names = set(unknown_groups) & set(external_unknown)
        if duplicate_names:
            raise ValueError(f"duplicate unknown group names: {sorted(duplicate_names)}")
        unknown_groups.update(external_unknown)
    if not known_groups:
        raise RuntimeError(f"no known grouped images found in {dataset_dir}")

    app = build_app()
    enrolled: dict[str, dict] = {}
    enrollment_rows = []
    known_queries: dict[str, list[Path]] = {}
    for person_name, images in known_groups.items():
        info, person_rows, query_images = prepare_enrollment(
            app, person_name, images, args.enroll_count,
            strict_single_face=args.strict_single_face,
        )
        enrollment_rows.extend(person_rows)
        known_queries[person_name] = query_images
        if info:
            enrolled[person_name] = info
    if not enrolled:
        raise RuntimeError("no known people could be enrolled")

    rows = []
    for person, images in known_queries.items():
        for image in images:
            rows.append(evaluate_query(
                app, "known", person, image, enrolled, args.threshold,
                strict_single_face=args.strict_single_face,
            ))
    for person, images in unknown_groups.items():
        for image in images:
            rows.append(evaluate_query(
                app, "unknown", person, image, enrolled, args.threshold,
                strict_single_face=args.strict_single_face,
            ))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct_count = sum(row["correct"] == "true" for row in rows)
    no_face_count = sum(row["decision"] == "no_face" for row in rows)
    multiple_face_count = sum(row["decision"] == "multiple_faces" for row in rows)
    print(f"dataset: {dataset_dir}")
    print(f"known_groups: {len(known_groups)}")
    print(f"unknown_groups: {sorted(unknown_groups)}")
    print(f"enrolled_people: {len(enrolled)}")
    print(f"queries: {total} (known={sum(r['query_type'] == 'known' for r in rows)}, unknown={sum(r['query_type'] == 'unknown' for r in rows)})")
    print(f"correct_at_{args.threshold:.2f}: {correct_count}/{total}")
    print(f"no_face: {no_face_count}")
    print(f"multiple_faces: {multiple_face_count}")
    print(f"report: {output_path}")

    if unknown_groups:
        summaries = [summarize_threshold(rows, value) for value in threshold_values(
            args.sweep_start, args.sweep_end, args.sweep_step
        )]
        sweep_path = (user_path(args.sweep_output) if args.sweep_output else
                      output_path.with_name(f"{output_path.stem}_thresholds.csv"))
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with sweep_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS)
            writer.writeheader()
            writer.writerows(summaries)
        recommended = choose_recommended_threshold(summaries)
        print(f"threshold_sweep: {sweep_path}")
        if recommended:
            print(f"recommended_threshold: {float(recommended['threshold']):.2f}")
            print(f"known_success_rate: {recommended['known_success_rate']}")
            print(f"false_accept_rate: {recommended['false_accept_rate']}")
            print(f"unknown_rejection_rate: {recommended['unknown_rejection_rate']}")
            print(f"balanced_accuracy: {recommended['balanced_accuracy']}")

    print("enrollment:")
    for row in enrollment_rows:
        print(row)


if __name__ == "__main__":
    main()
