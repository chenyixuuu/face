#!/usr/bin/env python3
"""Unified command-line entry point for the face-recognition baseline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def absolute_user_path(value: str) -> Path:
    """Resolve a user-provided path against the directory where the command ran."""
    return Path(value).expanduser().resolve()


def run_script(script_name: str, arguments: list[str]) -> int:
    command = [sys.executable, str(SCRIPTS_DIR / script_name), *arguments]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for face detection, registration, recognition, and evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="detect a face or compare two face images")
    demo.add_argument("--image", required=True, help="first image path")
    demo.add_argument("--image2", help="optional second image path")
    demo.add_argument("--threshold", type=float, default=0.50, help="same-person threshold")

    register = subparsers.add_parser("register", help="register one person from one or more images")
    register.add_argument("--name", required=True, help="display name")
    register.add_argument("--image", required=True, nargs="+", action="extend", help="registration image paths; repeatable")
    register.add_argument("--data-dir", help="registry data directory")

    recognize = subparsers.add_parser("recognize", help="recognize one image")
    recognize.add_argument("--image", required=True, help="query image path")
    recognize.add_argument("--threshold", type=float, default=0.50, help="match threshold")
    recognize.add_argument("--data-dir", help="registry data directory")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a grouped face dataset")
    evaluate.add_argument("--dataset", help="grouped dataset directory")
    evaluate.add_argument("--output", help="CSV report path")
    evaluate.add_argument("--threshold", type=float, default=0.50, help="match threshold")
    evaluate.add_argument("--enroll-count", type=int, default=3, help="enrollment images per person")
    evaluate.add_argument("--exclude", action="append", default=None, help="group to exclude; repeatable")
    evaluate.add_argument("--min-images", type=int, default=2, help="minimum images per group")
    evaluate.add_argument("--unknown-group", action="append", default=None,
                          help="stranger group used only for queries; repeatable")
    evaluate.add_argument("--unknown-dataset",
                          help="separate grouped dataset containing only stranger queries")
    evaluate.add_argument("--strict-single-face", action="store_true",
                          help="exclude no-face and multi-face images from threshold metrics")
    evaluate.add_argument("--sweep-output", help="threshold sweep CSV path")
    evaluate.add_argument("--sweep-start", type=float, default=0.30)
    evaluate.add_argument("--sweep-end", type=float, default=0.80)
    evaluate.add_argument("--sweep-step", type=float, default=0.01)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "demo":
        forwarded = [
            "--image",
            str(absolute_user_path(args.image)),
            "--threshold",
            str(args.threshold),
        ]
        if args.image2:
            forwarded.extend(["--image2", str(absolute_user_path(args.image2))])
        return run_script("face_demo.py", forwarded)

    if args.command == "register":
        data_dir = absolute_user_path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
        return run_script(
            "register_person.py",
            [
                "--name",
                args.name,
                "--image",
                *[str(absolute_user_path(value)) for value in args.image],
                "--data-dir",
                str(data_dir),
            ],
        )

    if args.command == "recognize":
        data_dir = absolute_user_path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
        return run_script(
            "recognize_person.py",
            [
                "--image",
                str(absolute_user_path(args.image)),
                "--threshold",
                str(args.threshold),
                "--data-dir",
                str(data_dir),
            ],
        )

    if args.command == "evaluate":
        dataset = absolute_user_path(args.dataset) if args.dataset else DEFAULT_DATA_DIR / "face_test_by_person"
        output = absolute_user_path(args.output) if args.output else PROJECT_ROOT / "reports" / "face_eval.csv"
        forwarded = [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--threshold",
            str(args.threshold),
            "--enroll-count",
            str(args.enroll_count),
            "--min-images",
            str(args.min_images),
            "--sweep-start",
            str(args.sweep_start),
            "--sweep-end",
            str(args.sweep_end),
            "--sweep-step",
            str(args.sweep_step),
        ]
        unknown_groups = args.unknown_group or []
        default_excluded = [] if "test" in unknown_groups else ["test"]
        for excluded in args.exclude if args.exclude is not None else default_excluded:
            forwarded.extend(["--exclude", excluded])
        for unknown in unknown_groups:
            forwarded.extend(["--unknown-group", unknown])
        if args.unknown_dataset:
            forwarded.extend(["--unknown-dataset", str(absolute_user_path(args.unknown_dataset))])
        if args.strict_single_face:
            forwarded.append("--strict-single-face")
        if args.sweep_output:
            forwarded.extend(["--sweep-output", str(absolute_user_path(args.sweep_output))])
        return run_script("evaluate_grouped_faces.py", forwarded)

    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
