#!/usr/bin/env python3
"""
Minimal InsightFace demo.

Usage:
  python face_demo.py --image img1.jpg
  python face_demo.py --image img1.jpg --image2 img2.jpg
  python face_demo.py --image img1.jpg --image2 img2.jpg --threshold 0.55

Outputs:
  - detected face count
  - bounding box / score for the largest face
  - embedding dimension
  - optional cosine similarity for two images
  - optional same/different decision when --image2 is provided
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def choose_providers() -> list[str]:
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def build_app() -> FaceAnalysis:
    app = FaceAnalysis(
        name="buffalo_l",
        providers=choose_providers(),
    )
    # ctx_id=0 for GPU, -1 for CPU
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in ort.get_available_providers() else -1, det_size=(640, 640))
    return app


def pick_largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def analyze_image(app: FaceAnalysis, image_path: str):
    img = load_image(image_path)
    faces = app.get(img)
    face = pick_largest_face(faces)
    if face is None:
        return {
            "image": image_path,
            "face_count": 0,
            "face": None,
        }
    return {
        "image": image_path,
        "face_count": len(faces),
        "face": {
            "bbox": [round(float(x), 2) for x in face.bbox.tolist()],
            "det_score": round(float(face.det_score), 4),
            "embedding": face.embedding,
            "embedding_dim": int(face.embedding.shape[0]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="first image path")
    parser.add_argument("--image2", help="optional second image path")
    parser.add_argument("--threshold", type=float, default=0.55, help="similarity threshold for same-person decision")
    args = parser.parse_args()

    app = build_app()
    result1 = analyze_image(app, args.image)

    print(f"image: {result1['image']}")
    print(f"faces_detected: {result1['face_count']}")
    if result1["face"] is None:
        print("no face found")
        return

    face1 = result1["face"]
    print(f"bbox: {face1['bbox']}")
    print(f"det_score: {face1['det_score']}")
    print(f"embedding_dim: {face1['embedding_dim']}")

    if args.image2:
        result2 = analyze_image(app, args.image2)
        print(f"image2: {result2['image']}")
        print(f"faces_detected2: {result2['face_count']}")
        if result2["face"] is None:
            print("no face found in image2")
            return
        face2 = result2["face"]
        sim = cosine_similarity(face1["embedding"], face2["embedding"])
        decision = "likely_same_person" if sim >= args.threshold else "likely_different_person"
        print(f"image2_bbox: {face2['bbox']}")
        print(f"image2_det_score: {face2['det_score']}")
        print(f"cosine_similarity: {sim:.6f}")
        print(f"threshold: {args.threshold:.6f}")
        print(f"decision: {decision}")


if __name__ == "__main__":
    main()
