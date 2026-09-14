"""Offline tests for open-set threshold evaluation."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import face_cli
import evaluate_grouped_faces as evaluation


def row(query_type, true_person, best_person, similarity, decision="matched"):
    return {
        "query_type": query_type,
        "true_person": true_person,
        "best_person": best_person,
        "similarity": str(similarity),
        "decision": decision,
    }


class EvaluationTests(unittest.TestCase):
    def test_threshold_range_is_decimal_stable(self):
        self.assertEqual(evaluation.threshold_values(0.30, 0.32, 0.01), [0.3, 0.31, 0.32])
        with self.assertRaises(ValueError):
            evaluation.threshold_values(0.8, 0.3, 0.01)

    def test_open_set_metrics(self):
        rows = [
            row("known", "alice", "alice", 0.61),
            row("known", "bob", "alice", 0.70),
            row("known", "cara", "cara", 0.49),
            row("unknown", "stranger1", "alice", 0.55),
            row("unknown", "stranger2", "bob", 0.20),
            row("unknown", "undetected", "", "", decision="no_face"),
            row("unknown", "crowded", "", "", decision="multiple_faces"),
        ]
        result = evaluation.summarize_threshold(rows, 0.50)
        self.assertEqual(result["known_correct_accepts"], 1)
        self.assertEqual(result["known_success_rate"], "0.333333")
        self.assertEqual(result["unknown_false_accepts"], 1)
        self.assertEqual(result["false_accept_rate"], "0.500000")
        self.assertEqual(result["balanced_accuracy"], "0.416667")

    def test_strict_query_rejects_multiple_faces_before_matching(self):
        embedding = evaluation.np.array([1.0, 0.0], dtype=evaluation.np.float32)
        result = {"face_count": 2, "face": {"embedding": embedding, "det_score": 0.9}}
        with patch.object(evaluation, "get_embedding", return_value=(embedding, result)), \
             patch.object(evaluation, "best_match") as best_match:
            output = evaluation.evaluate_query(
                object(), "unknown", "stranger", Path("two.jpg"), {}, 0.5,
                strict_single_face=True,
            )
        self.assertEqual(output["decision"], "multiple_faces")
        self.assertEqual(output["query_face_count"], 2)
        best_match.assert_not_called()

    def test_strict_enrollment_fills_requested_valid_count(self):
        images = [Path(f"image{i}.jpg") for i in range(5)]
        embedding = evaluation.np.array([1.0, 0.0], dtype=evaluation.np.float32)
        multi = (embedding, {"face_count": 2, "face": {"det_score": 0.9}})
        valid = (embedding, {"face_count": 1, "face": {"det_score": 0.9}})
        with patch.object(evaluation, "get_embedding", side_effect=[multi, valid, valid, valid]):
            info, rows, queries = evaluation.prepare_enrollment(
                object(), "alice", images, 3, strict_single_face=True
            )
        self.assertEqual(len(info["embeddings"]), 3)
        self.assertEqual([row["status"] for row in rows], ["multiple_faces", "ok", "ok", "ok"])
        self.assertEqual(queries, [Path("image4.jpg")])

    def test_recommendation_breaks_ties_conservatively(self):
        summaries = [
            {"threshold": "0.50", "balanced_accuracy": "0.90", "false_accept_rate": "0.10"},
            {"threshold": "0.55", "balanced_accuracy": "0.90", "false_accept_rate": "0.00"},
            {"threshold": "0.60", "balanced_accuracy": "0.90", "false_accept_rate": "0.00"},
        ]
        self.assertEqual(evaluation.choose_recommended_threshold(summaries)["threshold"], "0.60")

    def test_unknown_groups_are_loaded_even_with_one_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stranger").mkdir()
            (root / "stranger" / "one.jpg").write_bytes(b"image")
            loaded = evaluation.load_unknown_groups(root, {"stranger"})
            self.assertEqual([path.name for path in loaded["stranger"]], ["one.jpg"])

    def test_grouped_loader_ignores_empty_unknown_folders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "person1").mkdir()
            (root / "person1" / "one.jpg").write_bytes(b"image")
            (root / "empty").mkdir()
            loaded = evaluation.load_grouped_dataset(root, excluded=set(), min_images=1)
            self.assertEqual(set(loaded), {"person1"})

    def test_cli_forwards_unknown_groups_and_does_not_exclude_test(self):
        with patch.object(sys, "argv", [
            "face_cli.py", "evaluate", "--unknown-group", "test", "--unknown-group", "zh"
        ]), patch.object(face_cli, "run_script", return_value=0) as run:
            self.assertEqual(face_cli.main(), 0)
        forwarded = run.call_args.args[1]
        self.assertEqual(forwarded.count("--unknown-group"), 2)
        self.assertNotIn("--exclude", forwarded)

    def test_cli_forwards_separate_unknown_dataset(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, "argv", [
            "face_cli.py", "evaluate", "--unknown-dataset", temporary,
            "--strict-single-face",
        ]), patch.object(face_cli, "run_script", return_value=0) as run:
            self.assertEqual(face_cli.main(), 0)
        forwarded = run.call_args.args[1]
        index = forwarded.index("--unknown-dataset")
        self.assertEqual(forwarded[index + 1], str(Path(temporary).resolve()))
        self.assertIn("--strict-single-face", forwarded)


if __name__ == "__main__":
    unittest.main()
