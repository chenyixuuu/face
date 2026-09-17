import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_unknown_cats.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("evaluate_unknown_cats", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class UnknownCatTests(unittest.TestCase):
    def test_identity_roles_are_disjoint_and_deterministic(self):
        identities = np.asarray([str(number) for number in range(20)])
        roles = MODULE.partition_identities(identities, 7)
        self.assertEqual(roles, MODULE.partition_identities(identities[::-1], 7))
        self.assertEqual([len(group) for group in roles.values()], [5] * 4)
        self.assertEqual(len(set().union(*map(set, roles.values()))), 20)

    def test_threshold_respects_unknown_far_with_float32_ties(self):
        scores = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        threshold = MODULE.choose_threshold(scores, 0.0)
        self.assertGreater(threshold, float(scores.max()))
        self.assertEqual(float(np.mean(scores >= threshold)), 0.0)
        tied = np.asarray([0.1, 0.2, 0.8, 0.8], dtype=np.float32)
        self.assertEqual(float(np.mean(tied >= MODULE.choose_threshold(tied, 0.25))), 0.0)

    def test_known_and_unknown_queries_do_not_overlap_gallery(self):
        vectors = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [-1, 0]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        identities = np.asarray(["a", "a", "b", "b", "u"])
        paths = np.asarray(["a0", "a1", "b0", "b1", "u0"])
        role = MODULE.evaluate_role(vectors, identities, paths, ["a", "b"], ["u"], 1)
        result = MODULE.summarize(role, 0.5)
        self.assertEqual(result["known_queries"], 2)
        self.assertEqual(result["unknown_queries"], 1)
        self.assertEqual(result["known_correct_accept_rate"], 1.0)
        self.assertEqual(result["unknown_false_accept_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
