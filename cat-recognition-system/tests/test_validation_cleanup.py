import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_validation_exclusions.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_validation_exclusions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ValidationCleanupTests(unittest.TestCase):
    def test_transitive_identity_pairs_form_one_component(self):
        self.assertTrue(SCRIPT.exists(), "validation cleanup script is missing")
        module = load_module()
        pairs = [
            {"left_identity": "a", "right_identity": "b", "distance": 0},
            {"left_identity": "b", "right_identity": "c", "distance": 2},
            {"left_identity": "d", "right_identity": "e", "distance": 4},
        ]

        components = module.build_components(pairs)

        self.assertEqual(components, [("a", "b", "c"), ("d", "e")])

    def test_candidate_action_never_auto_merges_identities(self):
        self.assertTrue(SCRIPT.exists(), "validation cleanup script is missing")
        module = load_module()

        self.assertEqual(module.candidate_action(0), "exclude_pending_review")
        self.assertEqual(module.candidate_action(2), "exclude_pending_review")
        self.assertEqual(module.candidate_action(4), "manual_review")


if __name__ == "__main__":
    unittest.main()
