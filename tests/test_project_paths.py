import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DEFAULT_DATA_DIR, DEFAULT_GROUPED_DATASET, DEFAULT_REPORT, PROJECT_ROOT, registry_reference


class ProjectPathTests(unittest.TestCase):
    def test_defaults_are_anchored_to_project(self):
        self.assertEqual(PROJECT_ROOT, ROOT)
        self.assertEqual(DEFAULT_DATA_DIR, ROOT / "data")
        self.assertEqual(DEFAULT_GROUPED_DATASET, ROOT / "data" / "face_test_by_person")
        self.assertEqual(DEFAULT_REPORT, ROOT / "reports" / "face_eval.csv")

    def test_legacy_data_path_resolves_from_project(self):
        registry = ROOT / "data" / "registry.json"
        expected = ROOT / "data" / "people" / "person_001" / "embeddings.npy"
        self.assertEqual(registry_reference("data/people/person_001/embeddings.npy", registry), expected)

    def test_registry_relative_path_resolves_beside_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary).resolve()
            expected = data / "people" / "person_001" / "embeddings.npy"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(registry_reference("people/person_001/embeddings.npy", data / "registry.json"), expected)

    def test_absolute_path_stays_absolute(self):
        with tempfile.TemporaryDirectory() as temporary:
            expected = Path(temporary).resolve() / "vectors.npy"
            self.assertEqual(registry_reference(expected, ROOT / "data" / "registry.json"), expected)


if __name__ == "__main__":
    unittest.main()
