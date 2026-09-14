"""Offline storage/CLI tests; model inference is injected, NumPy is real."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import face_cli
import register_person as registration


def analyze(app, path):
    name = Path(path).stem
    if name == "broken":
        raise FileNotFoundError("cannot decode image")
    count = {"none": 0, "group": 2}.get(name, 1)
    vector = np.ones(512, dtype=np.float32)
    if name == "invalid":
        vector[0] = np.nan
    return {"face_count": count, "face": {"embedding": vector} if count else None}


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / "data"

    def photo(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        return path

    def register(self, paths):
        return registration.register_images(" Alice ", paths, self.data, analyzer=analyze)

    def test_single_photo_compatible(self):
        result = self.register([self.photo("one.jpg")])
        self.assertEqual(np.load(result["embedding_file"]).shape, (1, 512))
        self.assertEqual(result["person_id"], "person_001")

    def test_multiple_and_basename_collision(self):
        sources = [self.photo("a/photo.jpg"), self.photo("b/photo.jpg")]
        result = self.register(sources)
        array = np.load(result["embedding_file"])
        self.assertEqual(array.shape, (2, 512))
        self.assertEqual(array.dtype, np.float32)
        person = registration.load_registry(self.data / "registry.json")["people"]["person_001"]
        self.assertEqual(person["name"], "Alice")
        self.assertEqual(len(set(person["photos"])), 2)
        for saved, source in zip(person["photos"], sources):
            self.assertEqual(Path(saved).read_bytes(), source.read_bytes())

    def test_mixed_rejections(self):
        good = self.photo("good.jpg")
        result = self.register([good, good, self.photo("none.jpg"), self.photo("group.jpg"),
                                self.photo("broken.jpg"), self.photo("invalid.jpg"),
                                self.photo("file.txt"), self.root / "missing.jpg"])
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(len(result["rejected"]), 7)

    def test_all_rejected_no_artifacts(self):
        result = self.register([self.photo("group.jpg"), self.photo("none.jpg")])
        self.assertIsNone(result["person_id"])
        self.assertFalse(self.data.exists())

    def test_existing_registry_preserved(self):
        first = self.register([self.photo("first.jpg")])
        registry_path = self.data / "registry.json"
        before = registry_path.read_bytes()
        self.register([self.photo("none.jpg")])
        self.assertEqual(registry_path.read_bytes(), before)
        second = self.register([self.photo("second.jpg")])
        self.assertEqual(second["person_id"], "person_002")
        people = registration.load_registry(registry_path)["people"]
        self.assertEqual(people["person_001"], json.loads(before)["people"]["person_001"])
        self.assertTrue(Path(first["embedding_file"]).exists())

    def test_registry_replace_failure_rolls_back(self):
        self.register([self.photo("first.jpg")])
        registry_path = self.data / "registry.json"
        before = registry_path.read_bytes()
        with patch.object(registration.os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.register([self.photo("second.jpg")])
        self.assertEqual(registry_path.read_bytes(), before)
        self.assertFalse((self.data / "people" / "person_002").exists())
        self.assertEqual(list(self.data.glob(".registry-*")), [])

    def test_existing_orphan_folder_not_removed(self):
        target = self.data / "people" / "person_001"
        target.mkdir(parents=True)
        marker = target / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.register([self.photo("good.jpg")])
        self.assertEqual(marker.read_text(), "keep")

    def test_blank_name(self):
        with self.assertRaises(ValueError):
            registration.register_images("  ", [], self.data, analyzer=analyze)
        self.assertFalse(self.data.exists())

    def test_cli_resolves_each_image_from_calling_directory(self):
        old_cwd = Path.cwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(self.root)
        with patch.object(sys, "argv", ["face_cli.py", "register", "--name", "Alice",
                                       "--image", "a.jpg", "b.jpg", "--image", "c.jpg"]):
            with patch.object(face_cli, "run_script", return_value=0) as run:
                self.assertEqual(face_cli.main(), 0)
        self.assertEqual(run.call_args.args, ("register_person.py", [
            "--name", "Alice", "--image", str(self.root / "a.jpg"),
            str(self.root / "b.jpg"), str(self.root / "c.jpg"),
            "--data-dir", str(ROOT / "data")]))

    def test_main_all_rejected_exit_status(self):
        with patch.object(sys, "argv", ["register_person.py", "--name", "Alice", "--image",
                                       str(self.root / "missing.jpg"), "--data-dir", str(self.data)]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(registration.main(), 1)
        self.assertIn("registry unchanged", output.getvalue())


if __name__ == "__main__":
    unittest.main()
