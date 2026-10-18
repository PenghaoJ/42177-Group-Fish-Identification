import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts import install_freshness_model as installer
from fish_freshness.final_predict import FinalFreshnessPredictor


class FreshnessInstallTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = b"fixture weights for installer only"
        self.digest = hashlib.sha256(self.weights).hexdigest()
        self.pin = patch.object(installer, "MODEL_SHA256", self.digest)
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def archive(self, weights=None):
        path = self.root / "delivery.zip"
        decision = {"selected_family": "resnet18", "test_used_for_selection": False,
                    "manifest_hashes": {}, "candidates": [{"family": "resnet18", "checkpoint_sha256": self.digest}]}
        with ZipFile(path, "w") as bundle:
            bundle.writestr("package/evaluation/selection_before_test.json", json.dumps(decision))
            bundle.writestr("package/runs/resnet18/best.pt", self.weights if weights is None else weights)
            bundle.writestr("package/unrelated.py", "raise AssertionError('must not execute')")
        return path

    def test_install_is_idempotent_and_preserves_different_existing_model(self):
        archive = self.archive()
        destination = self.root / "models"
        folder = installer.install(archive, destination)
        installer.install(archive, destination)
        self.assertEqual({p.name for p in folder.iterdir()}, {"model.pt", "selection.json"})
        (folder / "model.pt").write_bytes(b"another model")
        with self.assertRaises(FileExistsError):
            installer.install(archive, destination)
        self.assertEqual((folder / "model.pt").read_bytes(), b"another model")

    def test_corrupt_delivery_does_not_write_any_model(self):
        destination = self.root / "models"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            installer.install(self.archive(b"corrupt"), destination)
        self.assertFalse(destination.exists())

    def test_runtime_rejects_modified_checkpoint_before_loading(self):
        folder = installer.install(self.archive(), self.root / "models")
        (folder / "model.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "selected checkpoint"):
            FinalFreshnessPredictor(folder)
