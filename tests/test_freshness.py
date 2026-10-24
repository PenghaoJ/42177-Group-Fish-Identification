from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

from fish_freshness.cnn import FreshnessCNNPredictor, build_network, image_tensor
from fish_freshness.data import EXPECTED_COUNTS, LABELS, image_path, validate_rows
from fish_freshness.train_cnn import build_cache


def rows_fixture():
    splits = {}
    for split, counts in EXPECTED_COUNTS.items():
        rows = []
        for label_id, count in enumerate(counts):
            for index in range(count):
                uid = f"{split}_{label_id}_{index}"
                rows.append({"sample_id": uid, "label_id": str(label_id), "label": LABELS[label_id],
                             "split": split, "file_path": f"ffe3/{split}/{label_id}_{LABELS[label_id]}/{uid}.png",
                             "source_duplicate_group": uid})
        splits[split] = rows
    return splits


class FreshnessTest(unittest.TestCase):
    def test_source_group_leakage_is_rejected(self):
        rows = rows_fixture()
        validate_rows(rows)
        rows["validation"][0]["source_duplicate_group"] = rows["train"][0]["source_duplicate_group"]
        with self.assertRaisesRegex(ValueError, "Cross-split source_duplicate_group"):
            validate_rows(rows)

    def test_exact_duplicate_leakage_is_rejected(self):
        rows = rows_fixture()
        rows["train"][0]["processed_sha256"] = "same-image"
        rows["test"][0]["processed_sha256"] = "same-image"
        with self.assertRaisesRegex(ValueError, "Cross-split processed_sha256"):
            validate_rows(rows)

    def test_training_cache_cannot_open_test(self):
        with self.assertRaisesRegex(ValueError, "must not read test"):
            build_cache(Path("missing-data"), "test", Path("unused"))

    def test_path_cannot_escape_data_root(self):
        with self.assertRaises(ValueError):
            image_path(Path.cwd(), {"file_path": "../outside.png"})

    def test_checkpoint_roundtrip_preserves_three_class_predictions(self):
        torch.set_num_threads(2)
        torch.manual_seed(42177)
        model, architecture = build_network("resnet18")
        model.eval()
        im = Image.new("RGB", (120, 80), (80, 150, 170))
        with torch.inference_mode():
            expected = model(image_tensor(im)[None]).softmax(1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pt"
            payload = {"family": "resnet18", "architecture": architecture, "state_dict": model.state_dict(),
                       "labels": LABELS, "image_size": 224, "model_version": "test"}
            torch.save(payload, path)
            result = FreshnessCNNPredictor(path)(im)
            self.assertEqual(result["label"], LABELS[int(expected.argmax())])
            self.assertAlmostEqual(result["score"], float(expected.max()), places=6)
            payload["labels"] = list(reversed(LABELS))
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "label order"):
                FreshnessCNNPredictor(path)


if __name__ == "__main__":
    unittest.main()
