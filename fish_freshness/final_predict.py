"""Load the frozen FFE classifier once for the local web service."""
import json
from pathlib import Path
import threading

from .data import LABELS, sha256_file


class FinalFreshnessPredictor:
    def __init__(self, folder, device="cpu"):
        folder = Path(folder)
        self.selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        if (self.selection.get("family") != "resnet18" or self.selection.get("labels") != LABELS
                or self.selection.get("inference_mode") != "single"):
            raise ValueError("Unexpected selected FFE model or inference configuration")
        model_path = folder / "model.pt"
        if sha256_file(model_path) != self.selection["model_sha256"]:
            raise ValueError("Freshness model does not match the selected checkpoint")
        from .cnn import FreshnessCNNPredictor
        self.predictor = FreshnessCNNPredictor(model_path, device=device)
        checkpoint = self.predictor.checkpoint
        if (checkpoint["family"] != self.selection["family"] or checkpoint["image_size"] != 224
                or checkpoint["model_version"] != self.selection["model_version"]):
            raise ValueError("Freshness checkpoint does not match selection.json")
        self.lock = threading.Lock()

    def __call__(self, image):
        with self.lock:
            return self.predictor(image)
