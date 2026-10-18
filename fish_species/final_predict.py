"""Load the validation-selected species model and predict a complete fish crop."""
import argparse
import json
import threading
from pathlib import Path

from PIL import Image
from .data import sha256_file


class FinalSpeciesPredictor:
    def __init__(self, folder, device="cpu"):
        folder = Path(folder)
        self.selection = json.loads((folder / "selection.json").read_text())
        family = self.selection["family"]
        if family not in ("svm", "resnet18", "yolov8n_cls"):
            raise ValueError(f"Unsupported model family: {family}")
        model_path = folder / ("model.joblib" if family == "svm" else "model.pt")
        if sha256_file(model_path) != self.selection["model_sha256"]:
            raise ValueError("Model file does not match the selected checkpoint")
        if family == "svm":
            from .predict import SpeciesSVMPredictor
            self.predictor = SpeciesSVMPredictor(folder / "model.joblib")
        else:
            if self.selection.get("inference_mode", "single") == "flip4":
                from .tta import SpeciesTTAPredictor
                self.predictor = SpeciesTTAPredictor(folder / "model.pt", device=device)
            else:
                from .cnn import SpeciesCNNPredictor
                self.predictor = SpeciesCNNPredictor(folder / "model.pt", device=device)
        saved_labels = self.predictor.bundle["labels"] if family == "svm" else self.predictor.checkpoint["labels"]
        if family != "svm" and self.predictor.checkpoint["family"] != family:
            raise ValueError("Checkpoint architecture does not match selection.json")
        if list(saved_labels) != list(self.selection["labels"]):
            raise ValueError("Model label order does not match selection.json")
        self.lock = threading.Lock()

    def __call__(self, image):
        with self.lock:
            return self.predictor(image)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="artifacts/species_final")
    p.add_argument("--image", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    predictor = FinalSpeciesPredictor(args.model_dir, args.device)
    with Image.open(args.image) as image:
        print(json.dumps(predictor(image), indent=2))


if __name__ == "__main__":
    main()
