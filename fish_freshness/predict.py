"""Freshness predictor with the same PIL -> label/score/version contract as FishLens."""
import argparse
import json
from pathlib import Path
import joblib
from PIL import Image
from fish_species.features import FeatureConfig, extract_features
from .data import LABELS


class FreshnessSVMPredictor:
    def __init__(self, path):
        self.bundle = joblib.load(path)
        if self.bundle["labels"] != LABELS:
            raise ValueError("Unexpected FFE label order")
        self.config = FeatureConfig(**self.bundle["feature_config"])

    def __call__(self, image):
        vector = extract_features(image, self.config)
        predicted = int(self.bundle["pipeline"].predict(vector[None, :])[0])
        return {"label": LABELS[predicted], "score": None, "model_version": self.bundle["model_version"]}


def load_predictor(path, device="cpu"):
    if Path(path).suffix == ".joblib":
        return FreshnessSVMPredictor(path)
    from .cnn import FreshnessCNNPredictor
    return FreshnessCNNPredictor(path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    # Capture the image loader before importing Ultralytics.
    with Image.open(args.image) as image:
        rgb = image.convert("RGB")
    print(json.dumps(load_predictor(args.model, args.device)(rgb), indent=2))
