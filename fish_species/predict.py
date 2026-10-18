"""Single-image predictor; compatible with FishLens' species/SVM adapter.

The image must show a single complete fish, as in the training crops. SVM
decision values are not probabilities: the website score is intentionally None.
Only load model artifacts that you trust (joblib uses Python pickle).
"""
import argparse
import json
import os
from pathlib import Path

import joblib
from PIL import Image

from .features import extract_features


class SpeciesSVMPredictor:
    def __init__(self, model_path):
        self.bundle = joblib.load(model_path)

    def __call__(self, image):
        features = extract_features(image, self.bundle["feature_config"])
        label_id = int(self.bundle["pipeline"].predict(features[None, :])[0])
        return {"label": self.bundle["labels"][label_id], "score": None,
                "model_version": self.bundle["model_version"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    predictor = SpeciesSVMPredictor(args.model)
    with Image.open(args.image) as image:
        print(json.dumps(predictor(image), indent=2))


if __name__ == "__main__":
    main()
