"""Install the six verified model files from the group's completed delivery ZIP.

Uses only Python's standard library; does not execute or extract other ZIP files.
Run from the project root: python -m scripts.install_species_models --archive PATH
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["horse_mackerel", "whiting", "haddock", "cod", "hake"]
MODELS = {
    "resnet18": ("artifacts/species_final", "model.pt", "2ef8e62a5f8318118a56b0c464532566a0314ccbb295f49404b641951285b23c"),
    "svm": ("artifacts/species_comparison/svm", "model.joblib", "bf14fe1c979cec12299592751bcf3e4b284498846a73f0c25c1717de967a6899"),
    "yolov8n_cls": ("artifacts/species_comparison/yolov8n_cls", "model.pt", "59c3c1bd9ed75bf0d37123206fc1b473eb681d2373bc152514623c9d567381e5"),
}


def install(archive, destination):
    destination = Path(destination).resolve()
    pending = []
    with ZipFile(archive) as bundle:
        for family, (source, weight_name, expected_hash) in MODELS.items():
            contents = {}
            for filename in (weight_name, "selection.json"):
                relative = f"{source}/{filename}"
                matches = [entry for entry in bundle.infolist()
                           if entry.filename == relative or entry.filename.endswith("/" + relative)]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one {relative} in the delivery ZIP")
                if matches[0].file_size > (100_000 if filename == "selection.json" else 100_000_000):
                    raise ValueError(f"Unexpected file size: {relative}")
                contents[filename] = bundle.read(matches[0])
            selection = json.loads(contents["selection.json"])
            if (selection.get("family") != family or selection.get("labels") != LABELS
                    or selection.get("model_sha256") != expected_hash
                    or hashlib.sha256(contents[weight_name]).hexdigest() != expected_hash):
                raise ValueError(f"Delivery validation failed for {family}")
            expected_mode = "single" if family == "svm" else "flip4"
            if selection.get("inference_mode", "single") != expected_mode:
                raise ValueError(f"Unexpected inference mode for {family}")
            for filename, content in contents.items():
                target = destination / family / filename
                if target.exists() and target.read_bytes() != content:
                    raise FileExistsError(f"Different existing file: {target}. Back it up before installing.")
                pending.append((target, content))
    # Validate the complete delivery before writing any installed model.
    for target, content in pending:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "models" / "species")
    args = parser.parse_args()
    installed = install(args.archive, args.destination)
    print(f"Verified and installed SVM, ResNet-18 and YOLOv8n-cls in {installed}")
    print("Start or restart FishLens to load these models.")


if __name__ == "__main__":
    main()
