"""Install the selected FFE ResNet-18 from the verified training delivery ZIP.

Only the selected weight is extracted; training scripts in the ZIP are not run.
Uses Python's standard library so installation can precede dependency setup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["highly_fresh", "fresh", "not_fresh"]
MODEL_SHA256 = "47436cb20198b32d711d7f1e8d677c4eb901e44fa55eb053a9389ed338296778"
MODEL_VERSION = "resnet18-ffe3-seed42177-epoch20"


def read_member(bundle, relative, limit):
    matches = [info for info in bundle.infolist()
               if info.filename == relative or info.filename.endswith("/" + relative)]
    if len(matches) != 1 or matches[0].file_size > limit:
        raise ValueError(f"Expected exactly one size-limited {relative}")
    return bundle.read(matches[0])


def install(archive, destination):
    with ZipFile(archive) as bundle:
        decision = json.loads(read_member(bundle, "evaluation/selection_before_test.json", 50_000))
        candidates = [entry for entry in decision["candidates"] if entry["family"] == "resnet18"]
        if (decision.get("selected_family") != "resnet18" or decision.get("test_used_for_selection") is not False
                or len(candidates) != 1 or candidates[0]["checkpoint_sha256"] != MODEL_SHA256):
            raise ValueError("Delivery does not contain the expected validation-selected FFE model")
        weights = read_member(bundle, "runs/resnet18/best.pt", 60_000_000)
        if hashlib.sha256(weights).hexdigest() != MODEL_SHA256:
            raise ValueError("FFE checkpoint SHA-256 mismatch")
    selection = {"family": "resnet18", "labels": LABELS, "model_sha256": MODEL_SHA256,
                 "model_version": MODEL_VERSION, "inference_mode": "single", "image_size": 224,
                 "selection_basis": "Validation macro-F1 before test evaluation",
                 "manifest_hashes": decision["manifest_hashes"],
                 "limitation": "Provisional source-grouped split; specimen independence unverified"}
    contents = {"model.pt": weights, "selection.json": (json.dumps(selection, indent=2) + "\n").encode("utf-8")}
    folder = Path(destination).resolve() / "resnet18"
    # Validate every existing target before writing anything. Never overwrite a different model.
    for name, content in contents.items():
        target = folder / name
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f"Different existing file: {target}. Back it up before installing.")
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        target = folder / name
        if target.exists():
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT / "models/freshness")
    args = parser.parse_args()
    print(f"Verified and installed FFE ResNet-18 in {install(args.archive, args.destination)}")
    print("Restart FishLens to enable fish-eye freshness predictions.")
