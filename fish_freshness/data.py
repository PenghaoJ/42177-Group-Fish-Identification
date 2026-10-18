"""Read and audit the existing FFE split without reassigning any images."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

from PIL import Image

LABELS = ["highly_fresh", "fresh", "not_fresh"]
EXPECTED_COUNTS = {"train": [1235, 924, 915], "validation": [266, 199, 197], "test": [263, 197, 194]}
LIMITATION = "Provisional source/near-duplicate-grouped split; specimen IDs absent, so individual-fish independence is unverified."


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty report")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(root, split):
    if split not in EXPECTED_COUNTS:
        raise ValueError(split)
    with (Path(root) / "manifests/ffe3" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def image_path(root, row):
    root = Path(root).resolve()
    path = (root / row["file_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path escapes the data root")
    return path


def manifest_hashes(root):
    return {s: sha256_file(Path(root) / "manifests/ffe3" / f"{s}.csv") for s in EXPECTED_COUNTS}


def validate_rows(splits):
    fields = ("sample_id", "file_path", "processed_sha256", "native_pixel_sha256",
              "source_frame_id", "source_duplicate_group", "specimen_id")
    seen = {field: {} for field in fields}
    result = {}
    for split, rows in splits.items():
        if Counter(int(r["label_id"]) for r in rows) != Counter(dict(enumerate(EXPECTED_COUNTS[split]))):
            raise ValueError(f"Unexpected class counts in {split}")
        ids = set()
        for r in rows:
            label_id = int(r["label_id"])
            if r["sample_id"] in ids:
                raise ValueError(f"Repeated sample: {r['sample_id']}")
            ids.add(r["sample_id"])
            prefix = f"ffe3/{split}/{label_id}_{LABELS[label_id]}/"
            if r["split"] != split or r["label"] != LABELS[label_id] or not r["file_path"].startswith(prefix):
                raise ValueError(f"Invalid split/label/path: {r['sample_id']}")
            if not r["source_duplicate_group"]:
                raise ValueError("Missing source group")
            for field in fields:
                value = r.get(field, "")
                if value and value in seen[field] and seen[field][value] != split:
                    raise ValueError(f"Cross-split {field}: {value}")
                if value:
                    seen[field][value] = split
        result[split] = {"samples": len(rows), "class_counts": dict(Counter(r["label"] for r in rows)),
                         "source_groups": len({r["source_duplicate_group"] for r in rows}),
                         "missing_specimen_ids": sum(not r.get("specimen_id") for r in rows)}
    return result


def audit(root, output):
    root = Path(root).resolve()
    splits = {s: read_rows(root, s) for s in EXPECTED_COUNTS}
    result = validate_rows(splits)
    label_map = json.loads((root / "label_map.json").read_text(encoding="utf-8"))["ffe3"]
    if [(v["label_id"], v["label"]) for v in label_map] != list(enumerate(LABELS)):
        raise ValueError("Unexpected label map")
    def verify(row):
        path = image_path(root, row)
        if sha256_file(path) != row["processed_sha256"]:
            raise ValueError(f"Image hash mismatch: {row['sample_id']}")
        with Image.open(path) as im:
            if im.mode != "RGB" or im.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"Image dimensions/mode mismatch: {row['sample_id']}")
            im.verify()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, [r for rows in splits.values() for r in rows]))
    result.update({"status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                   "labels": LABELS, "image_hashes_verified": 4390, "manifest_hashes": manifest_hashes(root),
                   "limitation": LIMITATION, "source_groups_isolated": True,
                   "verified_specimen_independence": False})
    write_json(output, result)
    return result


def checked_training_rows(root, audit_path):
    report = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if report["status"] != "passed" or report["manifest_hashes"] != manifest_hashes(root):
        raise ValueError("Run a fresh FFE audit: fixed manifests changed")
    return {s: read_rows(root, s) for s in ("train", "validation")}, report


def versions():
    return {p: importlib.metadata.version(p) for p in
            ["torch", "torchvision", "ultralytics", "scikit-learn", "scikit-image", "numpy", "Pillow", "joblib"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.data_root, args.output), indent=2))
