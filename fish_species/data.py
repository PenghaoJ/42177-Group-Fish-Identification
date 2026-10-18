"""Read fixed manifests and verify/extract the supplied AutoFish package.

Bundled Python scripts and README instructions are never executed.
"""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

EXPECTED_SPLITS = {"train": 6019, "validation": 1300, "test": 1180}
LABELS = ["horse_mackerel", "whiting", "haddock", "cod", "hake"]
SCIENTIFIC_NAMES = [
    "Trachurus trachurus", "Merlangius merlangus", "Melanogrammus aeglefinus",
    "Gadus morhua", "Merluccius merluccius",
]


def read_rows(root, split):
    with (Path(root) / "manifests" / "autofish5" / f"{split}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(splits):
    """Check counts, label mapping, and recorded split isolation for all rows."""
    for split, rows in splits.items():
        if len(rows) != EXPECTED_SPLITS[split]:
            raise ValueError(f"Unexpected sample count for {split}: {len(rows)}")
        for row in rows:
            if row["split"] != split or LABELS[int(row["label_id"])] != row["label"]:
                raise ValueError(f"Split/label mismatch: {row['sample_id']}")
            expected_prefix = f"autofish5/{split}/{row['label_id']}_{row['label']}/"
            if not row["file_path"].startswith(expected_prefix):
                raise ValueError(f"Path/label mismatch: {row['file_path']}")
    fields = ("sample_id", "file_path", "processed_sha256", "native_pixel_sha256",
              "specimen_id", "group_id", "source_frame_id")
    for field in fields:
        seen = {}
        for split, rows in splits.items():
            for row in rows:
                value = row[field]
                if value and value in seen and seen[value] != split:
                    raise ValueError(f"Cross-split {field}: {value}")
                seen[value] = split
    return {
        split: {"samples": len(rows), "class_counts": dict(Counter(r["label"] for r in rows)),
                "specimens": len({r["specimen_id"] for r in rows}),
                "groups": sorted({int(r["group_id"]) for r in rows})}
        for split, rows in splits.items()
    }


def prepare_data(archive, root):
    """Extract AutoFish photos/masks only, checking every file's actual hash."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = Path(archive).resolve()
    prefix = "42177_selected_data/"
    with ZipFile(archive) as bundle:
        for relative in ["label_map.json", "class_counts.csv"] + [
            f"manifests/autofish5/{split}.csv" for split in EXPECTED_SPLITS
        ]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.read(prefix + relative))
        splits = {s: read_rows(root, s) for s in EXPECTED_SPLITS}
        audit = validate_rows(splits)
        count, total = 0, sum(map(len, splits.values()))
        for rows in splits.values():
            for row in rows:
                for field, hash_field in [("file_path", "processed_sha256"), ("mask_path", "mask_sha256")]:
                    relative = row[field]
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError("Unsafe archive path")
                    expected = row[hash_field]
                    if path.is_file() and sha256_file(path) == expected:
                        continue
                    data = bundle.read(prefix + relative)
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError(f"Corrupt data: {relative}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                count += 1
                if count % 1000 == 0:
                    print(f"Verified/extracted {count}/{total} AutoFish photos and masks", flush=True)
    audit.update({"archive_path": str(archive), "archive_bytes": archive.stat().st_size,
                  "image_hashes_verified": total, "mask_hashes_verified": total,
                  "recorded_group_and_specimen_isolation": "passed",
                  "manifest_hashes": {s: sha256_file(root / "manifests" / "autofish5" / f"{s}.csv")
                                      for s in EXPECTED_SPLITS}})
    (root / "autofish_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
