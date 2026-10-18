"""Check the installed FFE model and live API against the frozen delivery.

This is a validation-image integration regression, not a new test benchmark.
"""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from PIL.Image import open as open_image

from fish_freshness.data import LABELS, image_path, read_rows, sha256_file, write_json
from scripts.install_freshness_model import MODEL_VERSION
from backend import server


def request_json(url, payload=None):
    request = Request(url) if payload is None else Request(url,
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(".runtime/freshness-verification.json"))
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    audit = json.loads((args.reference_root / "ffe_audit.json").read_text())
    if sha256_file(args.data_root / "manifests/ffe3/validation.csv") != audit["manifest_hashes"]["validation"]:
        raise ValueError("Validation manifest differs from the delivery")
    rows = read_rows(args.data_root, "validation")
    selected = []
    for label in LABELS:
        group = [r for r in rows if r["label"] == label][:args.per_class]
        if len(group) != args.per_class:
            raise ValueError(f"Not enough images for {label}")
        selected.extend(group)
    with (args.reference_root / "runs/resnet18/validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
        reference = {r["sample_id"]: r for r in csv.DictReader(stream)}
    base = args.url.rstrip("/")
    health = request_json(base + "/api/health")
    config = request_json(base + "/api/config")
    assert ["freshness", "resnet18"] in health["connected_models"]
    assert config["tasks"]["freshness"]["mode"] == "trained"
    results = []
    samples = args.output.parent / "freshness-samples"
    samples.mkdir(parents=True, exist_ok=True)
    for row in selected:
        path = image_path(args.data_root, row)
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["processed_sha256"]
        payload = {"image_base64": base64.b64encode(raw).decode()}
        local = server.process("freshness", payload)
        live = request_json(base + "/api/predict/freshness", payload)
        assert live["mode"] == "trained" and live["model"] == "resnet18" and live["model_version"] == MODEL_VERSION
        assert live["scientific_name"] is None
        assert live["label"] == local["label"] == reference[row["sample_id"]]["predicted_label"]
        assert abs(live["score"] - local["score"]) < 1e-6
        delta = abs(live["score"] - float(reference[row["sample_id"]]["score_" + live["label"]]))
        results.append({"sample_id": row["sample_id"], "true_label": row["label"],
                        "reference_score_difference": delta, **live})
        sample = samples / (row["label"] + ".png")
        if not sample.exists():
            shutil.copyfile(path, sample)
        if row == next(r for r in selected if r["label"] == row["label"]):
            explanation = request_json(base + "/api/explain", {"result_id": live["id"]})
            assert "DEMO ONLY" not in explanation["text"] and "dataset-defined" in explanation["text"]
    with open_image(image_path(args.data_root, selected[0])) as im:
        jpeg = io.BytesIO()
        im.convert("RGB").save(jpeg, format="JPEG", quality=95)
    jpeg_payload = {"image_base64": base64.b64encode(jpeg.getvalue()).decode()}
    jpeg_live = request_json(base + "/api/predict/freshness", jpeg_payload)
    jpeg_local = server.process("freshness", jpeg_payload)
    assert jpeg_live["label"] == jpeg_local["label"] and abs(jpeg_live["score"] - jpeg_local["score"]) < 1e-6
    try:
        request_json(base + "/api/predict/freshness", {"image_base64": base64.b64encode(b"not an image").decode()})
        raise AssertionError("Invalid image accepted")
    except HTTPError as exc:
        assert exc.code == 400
    write_json(args.output, {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Integration regression on validation images, not a new accuracy estimate",
        "samples": len(selected), "all_labels_match_delivery": True, "live_matches_local_cpu": True,
        "max_reference_score_difference": max(r["reference_score_difference"] for r in results),
        "jpeg_upload": "passed", "invalid_upload": "HTTP 400", "health": health, "predictions": results})
    print(f"Passed: {len(selected)}/{len(selected)} live freshness predictions match delivery and local CPU inference")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
