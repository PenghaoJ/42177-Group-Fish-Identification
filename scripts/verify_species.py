"""Check installed models against saved predictions; optionally check the live API.

Run from the repository root with: python -m scripts.verify_species --help
This is an integration regression check, not model selection or a new benchmark.
"""
import argparse
import base64
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from backend import adapters, config, server


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def request_json(url, payload=None):
    request = Request(url) if payload is None else Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="Delivery artifacts/species_comparison directory")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--url", help="Optional running FishLens URL")
    parser.add_argument("--output", type=Path, default=Path(".runtime/species-verification.json"))
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    rows = read_csv(args.data_root / "manifests/autofish5/test.csv")
    # Round-robin recorded fish before taking additional photos of each fish.
    selected = []
    for label in adapters.LABELS["species"]:
        by_fish = defaultdict(list)
        for row in rows:
            if row["label"] == label:
                by_fish[row["specimen_id"]].append(row)
        ordered = [group[i] for i in range(max(map(len, by_fish.values())))
                   for group in by_fish.values() if i < len(group)]
        if len(ordered) < args.per_class:
            raise ValueError(f"Not enough samples for {label}")
        selected.extend(ordered[:args.per_class])
    payloads = {}
    for row in selected:
        raw = (args.data_root / row["file_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["processed_sha256"]:
            raise ValueError(f"Image hash mismatch: {row['sample_id']}")
        payloads[row["sample_id"]] = {"image_base64": base64.b64encode(raw).decode()}
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "integration regression check, not a new accuracy estimate",
              "samples_per_model": len(selected), "per_class": args.per_class,
              "recorded_fish": len({r['specimen_id'] for r in selected}),
              "image_hashes_verified": True, "models": {}, "live_api": []}
    previous = config.FINAL_MODELS.copy()
    failures = []
    try:
        for family in adapters.MODELS:
            reference = {r["sample_id"]: r for r in read_csv(args.reference_root / family / "test_predictions.csv")}
            config.FINAL_MODELS["species"] = family
            outputs = []
            for row in selected:
                result = server.process("species", payloads[row["sample_id"]])
                expected = reference[row["sample_id"]]
                matched = result["mode"] == "trained" and result["label"] == expected["predicted_label"]
                delta = None if result["score"] is None else abs(result["score"] - float(expected["score_" + result["label"]]))
                matched = matched and (delta is None or delta <= 1e-5)
                if not matched:
                    failures.append(f"{family}:{row['sample_id']}")
                outputs.append({"sample_id": row["sample_id"], "true_label": row["label"],
                                "reference_label": expected["predicted_label"], "matches_delivery": matched,
                                "score_absolute_difference": delta, **result})
            report["models"][family] = {
                "matched_delivery": sum(r["matches_delivery"] for r in outputs),
                "matches_dataset_label": sum(r["label"] == r["true_label"] for r in outputs),
                "max_score_difference": max((r["score_absolute_difference"] or 0) for r in outputs),
                "predictions": outputs,
            }
            print(f"{family}: {report['models'][family]['matched_delivery']}/{len(selected)} match delivery", flush=True)
    finally:
        config.FINAL_MODELS.clear()
        config.FINAL_MODELS.update(previous)
    samples_dir = args.output.parent / "species-samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    examples = [next(r for r in selected if r["label"] == label) for label in adapters.LABELS["species"]]
    for row in examples:
        shutil.copyfile(args.data_root / row["file_path"], samples_dir / (row["label"] + ".png"))
    if args.url:
        base = args.url.rstrip("/")
        report["health"] = request_json(base + "/api/health")
        report["config"] = request_json(base + "/api/config")
        if report["config"]["tasks"] != {"species": {"mode": "trained"}, "freshness": {"mode": "demo"}}:
            failures.append("Unexpected live task modes")
        for row in examples:
            result = request_json(base + "/api/predict/species", payloads[row["sample_id"]])
            if result["mode"] != "trained" or result["model"] != previous["species"] or result["label"] != row["label"]:
                failures.append(f"live:{row['sample_id']}")
            explanation = request_json(base + "/api/explain", {"result_id": result["id"]})
            if "DEMO ONLY" in explanation["text"]:
                failures.append("Live species explanation incorrectly says demo")
            report["live_api"].append({"sample_id": row["sample_id"], **result})
        fresh = request_json(base + "/api/predict/freshness", payloads[examples[0]["sample_id"]])
        if fresh["mode"] != "demo" or fresh["score"] is not None:
            failures.append("Freshness must remain explicit demo")
        report["freshness_demo_verified"] = True
    report["failures"] = failures
    report["passed"] = not failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.output.resolve()}")
    if failures:
        raise SystemExit(f"Verification failed: {failures}")


if __name__ == "__main__":
    main()
