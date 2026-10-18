"""Select a colour/texture SVM using train and validation only; never evaluate test."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

# Same deterministic image descriptor as the species baseline; no species model is used.
from fish_species.features import FeatureConfig, extract_features
import fish_species.features as feature_module
from .data import LABELS, LIMITATION, checked_training_rows, image_path, sha256_file, versions, write_csv, write_json
from .metrics import metrics, save_predictions


def extract_one(path):
    with Image.open(path) as im:
        return extract_features(im)


def feature_matrix(root, rows, cache_dir, workers=4):
    identity = {"images": [(r["sample_id"], r["processed_sha256"]) for r in rows],
                "feature_config": asdict(FeatureConfig()), "source": sha256_file(feature_module.__file__)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"features_{digest[:24]}.npy"
    if not path.exists():
        paths = [image_path(root, r) for r in rows]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            x = np.asarray(list(pool.map(extract_one, paths, chunksize=16)))
        np.save(path, x, allow_pickle=False)
    x = np.load(path, allow_pickle=False)
    if x.shape != (len(rows), 221) or not np.isfinite(x).all():
        raise ValueError("Invalid feature cache")
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".training/cache"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit("Choose an empty run directory; existing results are preserved")
    rows, audit_report = checked_training_rows(args.data_root, args.audit)
    config = {"family": "svm", "labels": LABELS, "seed": 42177, "features": asdict(FeatureConfig()),
              "standardisation": "StandardScaler fitted on train only", "augmentation": "none",
              "selection": "Validation macro-F1, then accuracy, then first candidate", "versions": versions(),
              "test_used_during_training": False, "manifest_hashes": audit_report["manifest_hashes"],
              "limitation": LIMITATION, "source_sha256": {"train_svm.py": sha256_file(__file__),
              "features.py": sha256_file(feature_module.__file__)}}
    write_json(out / "config.json", config)
    started = time.perf_counter()
    x, y = {}, {}
    for split in ("train", "validation"):
        print(f"Extracting {split} colour/texture features ({len(rows[split])} images)", flush=True)
        x[split] = feature_matrix(args.data_root, rows[split], args.cache, args.workers)
        y[split] = np.asarray([int(r["label_id"]) for r in rows[split]])
    candidates = []
    for weight in (None, "balanced"):
        candidates.extend({"kernel": "linear", "C": c, "gamma": "scale", "class_weight": weight} for c in (.1, 1., 10.))
        candidates.extend({"kernel": "rbf", "C": c, "gamma": g, "class_weight": weight} for c in (1., 10., 100.) for g in ("scale", .001, .01))
    history, best_rank, best_model, best_parameters = [], (-1., -1.), None, None
    with threadpool_limits(limits=1):
        for index, params in enumerate(candidates, 1):
            model = make_pipeline(StandardScaler(), SVC(**params, probability=False, random_state=42177, cache_size=512))
            model.fit(x["train"], y["train"])
            predicted = model.predict(x["validation"])
            val = metrics(y["validation"], predicted)
            history.append({"candidate": index, **params, "validation_accuracy": val["accuracy"], "validation_macro_f1": val["macro_f1"]})
            rank = (val["macro_f1"], val["accuracy"])
            if rank > best_rank:
                best_rank, best_model, best_parameters = rank, model, params
            write_csv(out / "candidates.csv", history)
            print(f"SVM {index:02d}/{len(candidates)} validation F1={val['macro_f1']:.4f}", flush=True)
    bundle = {"pipeline": best_model, "labels": LABELS, "feature_config": asdict(FeatureConfig()),
              "model_version": "svm-ffe3-seed42177-v1", "training_config": config, "parameters": best_parameters}
    joblib.dump(bundle, out / "model.joblib", compress=3)
    val = save_predictions(out, "validation", rows["validation"], best_model.predict(x["validation"]))
    write_json(out / "summary.json", {"family": "svm", "validation_macro_f1": val["macro_f1"],
               "validation_accuracy": val["accuracy"], "parameters": best_parameters,
               "training_wall_seconds": time.perf_counter() - started, "checkpoint_sha256": sha256_file(out / "model.joblib")})
    print(f"COMPLETED SVM: validation F1={val['macro_f1']:.5f}", flush=True)


if __name__ == "__main__":
    main()
