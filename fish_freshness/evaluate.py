"""Freeze the validation-selected models, then evaluate the held-out FFE test split."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from PIL.Image import open as open_image
import torch

from .cnn import FreshnessCNNPredictor, image_tensor
from .data import LABELS, LIMITATION, checked_training_rows, image_path, read_rows, sha256_file, write_csv, write_json
from .metrics import save_predictions, metrics
from .predict import FreshnessSVMPredictor
from .train_svm import feature_matrix

FAMILIES = ["svm", "resnet18", "yolov8n_cls"]


def freeze_selection(runs, audit, output):
    candidates = []
    for family in FAMILIES:
        folder = runs / family
        summary = json.loads((folder / "summary.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        if config["manifest_hashes"] != audit["manifest_hashes"] or config["test_used_during_training"]:
            raise ValueError(f"Invalid training protocol: {family}")
        model = folder / ("model.joblib" if family == "svm" else "best.pt")
        digest = sha256_file(model)
        if digest != summary["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint modified: {family}")
        candidates.append({"family": family, "checkpoint_sha256": digest,
                           "validation_macro_f1": summary["validation_macro_f1"],
                           "validation_accuracy": summary["validation_accuracy"]})
    selected = max(candidates, key=lambda c: (c["validation_macro_f1"], c["validation_accuracy"]))["family"]
    decision = {"selected_family": selected, "selection_basis": "Validation macro-F1, then accuracy; first family for exact ties",
                "candidates": candidates, "manifest_hashes": audit["manifest_hashes"], "limitation": LIMITATION,
                "test_used_for_selection": False}
    path = output / "selection_before_test.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous[k] != v for k, v in decision.items()):
            raise ValueError("Previously frozen model selection differs; do not retune against this test set")
    else:
        decision["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, decision)
    return decision


def cnn_predictions(predictor, root, rows, batch_size=64):
    result = []
    for start in range(0, len(rows), batch_size):
        batch = []
        for row in rows[start:start + batch_size]:
            with open_image(image_path(root, row)) as im:
                batch.append(image_tensor(im, predictor.checkpoint["image_size"]))
        x = torch.stack(batch).to(predictor.device)
        with torch.inference_mode():
            result.extend(predictor.model(x).softmax(1).cpu().tolist())
    probabilities = np.asarray(result)
    return probabilities.argmax(1), probabilities


def plots(output, runs, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    names = ["Highly fresh", "Fresh", "Not fresh"]
    for ax, family in zip(axes, FAMILIES):
        matrix = np.asarray(results[family]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=263)
        for y in range(3):
            for x in range(3):
                ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="white" if matrix[y, x] > 140 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
               xlabel="Predicted label", ylabel="True label", title=f"{family}\nMacro-F1 {results[family]['macro_f1']:.3f}")
    fig.suptitle("FFE test: 654 eye images | specimen independence unverified", fontsize=13)
    fig.savefig(output / "test_confusion_matrices.png", dpi=180)
    plt.close(fig)


def error_examples(output, root, selected):
    import matplotlib.pyplot as plt
    with (output / selected / "test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        mistakes = [r for r in csv.DictReader(stream) if r["true_label"] != r["predicted_label"]]
    chosen = [r for label in LABELS for r in [r for r in mistakes if r["true_label"] == label][:2]]
    if not chosen:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), chosen):
        with open_image(image_path(root, row)) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{row['sample_id']} | {row['species_metadata']}\nTrue: {row['true_label']}\nPredicted: {row['predicted_label']}", fontsize=9)
    fig.suptitle(f"{selected}: first two test errors per true class (manifest order)\nDiagnostics only; no subsequent retuning on test", fontsize=12)
    fig.savefig(output / "selected_model_error_examples.png", dpi=160)
    plt.close(fig)


def learning_curves(output, runs):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, family in zip(axes, FAMILIES[1:]):
        with (runs / family / "history.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        epochs = [int(r["epoch"]) for r in records]
        ax.plot(epochs, [float(r["train_accuracy"]) for r in records], label="Train accuracy")
        ax.plot(epochs, [float(r["validation_accuracy"]) for r in records], label="Validation accuracy")
        ax.plot(epochs, [float(r["validation_macro_f1"]) for r in records], label="Validation macro-F1")
        ax.set(title=family, xlabel="Epoch", ylabel="Score", ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".training/cache"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "comparison.csv").exists():
        raise SystemExit("Evaluation already completed. Read the saved report instead of repeating model selection.")
    torch.set_num_threads(4)
    training_rows, audit = checked_training_rows(args.data_root, args.audit)
    decision = freeze_selection(args.runs, audit, args.output)
    test_rows = read_rows(args.data_root, "test")
    results, comparisons, species_diagnostics = {}, [], []
    for family in FAMILIES:
        run = args.runs / family
        folder = args.output / family
        folder.mkdir(exist_ok=True)
        summary = json.loads((run / "summary.json").read_text())
        model_path = run / ("model.joblib" if family == "svm" else "best.pt")
        if family == "svm":
            predictor = FreshnessSVMPredictor(model_path)
            x = feature_matrix(args.data_root, training_rows["validation"], args.cache)
            validation_predictions = predictor.bundle["pipeline"].predict(x)
        else:
            predictor = FreshnessCNNPredictor(model_path, args.device)
            validation_predictions, _ = cnn_predictions(predictor, args.data_root, training_rows["validation"])
        with (run / "validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        expected = [r["predicted_label"] for r in saved]
        if expected != [LABELS[int(p)] for p in validation_predictions]:
            raise ValueError(f"Saved checkpoint failed validation prediction round-trip: {family}")
        started = time.perf_counter()
        if family == "svm":
            x = feature_matrix(args.data_root, test_rows, args.cache)
            predictions, probabilities = predictor.bundle["pipeline"].predict(x), None
        else:
            predictions, probabilities = cnn_predictions(predictor, args.data_root, test_rows)
        elapsed = time.perf_counter() - started
        result = save_predictions(folder, "test", test_rows, predictions, probabilities)
        results[family] = result
        macro = result["classification_report"]["macro avg"]
        comparisons.append({"family": family, "selected_by_validation": family == decision["selected_family"],
                            "validation_macro_f1": summary["validation_macro_f1"],
                            "test_accuracy": result["accuracy"], "test_macro_precision": macro["precision"],
                            "test_macro_recall": macro["recall"], "test_macro_f1": result["macro_f1"],
                            "best_epoch": summary.get("best_epoch", ""),
                            "training_seconds": summary["training_wall_seconds"],
                            "model_megabytes": model_path.stat().st_size / 1e6,
                            "test_pass_seconds_including_io": elapsed})
        for species in sorted({r["species"] for r in test_rows}):
            indices = [i for i, r in enumerate(test_rows) if r["species"] == species]
            diagnostic = metrics([int(test_rows[i]["label_id"]) for i in indices], [predictions[i] for i in indices])
            species_diagnostics.append({"family": family, "species_metadata": species, "test_images": len(indices),
                                        "accuracy": diagnostic["accuracy"], "macro_f1": diagnostic["macro_f1"]})
        print(f"TEST {family}: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}", flush=True)
        del predictor
    plots(args.output, args.runs, results)
    learning_curves(args.output, args.runs)
    error_examples(args.output, args.data_root, decision["selected_family"])
    write_csv(args.output / "species_diagnostics.csv", species_diagnostics)
    write_json(args.output / "evaluation.json", {"selected_family": decision["selected_family"], "test_samples": len(test_rows),
        "labels": LABELS, "limitation": LIMITATION, "single_seed_per_cnn": True, "test_time_augmentation": False,
        "checkpoint_roundtrip_validation_predictions": "all matched", "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_source_sha256": sha256_file(__file__), "results": results})
    write_csv(args.output / "comparison.csv", comparisons)


if __name__ == "__main__":
    main()
