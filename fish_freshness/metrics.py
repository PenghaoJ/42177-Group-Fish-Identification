"""Image-level metrics in the fixed FFE label order."""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .data import LABELS, write_csv, write_json


def metrics(truth, predicted):
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
            "classification_report": classification_report(truth, predicted, labels=[0, 1, 2],
                target_names=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist()}


def save_predictions(out, split, rows, predicted, probabilities=None):
    result = metrics([int(r["label_id"]) for r in rows], predicted)
    write_json(out / f"{split}_metrics.json", result)
    records = []
    for i, row in enumerate(rows):
        record = {"sample_id": row["sample_id"], "file_path": row["file_path"], "species_metadata": row["species"],
                  "true_label": row["label"], "predicted_label": LABELS[int(predicted[i])]}
        if probabilities is not None:
            record.update({f"score_{label}": float(probabilities[i][j]) for j, label in enumerate(LABELS)})
        records.append(record)
    write_csv(out / f"{split}_predictions.csv", records)
    return result
