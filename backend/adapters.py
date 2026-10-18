"""Model integration point. Keep preprocessing inside each predictor."""
SPECIES = {
    "horse_mackerel": "Trachurus trachurus", "whiting": "Merlangius merlangus",
    "haddock": "Melanogrammus aeglefinus", "cod": "Gadus morhua", "hake": "Merluccius merluccius",
}
LABELS = {"species": list(SPECIES), "freshness": ["highly_fresh", "fresh", "not_fresh"]}
MODELS = ["svm", "resnet18", "yolov8n_cls"]
PREDICTORS = {}

def register(task, model, predictor):
    """predictor(PIL RGB image) -> {label, score, model_version}.

    Scores must be finite numbers in [0,1], or None if unavailable. A score
    is not a calibrated probability. Register two predictors per model family.
    Load weights once, not for every request. Preserve training label order.
    """
    if task not in LABELS or model not in MODELS:
        raise ValueError("Unknown task or model")
    PREDICTORS[(task, model)] = predictor

def predict(task, model, image, mode):
    if mode == "demo":
        # Intentionally fixed: this is a UI example, never an image prediction.
        return {"label": LABELS[task][0], "score": None, "model_version": "demo-fixture-v1"}
    predictor = PREDICTORS.get((task, model))
    if predictor is None:
        raise LookupError("The selected trained model is unavailable. Install its weights and restart FishLens.")
    return predictor(image)

