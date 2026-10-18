"""Register the delivered species and freshness models once at service startup."""
import logging
import os
from pathlib import Path

from fish_species.final_predict import FinalSpeciesPredictor
from .adapters import LABELS, MODELS, register

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = Path(os.environ.get("FISHLENS_SPECIES_MODELS_DIR", ROOT / "models" / "species"))
FRESHNESS_ROOT = Path(os.environ.get("FISHLENS_FRESHNESS_MODELS_DIR", ROOT / "models" / "freshness"))
DEVICE = os.environ.get("FISHLENS_DEVICE", "cpu")
# Keep third-party caches inside this project's ignored runtime directory.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

available = [family for family in MODELS if (MODEL_ROOT / family / "selection.json").is_file()]
freshness_available = (FRESHNESS_ROOT / "resnet18" / "selection.json").is_file()
if available or freshness_available:
    import torch
    # Avoid CPU oversubscription for small, interactive four-view batches.
    torch.set_num_threads(int(os.environ.get("FISHLENS_CPU_THREADS", "1")))

for family in MODELS:
    if family not in available:
        logging.warning("Species model %s is not installed in %s", family, MODEL_ROOT)
        continue
    predictor = FinalSpeciesPredictor(MODEL_ROOT / family, device=DEVICE)
    if predictor.selection["family"] != family:
        raise ValueError(f"Model folder {family} contains a different architecture")
    if predictor.selection["labels"] != LABELS["species"]:
        raise ValueError(f"Model {family} has an unexpected species label order")
    register("species", family, predictor)

if freshness_available:
    from fish_freshness.final_predict import FinalFreshnessPredictor
    freshness_predictor = FinalFreshnessPredictor(FRESHNESS_ROOT / "resnet18", device=DEVICE)
    if freshness_predictor.selection["labels"] != LABELS["freshness"]:
        raise ValueError("Unexpected freshness label order")
    register("freshness", "resnet18", freshness_predictor)
else:
    logging.warning("FFE ResNet-18 is not installed in %s. Run scripts.install_freshness_model.", FRESHNESS_ROOT)
