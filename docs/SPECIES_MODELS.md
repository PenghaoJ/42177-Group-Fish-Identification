# Delivered species models

The 2026-10-18 `completed_species_models` delivery targets backend commit
`df9197acc3be241b890fef700136209d9ce1ba37`. This integration preserves its
inference source, class order, preprocessing, checkpoint hashes and validation
selection. It does not retrain models or change the fixed dataset split.

## Local setup

Use Python 3.12 and install `requirements.txt` in the project's `.venv`.
`Start-FishLens.cmd` reuses that environment and opens the website;
`Stop-FishLens.cmd` stops its background server.

The delivery is archived in the group's Google Drive:
<https://drive.google.com/drive/folders/1lWDdiUSdg_R4J1bKiLKUK3nDc7sXbY_M>.
Download `completed_species_models.zip` and place the following pairs of files:

The ZIP retains its existing Drive permissions; request access from its owner
if needed. Install its verified model files automatically (Python standard
library only, no archive scripts are executed):

```powershell
.venv/Scripts/python -m scripts.install_species_models --archive "PATH/completed_species_models.zip"
```

Alternatively, copy the following pairs manually:

| Delivery folder | Local folder | Files |
| --- | --- | --- |
| `artifacts/species_final/` | `models/species/resnet18/` | `model.pt`, `selection.json` |
| `artifacts/species_comparison/svm/` | `models/species/svm/` | `model.joblib`, `selection.json` |
| `artifacts/species_comparison/yolov8n_cls/` | `models/species/yolov8n_cls/` | `model.pt`, `selection.json` |

These files are installed in this local checkout, but ignored by Git. A new clone
needs the separate model download. Only load the trusted group delivery: the SVM
uses Python/joblib serialization. Each checkpoint's SHA-256 is checked against
its selection file, and the adapter verifies architecture and class order.

`backend/config.py` selects `resnet18` for species and `None` for freshness.
All three installed species models are registered; change the species value to
`svm` or `yolov8n_cls` and restart to use an alternative. The frontend deliberately
does not let users choose models. An absent selected model returns HTTP 503.

Optional environment variables:

- `FISHLENS_SPECIES_MODELS_DIR`: alternate root containing the three family folders.
- `FISHLENS_DEVICE`: defaults to `cpu`.
- `FISHLENS_CPU_THREADS`: defaults to `1` for interactive CPU inference.

## Preserved prediction contract

Input is one PIL RGB image of a complete fish. Output contains `label`, `score`
and `model_version`. Labels in checkpoint order are `horse_mackerel`, `whiting`,
`haddock`, `cod`, `hake`. SVM returns `score: null`; CNN scores are uncalibrated.
CNNs preserve the entire image by proportional resize and letterbox to 224 x 224.
ResNet-18 applies ImageNet normalization; YOLO uses values in [0,1]. Both delivered
CNNs average original, horizontal, vertical and both-flipped views (`flip4`).
The SVM's color/texture extraction excludes letterbox padding. No mask is needed.

## Verification

Run API contract tests with `.venv/Scripts/python -m unittest discover -s tests -v`.
They isolate configuration and registrations so tests for demo and unavailable
models remain valid when real weights are installed.

Real-model checks must additionally use original dataset images, verify their
manifest hashes and compare predictions to the delivered records. An integration
smoke check is not a new accuracy study or evidence about unseen fish species.
Freshness remains a separate task and its model has not been connected here.

With the original dataset and unpacked delivery available, run:

```powershell
.venv/Scripts/python -m scripts.verify_species --data-root "PATH/42177_selected_data" --reference-root "PATH/package/artifacts/species_comparison" --per-class 10 --url http://127.0.0.1:8765
```

The report defaults to `.runtime/species-verification.json`; five example images
are copied to `.runtime/species-samples/` for browser uploads.

Verified on Windows / Python 3.12 / CPU on 2026-10-18:

- All six API contract tests pass after isolating installed model registrations.
- 50 held-out images, ten per class from 44 recorded fish, pass hash checks.
- All 150 predictions (50 per model) reproduce the delivered labels. Maximum
  CNN score differences are below 0.000001 across the two machines.
- ResNet-18 and YOLO match 50/50 ground-truth labels in this smoke subset; SVM
  matches 47/50, reproducing its existing errors. This is not the full test metric.
- The live API correctly processes one image from each of the five classes.
- The browser's upload/analyze flow shows Cod / Gadus morhua with score 0.952.
- Startup, stop and restart work through the Windows launcher. The independently
  trained freshness ResNet-18 is now connected; see [freshness setup](FRESHNESS_MODEL.md).

Loading Ultralytics patches Pillow's global image opener. The server retains
Pillow's original JPEG/PNG decoder before model loading, so corrupt uploads
remain HTTP 400 instead of attempting optional HEIF imports and returning 500.

The copied `fish_species` modules are from the supplied group delivery. Dataset
attribution: AutoFish, Bengtson et al. (2025), CC BY 4.0,
<https://huggingface.co/datasets/vapaau/autofish>.
