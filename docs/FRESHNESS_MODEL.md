# FFE freshness model integration

The website uses the validation-selected **ResNet-18** for independent fish-eye
freshness classification. Species prediction is not required first. Upload an
already-cropped eye close-up; this model does not locate eyes in whole-fish photos.

## Install

Obtain `42177_FFE_freshness_baseline_2026-10-18.zip` from the team's freshness
training delivery. The code repository does not include weights. No retraining
or GPU is required to use the website.

Run from the repository root after setting up `.venv`:

```powershell
.venv\Scripts\python.exe -m scripts.install_freshness_model --archive "C:\path\to\42177_FFE_freshness_baseline_2026-10-18.zip"
```

On macOS/Linux use `.venv/bin/python` with the same module and arguments.
Stop and restart the service after installing. Select **Eye freshness**, upload
a JPG/PNG eye photograph and click **Analyze image**. The result must show
**Model output**, without a demo banner.

The installer checks the frozen validation selection and the expected checkpoint
SHA-256 before writing two files:

```text
models/freshness/resnet18/model.pt
models/freshness/resnet18/selection.json
```

The expected weight SHA-256 is
`47436cb20198b32d711d7f1e8d677c4eb901e44fa55eb053a9389ed338296778`.
Repeating installation with the same files is safe; different existing files are
never overwritten. ZIP contents other than the selection record and selected
checkpoint are not extracted or executed.

`FISHLENS_FRESHNESS_MODELS_DIR` may override `models/freshness`.
`FISHLENS_DEVICE` defaults to `cpu`. The launcher and website do not use the
separate `.training/venv` GPU environment.

## Inference contract

- Label order: `highly_fresh`, `fresh`, `not_fresh`.
- EXIF correction, RGB conversion, proportional 224×224 black letterbox,
  ImageNet channel normalisation; no random crop or test-time augmentation.
- Model version: `resnet18-ffe3-seed42177-epoch20`.
- Softmax score is uncalibrated; it is not a guarantee or an estimate of storage days.
- Weights load once at startup, with SHA-256 and metadata checks. Calls share a lock.
- A configured but missing model returns HTTP 503; a corrupt model causes a startup
  error. Neither condition substitutes demonstration output.

All three freshness models remain in the training delivery for comparison.
Only the selected ResNet-18 is registered in the website for this integration.

## Evaluation and limitations

On the fixed 654-image test split, ResNet-18 achieved 79.51% accuracy and 79.41%
Macro-F1. See [the recorded baseline](../evaluation/FFE_BASELINE_2026-10-18.md)
and [training guide](FRESHNESS_TRAINING_ZH.md). The missing specimen IDs mean that
individual-fish independence cannot be verified. These results describe a
single-seed dataset experiment, not a field validation or food-safety assessment.

## Integration verification

```powershell
.venv\Scripts\python.exe -m scripts.verify_freshness --data-root "C:\path\to\42177_selected_data" --reference-root "C:\path\to\2026-10-18_ffe_baseline" --url http://127.0.0.1:8765
```

This checks the actual API against the delivered checkpoint on balanced validation
images, including label/score/version, explanation and JPEG upload. It preserves
the original evaluation; it does not reselect a model using the test set.

The Windows CPU integration run matched all 30 validation-image labels against
the delivery, and live API outputs matched local CPU inference. The 15 automated
tests passed, including missing-weight errors, corrupted model rejection,
idempotent installation and preservation of different existing files. A browser
upload displayed a real `fresh` prediction with a score of 0.734. Existing species
predictors also passed their five-image-per-model integration regression.
