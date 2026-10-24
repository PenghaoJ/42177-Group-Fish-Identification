# Training

The FFE freshness training package is `fish_freshness/`. Follow
[the Chinese training guide](../docs/FRESHNESS_TRAINING_ZH.md) to audit the fixed
FFE splits, train SVM / ResNet-18 / YOLOv8n-cls, and evaluate the frozen models.

Run all commands from the repository root. Weights, caches, the GPU environment
and experiment outputs live in the ignored `.training/` directory. Dataset
images stay outside Git. Each run records package versions, seeds, label order,
preprocessing, source hashes and manifest hashes. Training entry points do not
evaluate test images; the evaluation command freezes validation-based model
selection before making test predictions.

The species inference package and the website's selected species model are
independent of freshness training.
