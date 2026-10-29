# Colab Modeling V1 — train and compare TF-IDF baselines

Open [notebooks/colab_modeling_runner.ipynb](../notebooks/colab_modeling_runner.ipynb) from branch `feat/colab-modeling-baselines-v1` using Google Colab (GitHub integration or upload notebook).

Runs two CPU-friendly, sequential pipelines from the **existing immutable Gold V1** snapshot:

- TF-IDF + multinomial Logistic Regression (multiclass with `predict_proba`).
- TF-IDF + LinearSVC (multiclass decision scores, not probabilities).
- Optional third experiment: TF-IDF + Logistic Regression with `class_weight=balanced` (set `RUN_BALANCED=True`).

The notebook checks out the branch, installs `.[modeling]`, mounts the same Drive path used by the Gold Colab runner, discovers one manifest (or asks you to select among several), verifies Gold checksums, runs `scripts/06_train_baseline.py` per experiment, and compares validation Macro-F1. Train/validation are used; Gold test is not fit or evaluated. Each experiment is stored once under `DATA_LAKE_ROOT/experiments/tfidf_baselines_v1/<name>/`. Complete existing outputs are verified and reused; partial/mismatching outputs fail loudly.

No GPU required. No dataset upload to GitHub. Do not run `scripts/07_evaluate_model.py` until the baseline has been selected on validation; final test evaluation is a separate explicit step. Existing VTSNLP domain labels are automatically assigned and have not been manually verified.
