# Modeling V1: TF-IDF baselines

Phase 0 skeleton is now implemented. The Gold snapshot stays immutable.

## Setup

```bash
python -m pip install -e ".[dev,modeling]"
ruff check src tests
pytest -q
```

## Train / validate

```bash
python scripts/06_train_baseline.py --manifest-name gold_manifest_v1_<id>.json --name tfidf_lr_v1 --output-root /content/drive/MyDrive/experiments
python scripts/06_train_baseline.py --manifest-name gold_manifest_v1_<id>.json --name tfidf_lr_balanced_v1 --class-weight balanced --output-root /content/drive/MyDrive/experiments
python scripts/06_train_baseline.py --manifest-name gold_manifest_v1_<id>.json --name tfidf_svc_v1 --classifier linear_svc --output-root /content/drive/MyDrive/experiments
```

Pass `--data-lake-root` or set `DATA_LAKE_ROOT` in .env. The training runner loads only Gold train and validation (checksums of all three partitions are verified). TF-IDF is fitted by the sklearn Pipeline on train only; validation reports contain accuracy, macro-F1, weighted-F1, per-class F1, and a 25-class confusion matrix. Select a model based on validation Macro-F1, then run the **final test only once**:

```bash
python scripts/07_evaluate_model.py --artifact-dir /content/drive/MyDrive/experiments/tfidf_lr_v1
```

Artifacts are write-once per experiment name: model.joblib, config.json, metadata.json, validation_metrics.json; test_metrics.json is only created explicitly. **Do not load untrusted joblib/pickle models.** Gold labels are automatically assigned source labels, not human-verified ground truth.
