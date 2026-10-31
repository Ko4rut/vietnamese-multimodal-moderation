# Colab Modeling V1 — train and compare TF-IDF baselines

Open [notebooks/v1/colab_modeling_runner.ipynb](../notebooks/v1/colab_modeling_runner.ipynb) from branch `main` (or set `VTC_GIT_REF` to the review branch) using Google Colab (GitHub integration or upload notebook).

Runs two CPU-friendly, sequential pipelines from the **existing immutable Gold V1** snapshot:

- TF-IDF + multinomial Logistic Regression (multiclass with `predict_proba`).
- TF-IDF + LinearSVC (multiclass decision scores, not probabilities).
- Optional third experiment: TF-IDF + Logistic Regression with `class_weight=balanced` (set `RUN_BALANCED=True`).

The notebook checks out the branch, installs `.[modeling]`, mounts the same Drive path used by the Gold Colab runner, discovers one manifest (or asks you to select among several), verifies Gold checksums, runs `scripts/06_train_baseline.py` per experiment, and compares validation Macro-F1. Train/validation are used; Gold test is not fit or evaluated. Each experiment is stored once under `DATA_LAKE_ROOT/experiments/tfidf_baselines_v1/<name>/`. Complete existing outputs are verified and reused; partial/mismatching outputs fail loudly.

No GPU required. No dataset upload to GitHub. Do not run `scripts/07_evaluate_model.py` until the baseline has been selected on validation; final test evaluation is a separate explicit step. Existing VTSNLP domain labels are automatically assigned and have not been manually verified.

For a single run from Bronze ingestion through both V1 baselines, open [V1 end-to-end Colab](../notebooks/v1/colab_end_to_end_v1.ipynb).

## Modeling V2 — Gold V2 on the same training engine

Open [the V2 Colab modeling runner](../notebooks/v2/colab_modeling_runner.ipynb). This reads **only** `03_gold/v2/manifests/gold_manifest_v2_*.json` and trains fresh models under `experiments/tfidf_baselines_v2/`. It uses the unchanged `training.factory.build_pipeline()` and `scripts/06_train_baseline.py`, not V1's trained model artifacts.

Default runs on Colab CPU:

- TF-IDF (1–2 grams, max 100,000 features, min_df=2) + Logistic Regression.
- TF-IDF (same params) + LinearSVC.
- Optional third run: `RUN_BALANCED=True` to add class-weight-balanced Logistic Regression.
- `RUN_DIAGNOSTICS=False` by default avoids expensive refit-based charts. Enable it to generate the existing `diagnostics_v1/` **format** on V2 experiments (folder name is the diagnostics format version).

For a local CLI run (example; choose your actual Gold V2 manifest filename and mounted lake path):

```bash
python -m pip install -e ".[modeling]"
python scripts/06_train_baseline.py \
  --data-lake-root /path/to/lake \
  --manifest-name gold_manifest_v2_YOUR_SNAPSHOT.json \
  --output-root /path/to/lake/experiments/tfidf_baselines_v2 \
  --name tfidf_logreg_v2_trial \
  --classifier logistic_regression \
  --class-weight none \
  --max-features 100000 --min-df 2 --seed 42
```

Use a distinct experiment name for each changed parameter set. Training only reads Gold **train** and **validation** records; the test split is checksum-verified but excluded from fitting and selection. Complete existing artifact sets are checked for matching manifest/config/lineage and skipped; partial or mismatched directories are rejected rather than overwritten.

The notebook optionally displays a **descriptive** side-by-side V1 and V2 validation Macro-F1 when there is precisely one compatible V1 experiment per classifier. Because cleaning can alter text and Gold split membership, **do not interpret V1→V2 delta as a controlled cleaning improvement**; a common, frozen evaluation dataset is necessary for causal comparisons.
