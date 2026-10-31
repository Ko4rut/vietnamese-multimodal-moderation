# Model diagnostics: TF-IDF sklearn baselines

`LogisticRegression(max_iter=1000)` and `LinearSVC` use solver iterations, **not epochs**. The initial V1 models fit once and save only final validation metrics; there are no per-iteration train/validation metrics. Their solvers may converge early based on tolerance; V1 does **not** implement validation Macro-F1 early stopping.

After fitting via `scripts/06_train_baseline.py`, create reproducible **post-fit** diagnostics for each trusted model on Gold train/validation only:

```bash
python -m pip install -e ".[dev,modeling]"
python scripts/08_plot_diagnostics.py \
  --data-lake-root /content/vtsnlp_drive/MyDrive/vietnamese-multilabel-topic-classification \
  --artifact-dir /content/vtsnlp_drive/MyDrive/vietnamese-multilabel-topic-classification/experiments/tfidf_baselines_v1/<experiment_name>
```

Or open `notebooks/v1/colab_modeling_runner.ipynb` on branch `feat/modeling-diagnostics-v1` and run its **diagnostics** cell. Existing trained models are **not overwritten**, and the Gold test split is not loaded into predictions.

One write-once `diagnostics_v1/` folder per experiment contains:

- `train_vs_validation.png`: full-model train and validation accuracy/Macro-F1/weighted-F1.
- `learning_curve_by_train_size.png`: separate independent refits of sklearn Pipeline on stratified 25/50/75/100%-sized subsets of Gold train. X-axis is *training data size*, not epochs.
- `correct_counts_by_class.png`: correct and incorrect counts for every true class (affected by support).
- `recall_by_class.png`: per-class recall, normalized for class support.
- `per_class_f1.png`: F1 of each of the 25 labels.
- `confusion_matrix_normalized.png`: row-normalized confusion matrix.
- `diagnostics.json`: full-model train vs validation scores/gaps, each learning-curve point, per-class statistics, the estimator's `n_iter_` when available.

The learning curve re-trains 3 additional models per experiment. This is **not** the original training history. Refit subset models use only train data; validation is used for measurement. Original models and experiment metadata are not modified. `diagnostics_v1/` is write-once and must not be overwritten; model artifacts must be trusted before joblib loading. A positive train-minus-validation gap may indicate overfitting, but size and pattern need interpretation and repeated runs for robust conclusions.

**Early stopping** based on validation Macro-F1 cannot be retrofitted into historical LBFGS and liblinear training runs with these files. It requires an explicitly different incremental training algorithm/loop with patience and a checkpointed best model (e.g. an SGDClassifier experiment). Do not mislabel solver iterations as epochs.
