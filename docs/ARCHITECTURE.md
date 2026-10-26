# Architecture — Vietnamese Text Topic Classification

## Current task

Single-label multiclass classification of Vietnamese text into **25 VTSNLP source domains**. Each record contains `id`, `text`, `domain`. Labels were assigned automatically upstream and must not be confused with verified human annotations.

## Hybrid execution boundary

```text
VS Code (code) → GitHub (versioning) → Google Colab (execution)
                                         ↕
                          Google Drive (Bronze/Silver/Gold/reports)
```

Data remains on Google Drive; credentials and bulk data do not belong in the repository.

## Data stages (planned; not implemented)

1. **Bronze:** immutable 50K source JSONL sample and source manifest.
2. **Silver:** schema checks, minimal Unicode NFC normalization, exact deduplication, provenance/length metadata. Record the rejected/flagged rows and counts. Avoid aggressive cleaning or truncation simply to fit a model.
3. **Gold:** fixed, stratified train/validation/test partitions and stable label mapping; check exact and near-duplicate leakage before modeling. Keep evaluation sets fixed across experiments.
4. **Model experiments:** TF-IDF + Logistic Regression baseline; optionally class weighting and later Transformers. Apply tokenization and padding only during model-specific processing.
5. **Evaluation:** Macro-F1 (primary), per-domain F1, Accuracy, confusion matrix, and manual review of a subset due to pseudo-label noise.

## Package boundaries

- `config/`: canonical versioned 25-domain configuration and runtime environment settings.
- `data/`: future preparation and split implementation.
- `training/`: future model implementation.
- `evaluation/`: future evaluation and reporting.
- `scripts/`: future thin CLI entry points for running those modules on Colab.

Legacy multimodal moderation documents and demo placeholders may remain in the repository for reference but are out of scope for the current model.

No dataset pipeline, model, training or prediction implementation is included in the current package/CI update.
