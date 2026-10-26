# Implementation plan — 25-domain single-label text classification

| Phase | Scope | Acceptance criteria |
| --- | --- | --- |
| 0 | Package, versioned domain config, README and CI | Editable installation works on Python 3.10/3.11; lint + offline tests pass; CLI lists 25 labels |
| 1 | Bronze → Silver | Process existing 50K Bronze without downloading again; preserve text and provenance, report exclusions and duplicates |
| 2 | Silver → Gold | Stratified, deterministic and leakage-checked train/validation/test splits and stable label mapping |
| 3 | Baseline experiments | TF-IDF + Logistic Regression trained on 50K candidate corpus; compare default vs class-weighted |
| 4 | Evaluation and audits | Macro-F1, per-class F1, confusion matrix, error analysis and human audit of pseudo-labeled samples |
| 5 | Data-size and model studies | Expand targeted classes or dataset size only when baseline evidence supports it |

## Scope decisions

- Current objective: **single-label, 25 domain classes** (not the former 10-topic multi-label annotation workflow or 16-category moderation taxonomy).
- VTSNLP source is already curated; do not duplicate unneeded cleanup.
- Silver is model-agnostic; no global padding or tokenization.
- No dataset redistribution until license status is checked.
- All bulk datasets remain on Google Drive; GitHub holds only code/config/docs and small reproducibility reports.

## Next implementation change (not part of this PR)

Build `scripts/02_prepare_silver.py` with `--input`, `--output`, explicit validation, reproducible logs and tests. Do not re-ingest existing Bronze.
