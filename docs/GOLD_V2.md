# Gold V2 — Silver V2 to ML-ready dataset

Gold V2 is an independent, immutable **Gold** snapshot built from a pinned, checksum-verified **Silver V2** report with a verified cleaning audit. Gold V1 and Silver V1 are never changed.

## Prerequisites

Run the Silver V2 notebook and publish an accepted `silver_quality_v2_*.json` report under `05_reports/`. The report must identify `pipeline_version=v2`, cleaning ruleset, output Silver JSONL, cleaning audit JSONL and their SHA-256 digests. Missing or tampered inputs fail before publishing.

**Quality gate:** If `review_rows` or `guardrail_rows` remains positive, Gold V2 refuses publication by default. Manually inspect the Silver V2 audit and decide whether it is acceptable to keep these unchanged records. Only then use the explicit `--allow-unreviewed` flag; its value is recorded in Gold manifest and quality report. This flag is *not* a claim that manual review was completed.

## Outputs

```text
04_gold/gold_train_v2_<source-sha>_<config-sha>.jsonl
04_gold/gold_validation_v2_<source-sha>_<config-sha>.jsonl
04_gold/gold_test_v2_<source-sha>_<config-sha>.jsonl
04_gold/manifests/gold_manifest_v2_<source-sha>_<config-sha>.json
05_reports/gold_quality_v2_<source-sha>_<config-sha>.json
```

The V2 pipeline retains the established 25-class label mapping, Gold V1's approximate group-aware near-duplicate splitter, and split configuration defaults (80/10/10, seed 42). The candidate search is heuristic, not exhaustive. It never fits TF-IDF or embeddings. A new Silver snapshot may change which samples appear in each split: **the V1 and V2 validation results are not directly comparable without a controlled evaluation protocol.**

Manifest uses `format_version=1` for compatibility with existing `GoldReader`, but `pipeline_version=v2` and V2 output filenames. The manifest and quality report include lineage to Silver V2's ruleset/audit and the decision on unreviewed records.

## Colab

Open [Colab Gold V2](../../notebooks/colab_gold_v2_runner.ipynb). The notebook checks for exactly one Silver V2 report, verifies SHA-256, checks for partial Gold publication and either skips an existing valid Gold V2 or runs the new CLI.

## CLI

```bash
python scripts/10_build_gold_v2.py \
  --data-lake-root /content/vtsnlp_drive/MyDrive/vietnamese-multilabel-topic-classification \
  --report-name silver_quality_v2_<actual-run-id>.json

# Explicit exception after human review:
python scripts/10_build_gold_v2.py \
  --data-lake-root /path/to/lake \
  --report-name silver_quality_v2_<actual-run-id>.json \
  --allow-unreviewed
```

Do not retrain on Gold V2 without inspecting the new quality report and frozen evaluation plan.
