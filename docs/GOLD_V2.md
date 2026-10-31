# Gold V2 — verified Silver V2 to ML-ready splits

Gold V2 is an independent, immutable experiment derived from a **complete, checksum-verified Silver V2** snapshot. It reuses the deterministic, group-aware near-duplicate splitter, 25-domain label mapping, and Gold record schema from V1. It **does not rerun** Bronze ingestion, Silver cleaning, TF-IDF, or model training.

## Input and safety

- Select one specific published `silver_quality_v2_*.json` in `04_reports/silver/v2/`.
- Validate that the Silver V2 accepted JSONL, rejected JSONL, audit JSONL and final report exist. Verify SHA256 values and counts; follow the source report and checksum back to Silver V1.
- A missing/partial Silver V2 publication is **not** repaired or silently overwritten. Fix or migrate its publication using a dedicated audited process first.
- Write Gold V2 artifacts only into dedicated versioned directories. Existing Gold V1 snapshot paths are never modified.
- Split once per source/report/config. Existing verified results can be reused; incomplete results fail closed.
- The splitter uses approximate candidate retrieval followed by exact Jaccard verification. **Near-duplicate detection is not exhaustive**; label quality also remains unverified.

## Layout

```text
DATA_LAKE_ROOT/
  02_silver/v2/vtsnlp_silver_v2_*.jsonl
  04_reports/silver/v2/silver_quality_v2_*.json
  03_gold/v2/
    gold_train_v2_*.jsonl
    gold_validation_v2_*.jsonl
    gold_test_v2_*.jsonl
    manifests/gold_manifest_v2_*.json
  04_reports/gold/v2/gold_quality_v2_*.json
```

Gold V2 records preserve `id`, cleaned `text`, `domain`, `text_length`, `source_snapshot`, `quality_flags`, plus canonical integer `label_id`. Output manifest and report record the Silver V2 source and its Silver V1 lineage, split config, checksums, class counts, and approximate leakage audit.

## CLI

```bash
python -m pip install -e ".[dev]"
python scripts/10_build_gold_v2.py \
  --data-lake-root /path/to/mounted/data-lake \
  --report-name silver_quality_v2_YOUR_SNAPSHOT.json
```

## Colab

Open [`notebooks/v2/colab_gold_v2_runner.ipynb`](../notebooks/v2/colab_gold_v2_runner.ipynb). It clones the chosen branch (defaults to `main` after merging), mounts Drive, lists source Silver V2 reports, checks source completeness, and opts into Gold V2 build only when `PUBLISH_GOLD_V2=True`. No GPU required.

**Modeling:** the existing modeling CLI consumes Gold manifest filenames through `GoldReader`, which now resolves V2 manifests in `03_gold/v2/manifests/`. Use distinct V2 experiment names; do not mix validation/test results between V1 and V2 when comparing cleaning effects.
