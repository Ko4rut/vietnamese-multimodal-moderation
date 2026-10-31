# Bronze ingestion

Run this after mounting the **correct Google Drive** in Colab:

```bash
python -m pip install -e ".[data]"
python scripts/01_ingest_bronze.py \
  --output-root "/content/drive/MyDrive/vietnamese-multilabel-topic-classification" \
  --sample-size 50000 --seed 42 --shuffle-buffer 10000
```

The output root is illustrative; inspect the actual mounted path first.

Bronze writes:
- \`01_bronze/vtsnlp_raw_sample_<revision>_seed<seed>_buf<buffer>_n<N>.jsonl\`
- \`00_source_metadata/source_manifest_<...>.json\`
- \`04_reports/bronze_ingestion_<...>.json\`

The ingestion source revision is pinned in \`config/domains_v1.yaml\`.
Source values are not cleaned, filtered, balanced or padded at this stage.
The class writes to a local temporary directory first, verifies copy checksums,
and refuses to overwrite an existing snapshot.

The old 50K Bronze file uses a different legacy naming scheme; this run
creates a new snapshot rather than modifying the old one.

Python package layout:
- `vietnamese_topic_classification.data.bronze.ingestor` provides `BronzeIngestor`.
- `vietnamese_topic_classification.data.bronze.reader` provides `BronzeReader`.
- `vietnamese_topic_classification.data.bronze.exceptions` provides Bronze-specific exceptions.
- Public imports such as `from vietnamese_topic_classification.data.bronze import BronzeIngestor, BronzeReader` are supported.

## Silver EDA notebook

Install `python -m pip install -e ".[dev,eda]"` (or `".[eda]"` without test tools). The `eda` extra includes JupyterLab, IPython, an IPython kernel, notebook format validation, Pandas, NumPy, Matplotlib, and scikit-learn. Run `python scripts/04_generate_silver_eda_notebook.py` to regenerate `notebooks/v1/eda/01_silver_eda.ipynb`; `--check` verifies that it is up to date. The notebook uses `SilverReader` and generates multiple inline charts plus optional read-only Bronze comparison. It does not write outputs by default.


## Gold V1 (new feature branch)

After verifying an existing Silver snapshot, build immutable Gold train/validation/test:

    python scripts/05_build_gold.py --report-name silver_quality_v1_b81fcce589_seed42_buf10000_n50000.json

Requires DATA_LAKE_ROOT (mounted/synced local Data Lake), or pass --data-lake-root.
Gold stores per-split checksums, source lineage, fixed 25-domain label mapping and
a separate quality report. See src/vietnamese_topic_classification/data/gold/README.md.

## Colab Gold runner

Use `notebooks/v1/colab_gold_runner.ipynb` from `feat/gold-pipeline-v1`.
It mounts the correct Drive account, verifies the existing Silver snapshot,
checks for conflicting immutable Gold files, runs `scripts/05_build_gold.py`
and verifies the completed Gold dataset. Install just the base package; Gold
needs neither the `data` nor the `eda` dependency extras.
