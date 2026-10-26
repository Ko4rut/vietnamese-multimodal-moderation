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
- \`05_reports/bronze_ingestion_<...>.json\`

The ingestion source revision is pinned in \`config/domains_v1.yaml\`.
Source values are not cleaned, filtered, balanced or padded at this stage.
The class writes to a local temporary directory first, verifies copy checksums,
and refuses to overwrite an existing snapshot.

The old 50K Bronze file uses a different legacy naming scheme; this run
creates a new snapshot rather than modifying the old one.
