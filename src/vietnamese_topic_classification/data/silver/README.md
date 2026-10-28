# Silver V1 — VTSNLP

Silver V1 converts one **manifest-pinned, verified Bronze** snapshot into an auditable dataset.
The source is Vietnamese text with exactly one of 25 VTSNLP source-domain labels.
It **does not download the dataset again**, change Bronze, retrain labels or create Gold splits.

## 1. Input and prerequisites

| Item | Input / requirement |
| --- | --- |
| Data Lake root | An existing directory with `00_source_metadata/` and `01_bronze/` |
| Manifest | `00_source_metadata/source_manifest_b81fcce589_seed42_buf10000_n50000.json` |
| Dataset | The `file_name` from the manifest, inside `01_bronze/` |
| Expected source fields | `id`, `text`, `domain` |
| Label vocabulary | 25 values from `config/domains_v1.yaml` |
| Drive access | Google Drive mounted in Colab or made available on the local filesystem |
| Environment | `DATA_LAKE_ROOT` in `.env` or an explicit `--data-lake-root` CLI argument |

The pipeline first runs `BronzeReader.verify()` to check **file size, line count and SHA-256**
against the manifest. If that fails, no Silver output is published.

## 2. What Silver V1 actually does

| Stage | Rule used by the code | Result |
| --- | --- | --- |
| 1. Schema validation | `id` and `text` must be non-empty strings (not just whitespace); `domain` must match one of 25 configured labels | Invalid source rows go to rejected JSONL |
| 2. Text normalization | Unicode **NFC**, normalize CRLF/CR to LF, collapse horizontal whitespace, trim surrounding whitespace | Produce a minimally standardized `text` |
| 3. Duplicate ID detection | Compare `id.strip()` of valid records | Keep first accepted ID; reject subsequent duplicate IDs |
| 4. Exact text deduplication | Compare SHA-256 of the *complete normalized text* | Keep first accepted text; reject subsequent identical texts |
| 5. Label conflict tracking | Exact same normalized text but a different domain | Reject as `duplicate_text_conflicting_domain` (do not auto-relabel) |
| 6. Metadata enrichment | Attach source filename and normalized character length | Improve traceability |
| 7. Publish and audit | Stage files locally, verify copied files, then publish without overwriting | Accepted data, rejected records, quality report |

**Order matters:** rows are processed in Bronze order, and only **accepted** rows register an ID/text in
the deduplication sets. The first accepted row is retained. Conflicting labels are *flagged as a
rejection reason*, not resolved by a majority vote.

### Record rejection reasons

| Reason | Meaning |
| --- | --- |
| `invalid_id` | Missing, non-string, or whitespace-only `id` |
| `invalid_text` | Missing, non-string, or whitespace-only `text` |
| `invalid_domain` | Missing or unknown `domain` |
| `empty_after_normalization` | Text is empty after normalization |
| `duplicate_id` | An accepted row already has the same trimmed ID |
| `duplicate_text` | Normalized text is identical to an accepted row with the same domain |
| `duplicate_text_conflicting_domain` | Identical normalized text, different source domain |

An individual rejected row is **recorded**, not raised as a fatal exception. Errors accessing files,
invalid Bronze manifests and publication failures can stop the pipeline.

## 3. Output contracts

### Accepted row (`02_silver/vtsnlp_silver_v1_*.jsonl`)

| Field | Meaning |
| --- | --- |
| `id` | Original source ID (the value is retained) |
| `text` | Minimally normalized text, with paragraphs preserved |
| `domain` | Unchanged automatically assigned source label |
| `text_length` | Character count of normalized text |
| `source_snapshot` | Name of the source Bronze JSONL file |
| `quality_flags` | Empty list in V1 (no per-record heuristics implemented yet) |

### Rejected row (`02_silver/vtsnlp_rejected_v1_*.jsonl`)

| Field | Meaning |
| --- | --- |
| `record` | Original record from Bronze |
| `reason` | One rejection reason from the table above |
| `source_line` | 1-based source record position |
| `source_snapshot` | Bronze file from which the row came |

### Quality report (`05_reports/silver_quality_v1_*.json`)

Contains `input_rows`, `accepted_rows`, `rejected_rows`, `rejection_reasons`,
`accepted_domain_counts`, text-length **min/max/mean**, source manifest name and checksum,
pipeline version, output filenames, SHA-256 of accepted/rejected outputs, and run time.

The accounting invariant is:

```text
input_rows = accepted_rows + rejected_rows
```


**No guarantee of exactly 50,000 Silver rows:** Bronze has 50,000 rows, but schema failures and
duplicates can reduce the accepted count.

## 4. Run Silver

### Local machine / VS Code

From the repository root:

```bash
python -m pip install -e ".[dev,data]"
python -m pytest -q
python scripts/02_build_silver.py --manifest-name source_manifest_b81fcce589_seed42_buf10000_n50000.json
```


Set `DATA_LAKE_ROOT` to an existing *mounted or synchronized* Data Lake folder first. Alternatively
pass `--data-lake-root /absolute/path/to/project`. The root is **not** a Drive URL.

The project uses a **`src/` layout**: the import is
`from vietnamese_topic_classification.data.bronze import BronzeReader`, **not** `import src`.
The Colab notebook performs an editable pip install and then inserts
`/content/project/src` into the running kernel's `sys.path`. This avoids an immediate
`ModuleNotFoundError` when the kernel has not yet loaded editable-install paths.
When invoking the Silver CLI as a child Python process, the notebook also sets
`PYTHONPATH` to include that `src` directory.

### Google Colab (recommended for the existing Drive snapshot)

Run the dedicated notebook:
[**notebooks/colab_silver_runner.ipynb**](../../../../notebooks/colab_silver_runner.ipynb).

It performs these steps: fetch the Silver feature branch → install package → mount the **correct**
Google account → check manifest/Bronze and existing outputs → run the CLI → verify published
counts and checksums. It **never runs** `01_ingest_bronze.py`.

The repository is **public**. The notebook clones the `feat/silver-pipeline-v1` branch
directly over HTTPS, so **no GitHub token, login or Colab Secret is required**.

## 5. Output locations and rerun behavior

```text
DATA_LAKE_ROOT/
├── 00_source_metadata/
│   └── source_manifest_...json
├── 01_bronze/
│   └── vtsnlp_raw_sample_...jsonl       (unchanged)
├── 02_silver/
│   ├── vtsnlp_silver_v1_...jsonl         (accepted)
│   └── vtsnlp_rejected_v1_...jsonl       (rejected)
└── 05_reports/
    └── silver_quality_v1_...json        (audit report)
```


Published snapshot files are **not overwritten**. A rerun with existing Silver artifacts or stale
`.uploading` files stops intentionally. Inspect the existing run before choosing a new version/snapshot;
do not delete the Bronze snapshot to rerun Silver.

For publishing, Silver uses temporary local disk, then copies and verifies files on Drive.
Allow enough temporary disk space (especially with the 50K sample).

## 6. Deliberately out of scope in V1

- **No** spelling repair, Vietnamese word segmentation, tokenization, stop-word removal, truncation, or padding.
- **No** HTML/ad/sidebar/boilerplate removal; web-crawl artifacts remain until quality analysis motivates safe rules.
- **No** approximate/near-duplicate detection or semantic label correction.
- **No** class balancing, training, label encoding, train/validation/test splits (those belong to Gold/modeling).
- No manual label validation: VTSNLP domains are automatically assigned upstream, not verified ground truth.

Source: `validator.py`, `transformer.py`, `pipeline.py`. Offline fixtures: `tests/test_silver.py`.


## 7. Read an existing Silver snapshot

Use `SilverReader` after the Silver pipeline has published its quality report.
The reader resolves the accepted JSONL filename from `output_files.silver` in that report
and checks `silver_sha256` and `accepted_rows` before the preview script reads rows.

```bash
python scripts/03_read_silver.py \
  --data-lake-root /path/to/data-lake \
  --report-name silver_quality_v1_b81fcce589_seed42_n50000.json \
  --limit 3
```

Use the actual report filename in `05_reports/`. The command only verifies and previews
the accepted Silver rows; it does not re-run Silver, read rejected records, or modify artifacts.
For code usage:

```python
from vietnamese_topic_classification.data.silver import SilverReader

reader = SilverReader(data_lake_root, metadata_name)
reader.verify()
for record in reader.iter_records():
    ...  # Build downstream Gold/modeling inputs
```


## Shared Reader interface

`BronzeReader` and `SilverReader` both accept `(data_lake_root, metadata_name)`. Bronze resolves
the name under `00_source_metadata/`, while Silver resolves it under `05_reports/`. This
unifies the Python API without changing already-published snapshots.

To run the shared read-only smoke check:

```bash
python -m vietnamese_topic_classification.data.main \
  --layer silver \
  --metadata-name silver_quality_v1_b81fcce589_seed42_buf10000_n50000.json \
  --limit 3
```
