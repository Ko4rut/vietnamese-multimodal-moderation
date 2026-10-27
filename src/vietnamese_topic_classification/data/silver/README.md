# Silver V1 — VTSNLP

The Silver pipeline reads an existing **manifest-pinned Bronze** JSONL snapshot, verifies its
byte count, line count, and SHA-256, then emits immutable Silver outputs. It does **not** re-ingest
Hugging Face or change the Bronze file.

Run from the project root after mounting/syncing Drive and installing the package:

```bash
python scripts/02_build_silver.py --manifest-name source_manifest_b81fcce589_seed42_buf10000_n50000.json
```

Specify `DATA_LAKE_ROOT` in `.env` or pass `--data-lake-root /path/to/project`.
The data root must contain `00_source_metadata/` and `01_bronze/`.

## Contract

Accepted: source `id` and `text` are non-empty strings; `domain` is exactly one of the
25 configured labels. Invalid rows are recorded in a rejected JSONL file with their original
payload, source line and reason.

Normalization: Unicode NFC, CRLF/CR to LF, collapse repeated *horizontal* whitespace, strip
surrounding whitespace. No spelling correction, HTML/boilerplate removal, word tokenization,
truncation, padding, or label reassignment. Source labels are automatically assigned upstream.

Deduplication: preserve first valid row by ID or normalized full text. Text duplicates
with conflicting domains are rejected and counted separately. No near-duplicate detection yet.

Output: `02_silver/vtsnlp_silver_v1_*.jsonl`, `02_silver/vtsnlp_rejected_v1_*.jsonl`,
`05_reports/silver_quality_v1_*.json`. Report includes source checksum, accepted/rejected counts,
reasons, accepted label counts, character-length statistics and output SHA-256 checksums.
Reruns refuse to overwrite existing artifacts.

The pipeline stages outputs locally before copying and verifying on Drive. Leave enough free
local disk space for staged Silver and rejected data. Tests use offline fixtures.
