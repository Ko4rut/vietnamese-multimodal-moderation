# Layer models

Each Data Lake layer defines its **JSON data contract** in `model.py`:

- `bronze/model.py`: raw `BronzeRecord`, pinned `BronzeManifest`, published
  source metadata, ingestion report and ingestion result.
- `silver/model.py`: `SilverRecord`, rejected record, minimal quality-report
  contract, extended published report, and the frozen `ValidationResult`.
- `gold/model.py`: `GoldRecord` (Silver fields + `label_id`), split metadata,
  near-duplicate audits, Gold manifest, report and pipeline result.

The models are Python `TypedDict` contracts (plus the existing frozen
`ValidationResult` dataclass). `TypedDict` **does not validate runtime JSON**;
Readers and Validators keep their current integrity and schema checks. This
refactor deliberately does **not** change the shape of published JSONL or JSON
files, filenames, cleaning, deduplication, label mapping, or Gold split algorithm.

Existing public imports continue working, including
`from vietnamese_topic_classification.data.silver.validator import ValidationResult`
and `from vietnamese_topic_classification.data.bronze import BronzeConfig`.

`GoldConfig` stays in `gold/config.py` because it is configuration rather than
a serialized data contract; `BronzeConfig` likewise stays in
`bronze/ingestor.py` to retain its existing constructor and import path.
