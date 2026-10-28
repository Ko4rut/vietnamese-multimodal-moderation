# Gold V1 — ML-ready topic classification splits

**Source:** one immutable Silver accepted JSONL snapshot pinned by its existing quality report
under DATA_LAKE_ROOT/05_reports/. Gold does not download data, clean text again,
tokenize, produce TF-IDF features, train a model, or change the source topic labels.

## Run locally (from repository root)

Install the package: python -m pip install -e ".[dev]"

    python scripts/05_build_gold.py --report-name silver_quality_v1_b81fcce589_seed42_buf10000_n50000.json

Set DATA_LAKE_ROOT in .env or pass --data-lake-root with an existing local directory.
This is a **read/write** operation: Silver remains unchanged; Gold is published once,
without overwriting any previous Gold snapshot. No Google Drive credentials are required
by the package itself; mount or sync the intended Drive separately.

## Input validation

- SilverReader.verify() checks the Silver SHA256 and line count first.
- All Silver records must have valid ID, nonempty text, a known domain, accurate
  text_length, source_snapshot, and quality_flags.
- The canonical 25-label order comes from config/domains_v1.yaml. label_id is
  0-based and stable. Unknown labels, duplicate IDs and exact duplicate text fail.
- No rows are silently filtered.

## Split strategy and leakage limitations

Default train/validation/test ratio: 80/10/10 with seed 42. Full-text word 5-grams
are hashed into a small one-permutation MinHash signature to generate **candidate**
near-duplicate pairs. Candidate pairs are checked with exact word-shingle Jaccard
(default threshold >= 0.90); matched records are merged by union-find and assigned
to the same split. A greedy, seeded group allocator aims to approximate the
distribution of the source topic labels, which is **not guaranteed to be exact**.
The report includes actual ratios and per-split label counts.

**Important:** candidate generation is approximate. It can miss near duplicates.
The output report explicitly marks exhaustive=false. This is a Gold V1 baseline
leakage *mitigation*, not proof of zero semantic/near-duplicate leakage. Candidate
explosion causes GoldLeakageError rather than silently skipping possible matches.
For rigorous benchmarks, run an independent full-corpus leakage audit and manually
review candidate clusters/label quality before finalizing the test set.

No tokenizer/model features are stored in Gold. Fit TF-IDF only on the training
split; transformer tokenization belongs to modeling.

## Output layout

    DATA_LAKE_ROOT/
      03_gold/
        gold_train_v1_<source-config-id>.jsonl
        gold_validation_v1_<source-config-id>.jsonl
        gold_test_v1_<source-config-id>.jsonl
        manifests/gold_manifest_v1_<source-config-id>.json
      05_reports/
        gold_quality_v1_<source-config-id>.json

Each Gold record has ID, normalized text, original domain, stable label_id,
text_length, source_snapshot, and quality_flags. The manifest records source
SHA256, Silver quality report, pipeline version, seed/config, label mapping,
split filenames, record counts, and each split's SHA256. Published JSONL content
is deterministic for the same inputs/config. Publication uses temporary staging,
checksum comparison and rollback on errors, with the manifest published last.

The quality report records counts, class proportions, discovered groups,
candidate-pair checks, observed near matches and important limitations.

## Reading the published snapshot

    from vietnamese_topic_classification.data.gold import GoldReader

    reader = GoldReader(root, manifest_name)  # manifest filename only
    print(reader.verify())                    # verifies all three splits
    train_rows = reader.iter_split("train")   # lazy iterator
    validation_rows = reader.iter_split("validation")
    test_rows = reader.iter_split("test")

GoldReader also supports the shared BaseReader.iter_records(), yielding all
train, validation and test rows in that order. To avoid training leakage,
prefer iter_split("train") for model fit.

## Tests

    python -m pytest tests/test_gold.py -q
    python -m pytest -q
    ruff check src tests

Gold V1 is a baseline. Do not claim that the 25 automatically assigned VTSNLP
topic labels have been manually verified, or that crawl noise has been removed.

## Google Colab (existing Silver snapshot)

Open [notebooks/colab_gold_runner.ipynb](../../../../notebooks/colab_gold_runner.ipynb)
using the `feat/gold-pipeline-v1` branch. The notebook clones the public repository,
installs only the base package dependencies, mounts the correct Drive account,
verifies Silver, checks deterministic Gold destination paths, runs the existing
`scripts/05_build_gold.py` CLI, then validates the manifest, all three split
checksums, row counts, label mapping and cross-split IDs.

The notebook refuses to rebuild Gold if files or stale `.uploading` artifacts
already exist. Gold only needs CPU; no Hugging Face redownload or GPU is required.
The full 50K near-duplicate candidate scan may need substantial time/RAM.
