# Vietnamese Text Topic Classification

**Current scope:** Vietnamese **single-label, multiclass** text topic classification using the **25 `domain` labels** in `VTSNLP/vietnamese_curated_dataset`.

> The GitHub repository URL retains the older `vietnamese-multimodal-moderation` name. Earlier multimodal/multi-label assets in `docs/` are historical references, **not** the current research objective.

## Research setup

- **Input:** one Vietnamese text document (`text`) with source record ID (`id`).
- **Target:** exactly one of 25 source `domain` categories. Canonical order lives in `src/vietnamese_topic_classification/config/domains_v1.yaml`.
- **Source revision:** `b81fcce58945970117a1b56d50ec81be2628a5c3` (pinned in the YAML config).
- **Labels are automatically assigned source labels**, not independently verified human ground truth. Any downstream evaluation must make that limitation explicit.
- **First experiment:** 50K Bronze sample, followed by Silver validation and fixed Gold train/validation/test splits; compare TF-IDF + Logistic Regression with and without class weighting.
- **Primary evaluation:** Macro-F1 and per-class F1; also report Accuracy and confusion matrix. A manually reviewed test subset is recommended.

## Repository layout

```text
src/vietnamese_topic_classification/
  __main__.py                # Package CLI
  config/domains_v1.yaml     # Task, source revision, 25 labels
  config/settings.py        # DataLakeConfig and DATA_LAKE_ROOT configuration
  data/bronze/              # Bronze ingestor, reader and exceptions
  data/silver/              # Silver V1 validation, normalization and publishing
  training/                 # Reserved
  evaluation/               # Reserved
tests/test_package.py       # Offline packaging and config tests
.github/workflows/ci.yml    # Lint, tests and installed-package smoke test
scripts/                    # Bronze ingestion CLI and related tooling
docs/                       # Current architecture/plan plus legacy references
```

## Local development (VS Code)

Python 3.10+ is required.

```bash
python -m pip install -e ".[dev]"
ruff check src tests
pytest -q
python -m vietnamese_topic_classification --list-domains
vtc --version
```

The `vtc` CLI currently only displays metadata; ingestion runs through `scripts/01_ingest_bronze.py`, while `python -m vietnamese_topic_classification.data.main` validates/reads an existing Bronze snapshot.

## Hybrid workflow: VS Code → GitHub → Colab → Google Drive

1. Develop and commit Python source locally in VS Code. Keep raw datasets and credentials out of Git.
2. Push to this **public** GitHub repository. Colab can clone it over HTTPS without a GitHub token.
3. In Colab, clone and install the package (`python -m pip install -e ".[dev]"`).
4. Mount the **Google account containing the VTSNLP project**; verify the 50K Bronze file before running future pipeline scripts.
5. Copy `.env.example` to `.env` and set `DATA_LAKE_ROOT` to the verified local/mounted Data Lake folder. The `DataLakeConfig` factory loads `.env` from the working directory (explicit OS environment variables take precedence).
6. Store output data and reports on Google Drive, not in GitHub.

The existing Drive folder may still be named `vietnamese-multilabel-topic-classification`. This historical folder name does not change the current single-label task. Avoid renaming it before updating notebook paths.

## Data layers (planned)

- **Bronze:** original downloaded JSONL records.
- **Silver:** minimally normalized, schema-validated, deduplicated text and source metadata. **No padding/tokenization here.**
- **Gold:** reproducible split datasets and label mappings; protect against duplicate/near-duplicate train–test leakage.
- **Training:** model-specific tokenization/padding for Transformers; TF-IDF baseline needs no padding.

The earlier 3K balanced Silver candidate file and multi-label annotation template are legacy exploration artifacts, **not** the intended 50K single-label baseline. Silver V1 is implemented on this feature branch; Gold transformations are not implemented yet.

See [Silver V1 pipeline details](src/vietnamese_topic_classification/data/silver/README.md), [Colab Silver runner](notebooks/colab_silver_runner.ipynb), [architecture](docs/ARCHITECTURE.md) and [implementation plan](docs/IMPLEMENTATION_PLAN.md).
