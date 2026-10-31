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
  data/silver/v1/           # Silver V1 schema validation and normalization
  data/silver/v2/           # Silver V2 corpus-aware cleaning and audit
  data/silver/reader.py     # Version-aware Silver snapshot verification
  data/colab_lake.py        # Teammate-specific Drive mount resolution
  data/gold/                # Gold V1 train/validation/test and manifests
  training/                 # TF-IDF Logistic Regression and LinearSVC baselines
  evaluation/               # Metrics and diagnostics
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

The earlier 3K balanced Silver candidate file and multi-label annotation template are legacy exploration artifacts, **not** the intended 50K single-label baseline. Silver V1, Silver V2, Gold V1 and TF-IDF modeling are implemented; Silver snapshots on Drive are segregated by version.

See [Silver V1 pipeline details](src/vietnamese_topic_classification/data/silver/README.md), [Colab Silver runner](notebooks/v1/colab_silver_runner.ipynb), [architecture](docs/ARCHITECTURE.md) and [implementation plan](docs/IMPLEMENTATION_PLAN.md).


## Team onboarding: run Colab without rewriting the lake

For colleagues, start with **[End-to-end Colab runner](notebooks/v2/colab_data_lake_end_to_end.ipynb)** and **[Team setup guide](docs/COLAB_TEAM_SETUP.md)**. After the refactor PR is merged, the notebook fetches `main`, installs the project, mounts each user's Drive, verifies existing immutable data, processes missing Bronze/Silver V1/Gold snapshots, and runs Silver V2 as a dry-run. Publishing Silver V2 is explicitly opt-in.

**Required:** The data owner must share the Google Drive Data Lake folder with each collaborator; each user must authorize Colab's Drive mount. A GitHub clone cannot download private Drive data or mount the cloud folder onto a teammate's physical computer. Add a shortcut named `vietnamese-multilabel-topic-classification` to My Drive, or provide the real Colab-mounted `DATA_LAKE_ROOT`.

Silver V1: `02_silver/v1/` and `04_reports/silver/v1/`. Silver V2: `02_silver/v2/` and `04_reports/silver/v2/`. Bronze and Gold retain their existing locations. No duplicate root-level Silver implementation modules remain.

## Versioned Colab notebooks

- [`notebooks/v1/colab_end_to_end_v1.ipynb`](notebooks/v1/colab_end_to_end_v1.ipynb): one run from pinned Bronze ingestion through Silver V1, Gold V1, and both TF-IDF baselines (no held-out test evaluation).
- [`notebooks/v1/`](notebooks/v1/): separate Bronze, Silver V1, Gold V1, modeling and Silver V1 EDA notebooks.
- [`notebooks/v2/`](notebooks/v2/): Silver V2 runner and the earlier optional V2 Data Lake workflow. Gold V2 is supported by its dedicated [Colab runner](notebooks/v2/colab_gold_v2_runner.ipynb).

Moving notebooks does **not** move or rename the Google Drive data snapshots; V1 experiments still use `experiments/tfidf_baselines_v1/`.

## Gold V2

The separate Gold V2 pipeline reads a **complete, published Silver V2 snapshot** and writes immutable Gold V2 splits to `03_gold/v2/`, with reports under `04_reports/gold/v2/`. It checks the Silver V2 accepted/rejected/audit files and SHA256 lineage to Silver V1. No V1 snapshots are overwritten. See [Gold V2 guide](docs/GOLD_V2.md) and [Gold V2 Colab runner](notebooks/v2/colab_gold_v2_runner.ipynb).

## Modeling V2 — retrain, never reuse V1 model

Use [Modeling V2 on Colab](notebooks/v2/colab_modeling_runner.ipynb) after Gold V2 has been published and verified. The notebook reuses the existing TF-IDF + Logistic Regression/LinearSVC factory and CLI, but fits new parameters from `03_gold/v2/manifests/` and publishes isolated artifacts in `experiments/tfidf_baselines_v2/`. Validation Macro-F1 is used for selection; test is held out. See [the Modeling guide](docs/COLAB_MODELING.md). A V1/V2 score delta alone does not isolate the cleaning effect because the Gold splits may differ.
