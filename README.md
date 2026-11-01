# Vietnamese Text Topic Classification

A reproducible Python research pipeline for **single-label, multiclass Vietnamese text topic classification** over the **25 `domain` categories** in `VTSNLP/vietnamese_curated_dataset`.

> **Scope clarification:** The repository URL is still `vietnamese-multimodal-moderation` for historical reasons. This implementation focuses on **text topic classification**, not multimodal content moderation or multi-label prediction. Older proposal/design files may describe a different objective.

## Contents

- [Research scope and limitations](#research-scope-and-limitations)
- [System architecture](#system-architecture)
- [Repository structure](#repository-structure)
- [Quick start (local)](#quick-start-local)
- [Google Drive Data Lake](#google-drive-data-lake)
- [Run pipelines on Google Colab](#run-pipelines-on-google-colab)
- [Run individual stages from the CLI](#run-individual-stages-from-the-cli)
- [Modeling and evaluation](#modeling-and-evaluation)
- [Development and CI](#development-and-ci)
- [Troubleshooting](#troubleshooting)
- [Documentation and project status](#documentation-and-project-status)

## Research scope and limitations

| Item | Current implementation |
| --- | --- |
| Input | One Vietnamese document with source `id` and `text` |
| Target | Exactly one of 25 source `domain` categories |
| Canonical labels | [`config/domains_v1.yaml`](src/vietnamese_topic_classification/config/domains_v1.yaml), with stable 0-based `label_id` |
| Source revision | `b81fcce58945970117a1b56d50ec81be2628a5c3`, pinned in config |
| Initial corpus | Reproducible 50,000-record Bronze sample |
| Default Gold split | 80% train / 10% validation / 10% test, seed 42 |
| Available baseline models | TF-IDF + Logistic Regression; TF-IDF + LinearSVC; optional class-weighted Logistic Regression |
| Model selection | Validation Macro-F1 (also inspect per-class F1 and accuracy) |
| Final evaluation | Separate, explicit held-out test evaluation; confusion matrix and diagnostics are available |

**Label caveat:** The 25 domains are source-assigned labels, **not independently validated human ground truth**. Report metrics with this limitation and manually audit samples before making strong claims about real-world performance. Gold performs approximate near-duplicate leakage mitigation; it does **not** prove that every near-duplicate or semantic overlap has been eliminated.

## System architecture

The pipeline uses **VS Code/local development → GitHub for versioned code → Google Colab for execution → Google Drive for persistent data and experiment artifacts**. The Python package can also run locally when the Data Lake is accessible on that machine.

```text
Pinned VTSNLP source
       |
       v
  Bronze (raw, immutable JSONL + source manifest)
       |
       v
  Silver V1 (schema validation, NFC normalization, exact deduplication)
       |                                  |
       |                                  v
       |                           Silver V2 (audited,
       |                           corpus-aware cleaning)
       v                                  |
  Gold V1 (train/val/test)                v
       |                           Gold V2 (train/val/test)
       |                                  |
       v                                  v
  Modeling V1                         Modeling V2
       |                                  |
       +-----------+----------------------+
                   |
          Shared training factory
          TF-IDF + LR / LinearSVC
                   |
            Validation metrics
            + diagnostics
                   |
       Selected model -> held-out test
```

**Important boundaries:**

- **Bronze** preserves sampled source records and provenance; it does not clean labels or text.
- **Silver V1** validates, minimally normalizes and deduplicates. **Silver V2** reads a published Silver V1 snapshot and adds audited corpus-aware cleaning; its publication is opt-in.
- **Gold V1/V2** produces immutable, versioned train/validation/test splits, label mapping, manifests and quality reports. It does **not** tokenize or create TF-IDF features. Gold V2 requires a complete published Silver V2 snapshot.
- **Training** fits TF-IDF on training data and fits separate model parameters for each experiment. Modeling V2 **reuses the factory/code but never reuses V1's fitted model**.
- **Evaluation** uses validation for comparison; access to the held-out test set is a separate explicit action. Keep track of manifest checksums, source lineage and experiment configuration.
- Versioned data snapshots and experiment artifacts are never meant to overwrite one another. V1/V2 validation deltas **do not isolate the effect of cleaning**, because Gold split membership may differ.

## Repository structure

```text
.
├── .github/workflows/ci.yml            # Ruff, pytest, installed-package smoke check
├── .env.example                        # DATA_LAKE_ROOT template (no credentials)
├── apps/api/                            # Future API placeholder, not a working service
├── apps/web/                            # Future web UI placeholder
├── docs/                                # Guides, experiment notes, historical documents
├── notebooks/
│   ├── v1/                              # Bronze -> Silver V1 -> Gold V1 -> Modeling V1
│   └── v2/                              # Silver V2 -> Gold V2 -> Modeling V2
├── scripts/
│   ├── 01_ingest_bronze.py
│   ├── 02_build_silver.py
│   ├── 03_read_silver.py
│   ├── 04_generate_silver_eda_notebook.py
│   ├── 05_build_gold.py
│   ├── 06_train_baseline.py
│   ├── 07_evaluate_model.py
│   ├── 08_plot_diagnostics.py
│   ├── 09_build_silver_v2.py
│   └── 10_build_gold_v2.py
├── src/vietnamese_topic_classification/
│   ├── __main__.py                      # vtc metadata/list-domains CLI
│   ├── config/                          # Domain list and DataLakeConfig
│   ├── data/
│   │   ├── bronze/                      # Ingestor, Reader, contracts
│   │   ├── silver/
│   │   │   ├── v1/                      # Validation and normalization
│   │   │   ├── v2/                      # Cleaning, corpus audit, publication
│   │   │   └── reader.py               # Version-aware snapshot reader
│   │   ├── gold/
│   │   │   ├── v2/                      # V2 pipeline and lineage verification
│   │   │   ├── pipeline.py              # V1 Gold implementation
│   │   │   ├── splitter.py              # Deterministic, group-aware split
│   │   │   └── reader.py                # Verify/read Gold V1 or V2
│   │   └── colab_lake.py               # Mounted Drive path resolution
│   ├── training/                        # Config, dataset, factory, runner, artifacts
│   └── evaluation/                      # Metrics, reports, diagnostics
├── tests/                               # Offline tests and synthetic snapshots
└── pyproject.toml                       # Package and optional dependency groups
```

The `data/` and `experiments/` folders in Git are placeholders, **not** copies of the real Google Drive lake.

## Quick start (local)

**Prerequisites:** Python 3.10+, Git, and a virtual environment. VS Code is optional. A local smoke test needs **no dataset, Colab session, Google token, or GPU**.

### 1. Clone and create the environment

```bash
git clone https://github.com/Ko4rut/vietnamese-multimodal-moderation.git
cd vietnamese-multimodal-moderation

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell (instead of the line above):
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev,modeling,data]"
```

Dependency groups in `pyproject.toml`:

| Extra | Purpose |
| --- | --- |
| `dev` | `pytest` and `ruff` |
| `data` | Hugging Face `datasets` for Bronze ingestion |
| `modeling` | `scikit-learn`, `joblib`, `matplotlib` for training/evaluation |
| `eda` | Notebook, Pandas/NumPy and plotting tools for Silver EDA |

Use `pip install -e ".[dev,eda,modeling,data]"` if you need everything, including local EDA notebooks.

### 2. Verify installation

```bash
ruff check src tests
pytest -q
python -m vietnamese_topic_classification --list-domains
vtc --version
```

These commands verify packaging, lint and offline tests. **They do not train a model.** The `vtc` entrypoint currently prints metadata/domains, not a pipeline runner; stage entrypoints live under `scripts/`.

### 3. Point the package at your existing lake (only for data/modeling)

Copy `.env.example` to `.env` and set `DATA_LAKE_ROOT` to a **directory accessible to the runtime**, for example:

```dotenv
# Example only: use the path of your own mounted/synced Drive folder
DATA_LAKE_ROOT=/content/drive/MyDrive/vietnamese-multilabel-topic-classification
```

The path must exist; in Colab, mount Google Drive **before** reading it. The configuration loader reads `.env` from the current working directory, while an explicitly set OS environment variable takes precedence. Never commit `.env`, tokens, raw datasets, or experiment models.

## Google Drive Data Lake

GitHub contains **code/config/docs**, not the complete dataset. A collaborator needs access granted by the Data Lake owner and must mount Drive in **their own Colab runtime** (or mount/sync the folder locally). A GitHub clone alone cannot access private Google Drive files.

Representative published layout:

```text
DATA_LAKE_ROOT/
├── 00_source_metadata/                 # Bronze manifests
├── 01_bronze/                          # Raw source JSONL
├── 02_silver/
│   ├── v1/                             # Silver V1 accepted/rejected
│   └── v2/                             # Silver V2 accepted/rejected
├── 03_gold/
│   ├── manifests/                      # Gold V1 manifests
│   ├── gold_*_v1_*.jsonl                # Gold V1 splits
│   └── v2/
│       ├── manifests/                  # Gold V2 manifests
│       └── gold_*_v2_*.jsonl           # Gold V2 splits
├── 04_reports/
│   ├── silver/v1/                       # Silver V1 quality reports
│   ├── silver/v2/                       # Silver V2 reports/audits
│   ├── gold/v2/                         # Gold V2 quality reports
│   └── gold_quality_v1_*.json          # Gold V1 reports
└── experiments/
    ├── tfidf_baselines_v1/             # Trained V1 experiments
    └── tfidf_baselines_v2/             # Freshly trained V2 experiments
```

Some older lake snapshots use flat legacy paths. The readers support version-aware resolution; do **not** manually move or rename shared snapshots just to make them look like the tree above. The shared folder may still be named `vietnamese-multilabel-topic-classification`; the name does not change the current task.

**Sharing:** Ask the owner to share the lake folder with your Google account. Where appropriate, add a shortcut to **My Drive** with the expected folder name or set the notebook's mount-path override. Read access can suffice for verification/training from existing data; publishing snapshots requires write access. See [team setup](docs/COLAB_TEAM_SETUP.md).

## Run pipelines on Google Colab

**Recommended for teammates:** Open the notebook from GitHub or upload it to Colab, run its cells from top to bottom, authorize `drive.mount()`, and confirm that `DATA_LAKE_ROOT` resolves to the correct folder. Notebook setup clones the chosen Git branch (`VTC_GIT_REF` defaults to `main`) and installs the required package. CPU is sufficient for the current TF-IDF baselines; a GPU is not required.

### V1 — one notebook, Bronze through baseline training

Start with **[V1 end-to-end Colab runner](notebooks/v1/colab_end_to_end_v1.ipynb)**. It verifies/reuses or produces Bronze, Silver V1 and Gold V1, then trains TF-IDF + Logistic Regression and TF-IDF + LinearSVC sequentially. Optionally enable balanced Logistic Regression and diagnostics. **Held-out test evaluation is intentionally separate.**

For individual stages use [Bronze](notebooks/v1/colab_runner.ipynb), [Silver V1](notebooks/v1/colab_silver_runner.ipynb), [Gold V1](notebooks/v1/colab_gold_runner.ipynb), [Modeling V1](notebooks/v1/colab_modeling_runner.ipynb), or [Silver V1 EDA](notebooks/v1/eda/01_silver_eda.ipynb).

### V2 — separate, explicit publication steps

Run the following in order, against the **same verified lake**:

1. **[Silver V2 runner](notebooks/v2/colab_silver_v2_runner.ipynb):** read/verify Silver V1, inspect cleaning results and audit, then explicitly publish a complete Silver V2 snapshot if approved.
2. **[Gold V2 runner](notebooks/v2/colab_gold_v2_runner.ipynb):** choose the published Silver V2 report, verify all source files/checksums and set `PUBLISH_GOLD_V2=True` to publish isolated V2 splits.
3. **[Modeling V2 runner](notebooks/v2/colab_modeling_runner.ipynb):** choose the published Gold V2 manifest and fit **new** TF-IDF + Logistic Regression/LinearSVC models, saving them under `experiments/tfidf_baselines_v2/`.

The [V2 Data Lake end-to-end notebook](notebooks/v2/colab_data_lake_end_to_end.ipynb) is an **optional preparation workflow**, **not** an end-to-end Modeling V2 runner. It can verify/build earlier stages and perform a Silver V2 dry-run; use the dedicated Gold V2 and Modeling V2 notebooks for the full V2 path.

For consistency, use an explicit `VTC_GIT_REF` when reviewing a feature branch; on `main` leave it unset. Publishing an immutable snapshot is not the same as a dry-run. Do not launch concurrent writes to the same target snapshot.

## Run individual stages from the CLI

With your environment activated and `DATA_LAKE_ROOT` pointing to an **existing** mounted/synced lake, the script entrypoints can be run from the repository root. Replace example filenames with **actual manifest/report names from your lake**.

```bash
# 1. Only if the pinned Bronze source has not already been published:
python scripts/01_ingest_bronze.py --output-root "/path/to/lake" --sample-size 50000 --seed 42

# 2. Bronze -> Silver V1 (requires an existing source manifest):
python scripts/02_build_silver.py --manifest-name "source_manifest_YOUR_SNAPSHOT.json"

# 3. Silver V1 -> Gold V1:
python scripts/05_build_gold.py --report-name "silver_quality_v1_YOUR_SNAPSHOT.json"

# 4. Silver V1 -> Silver V2: inspect first, then publish intentionally:
python scripts/09_build_silver_v2.py --report-name "silver_quality_v1_YOUR_SNAPSHOT.json" --dry-run
# To publish, repeat without --dry-run after reviewing the output.

# 5. Published Silver V2 -> Gold V2:
python scripts/10_build_gold_v2.py --data-lake-root "/path/to/lake" \
  --report-name "silver_quality_v2_YOUR_SNAPSHOT.json"
```

All steps verify their sources and use versioned outputs. Some already-published targets are reusable after verification; partial or conflicting publications deliberately raise errors. **Do not delete shared outputs simply to bypass these checks.** The Bronze ingestion may require internet access and the optional `data` dependencies. See [scripts guide](scripts/README.md), [Silver documentation](src/vietnamese_topic_classification/data/silver/README.md), [Gold V1 documentation](src/vietnamese_topic_classification/data/gold/README.md) and [Gold V2 documentation](docs/GOLD_V2.md).

## Modeling and evaluation

The shared factory `src/vietnamese_topic_classification/training/factory.py` creates an **unfitted** scikit-learn pipeline: `TfidfVectorizer` (default word 1–2-grams, `min_df=2`, experiment-configured maximum features) followed by `LogisticRegression` or `LinearSVC`. Logistic Regression supports multiclass probability predictions; LinearSVC provides decision scores, not probabilities.

To train **one** experiment from an existing Gold V2 manifest:

```bash
python -m pip install -e ".[modeling]"
python scripts/06_train_baseline.py \
  --data-lake-root "/path/to/lake" \
  --manifest-name "gold_manifest_v2_YOUR_SNAPSHOT.json" \
  --output-root "/path/to/lake/experiments/tfidf_baselines_v2" \
  --name "tfidf_logreg_v2_trial" \
  --classifier logistic_regression \
  --class-weight none --max-features 100000 --min-df 2 --seed 42
```

For LinearSVC, use `--classifier linear_svc` and a **different** experiment name. Use the V1 manifest and `tfidf_baselines_v1` output directory for V1. A completed experiment is linked to its Gold manifest/configuration and cannot silently be overwritten by a mismatched run.

- **Selection:** Compare validation Macro-F1, per-class F1 and accuracy; optionally generate diagnostics with `scripts/08_plot_diagnostics.py` or the notebook option `RUN_DIAGNOSTICS=True` (can require costly refits).
- **Final test:** After choosing the model on validation, run `python scripts/07_evaluate_model.py --artifact-dir "/path/to/lake/experiments/tfidf_baselines_v2/tfidf_logreg_v2_trial" --data-lake-root "/path/to/lake"`. This writes a **single, write-once** `test_metrics.json` and refuses to overwrite an existing report.
- **Interpretation:** Do not use the test set repeatedly for tuning. V1/V2 scores are not a controlled cleaning comparison without a common fixed evaluation dataset. Inspect domain balance and suspected pseudo-label errors.

Detailed usage and notes: [Colab modeling](docs/COLAB_MODELING.md) and [model diagnostics](docs/MODELING_DIAGNOSTICS.md).

## Development and CI

```bash
python -m pip install -e ".[dev,modeling]"
ruff check src tests
pytest -q
python -m vietnamese_topic_classification --list-domains
```

[GitHub Actions CI](.github/workflows/ci.yml) runs on pushes and pull requests targeting `main`. It installs the package, checks Ruff, runs offline pytest tests and performs an installed-package CLI smoke check. Synthetic tests verify snapshot readers and pipelines; **CI does not mount collaborators' Drive, perform Google OAuth or validate access to a real cloud lake**. Run the corresponding Colab notebook manually to validate Drive integration.

Keep bulk data, credentials and generated models outside Git. `.gitignore` excludes `.env`, generated lake data, experiment files, checkpoints and common model artifacts.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `DATA_LAKE_ROOT environment variable is required` | Create `.env` in the working directory or export the variable; confirm the folder exists |
| Data Lake/manifest/report missing | Correct Google account, Drive sharing permissions, My Drive shortcut/mount path and exact versioned filename |
| `ModuleNotFoundError` in Colab | Rerun checkout/install cells, restart runtime if necessary, and re-import from the installed branch |
| Missing `sklearn`/`joblib` | Install `.[modeling]`; `.[dev]` alone does not include ML dependencies |
| Publication conflict or `.uploading` leftovers | Check source checksums, complete published files and write permissions; do not overwrite/mix V1/V2 snapshots |
| Slow diagnostics or large Gold grouping | Use CPU with sufficient RAM; turn optional `RUN_DIAGNOSTICS` off, and inspect logs rather than claiming a frozen session succeeded |
| Score changes between V1 and V2 | Confirm manifest, label mapping, Gold split membership, seed and train/validation protocol before interpreting the delta |

## Documentation and project status

- [Team Colab setup](docs/COLAB_TEAM_SETUP.md)
- [Silver V2 cleaning](docs/SILVER_V2.md)
- [Gold V2 and lineage checks](docs/GOLD_V2.md)
- [Modeling V1/V2](docs/COLAB_MODELING.md)
- [Notebook index](notebooks/README.md)
- [Layer data models](src/vietnamese_topic_classification/data/LAYER_MODELS.md)

**Implemented now:** Bronze/Silver V1/Silver V2/Gold V1/Gold V2 pipelines; immutable readers/manifests; TF-IDF + Logistic Regression/LinearSVC baselines; evaluation, diagnostics, Colab notebooks and offline CI tests.

**Not implemented as a production application:** a deployed inference API, web UI, image/video moderation, or Transformer/PhoBERT training. `apps/api` and `apps/web` are placeholders. Some older files such as [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) contain planned-stage wording from an earlier phase and should not override the current code/this README.

**License and data rights:** The repository is public but currently has no top-level `LICENSE` file. Do not assume source-code redistribution rights or dataset redistribution rights without checking and adding the appropriate license/permissions. Historical 3K/multi-label annotation material is not part of the current 50K single-label baseline.
