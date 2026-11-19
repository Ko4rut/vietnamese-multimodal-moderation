# Team Colab quick start — Vietnamese Topic Classification

## What is automated?

The notebook [`colab_data_lake_end_to_end.ipynb`](../notebooks/v2/colab_data_lake_end_to_end.ipynb)
runs **clone → pip install → Google Drive mount → Bronze verify/ingest → Silver V1
verify/build → Silver V2 dry-run (or opt-in publish) → Gold V1 verify/build**.
The optional V2 notebook deliberately keeps modeling separate. The dedicated V1 end-to-end notebook includes both baselines by design.

A GitHub clone only provides the source code. Data are **not** automatically
included in the repository, Google authentication cannot be bypassed, and
the Colab Drive mount lives inside the collaborator's **cloud runtime**, not
their own laptop filesystem.

For a V1-only full pipeline including modeling, open [the V1 end-to-end notebook](../notebooks/v1/colab_end_to_end_v1.ipynb).

## One-time setup for each teammate

1. Accept the Drive owner's invitation with the Google account used in Colab.
   Get edit access if you intend to publish new snapshots; read access is enough
   to verify and train from existing ones.
2. Add the shared lake folder to **My Drive** using a shortcut named
   `vietnamese-multilabel-topic-classification`. Or set the environment variable
   `DATA_LAKE_ROOT` to the exact path visible *inside the mounted Drive*.
3. Open [the end-to-end Colab notebook](../notebooks/v2/colab_data_lake_end_to_end.ipynb).
   Leave `VTC_GIT_REF` unset to use `main` after merging your changes.
   To preview the unmerged PR, set it in Colab's first cell:

   ```python
   import os
   os.environ["VTC_GIT_REF"] = "refactor/silver-v1-v2-versioned-layout"
   ```

4. Choose **Runtime → Run all**, authorize `drive.mount()` interactively, and
   review logs. Re-running completed stages verifies SHA256/row counts and skips
   rather than overwriting. A partial publication intentionally fails.
5. If you want to publish Silver V2, inspect its dry-run and change
   `PUBLISH_SILVER_V2 = True`. Only select `GOLD_SOURCE_VERSION = "v2"`
   after that V2 snapshot has been verified.

The normal directory structure is:

```text
DATA_LAKE_ROOT/
  00_source_metadata/         # Bronze source manifests
  01_bronze/                  # Bronze JSONL
  02_silver/
    v1/                       # Accepted/rejected Silver V1
    v2/                       # Accepted/rejected Silver V2
    v3/                       # Reviewed label-refined Silver V3
  03_gold/
    v2/                       # Gold V2 splits + manifests
    v3/                       # Gold V3 splits + manifests
  04_reports/
    silver/v1/                # Silver V1 quality
    silver/v2/                # Silver V2 quality + audit
    silver/v3/                # Keyword analysis, review decisions, Silver V3 audit
    gold/v2/                  # Gold V2 quality
    gold/v3/                  # Gold V3 quality
    gold_quality_*.json       # Gold V1 quality
  experiments/                # Modeling artifacts
```

## V3 — review in Silver, split in Gold, model separately

After a complete Silver V2 snapshot has been published, use the dedicated
[Silver V3 runner](../notebooks/v3/colab_silver_v3_runner.ipynb),
[Gold V3 runner](../notebooks/v3/colab_gold_v3_runner.ipynb), and
[Modeling V3 runner](../notebooks/v3/colab_modeling_v3_runner.ipynb). The
[V3 end-to-end notebook](../notebooks/v3/colab_end_to_end_v3.ipynb) sequences the same
stages in one notebook and includes an explicit human-review stop before publication.

1. Mount the same Drive with read access to inspect; use edit access to publish.
2. Silver: select the exact Silver V2 quality report. The runner verifies its report,
   accepted/rejected JSONL, audit, and Silver V1 lineage. Generate keyword/overlap
   evidence and review the deterministic up-to-300-row News pilot before freezing a
   non-overlapping label guide.
3. Create the full News decision file only after the pilot. Approve every row as a
   human reviewer, then opt in to publish Silver V3.
4. Gold: select the verified Silver V3 snapshot and opt in to create new
   train/validation/test splits from all accepted Silver V3 rows using the existing
   splitter. Gold V2 split membership is not reused.
5. Modeling: select the exact Gold V3 manifest. The baselines train on train and are
   compared on validation; test is checksum-verified but not loaded or evaluated.

The candidate News sublabels are provisional until you examine the actual corpus.
V3 is write-once and leaves all V1/V2 files unchanged. Since taxonomy/relabel decisions
are made before Gold V3 creates its split, this workflow uses the full source pool to
develop the taxonomy; document that limitation when reporting final test metrics.
The modeling runner writes to `experiments/tfidf_baselines_v3/` and uses the label
mapping embedded in the verified Gold V3 manifest.
See [V3 workflow details](V3_NEWS_RELABELING.md).

## Troubleshooting

- **Missing Data Lake:** Check Drive account access, shortcut name, or
  `DATA_LAKE_ROOT`. A shared-folder link is not the same thing as a local
  mounted filesystem path.
- **Cannot publish:** Confirm Drive write permission, remaining space, and
  that the target snapshot hasn't already been published. Ask the lake owner
  to publish once, then use the existing snapshot rather than writing duplicates.
- **Old imports in Colab:** Restart session and run from the first cell.
  The end-to-end notebook clears cached project imports after checkout.
- **Unexpected snapshot:** Stop. Verify source revision, manifest SHA256,
  sample size, and report version before continuing.
- **Concurrent users:** Do not trigger simultaneous writes of the same snapshot;
  immutable names and temporary `.uploading` files are a safeguard but not
  a distributed locking service.

## Verification scope

The repository CI runs Ruff, pytest, CLI smoke checks, notebook AST/structure
checks, and miniature synthetic snapshot tests. It **cannot automatically**
execute Google OAuth, mount another collaborator's Drive, verify their permissions,
download the full external dataset in Colab, or guarantee quota/network stability.
One human-in-the-loop Colab run from each distinct Drive setup is still needed
for actual cloud integration confidence. No manual *dataset labeling* is required.
