# Team Colab quick start — Vietnamese Topic Classification

## What is automated?

The notebook [`colab_data_lake_end_to_end.ipynb`](../notebooks/colab_data_lake_end_to_end.ipynb)
runs **clone → pip install → Google Drive mount → Bronze verify/ingest → Silver V1
verify/build → Silver V2 dry-run (or opt-in publish) → Gold V1 verify/build**.
The modeling notebook stays separate to avoid surprise training cost.

A GitHub clone only provides the source code. Data are **not** automatically
included in the repository, Google authentication cannot be bypassed, and
the Colab Drive mount lives inside the collaborator's **cloud runtime**, not
their own laptop filesystem.

## One-time setup for each teammate

1. Accept the Drive owner's invitation with the Google account used in Colab.
   Get edit access if you intend to publish new snapshots; read access is enough
   to verify and train from existing ones.
2. Add the shared lake folder to **My Drive** using a shortcut named
   `vietnamese-multilabel-topic-classification`. Or set the environment variable
   `DATA_LAKE_ROOT` to the exact path visible *inside the mounted Drive*.
3. Open [the end-to-end Colab notebook](../notebooks/colab_data_lake_end_to_end.ipynb).
   Once PR #15 is merged, leave `VTC_GIT_REF` unset, to use `main`.
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
  03_gold/                    # Gold V1 splits + manifests
  04_reports/
    silver/v1/                # Silver V1 quality
    silver/v2/                # Silver V2 quality + audit
    gold_quality_*.json       # Gold V1 quality
  experiments/                # Modeling artifacts
```

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
