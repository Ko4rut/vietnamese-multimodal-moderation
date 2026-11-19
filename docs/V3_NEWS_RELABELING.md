# V3 — Silver V2 to reviewed Silver V3 and Gold V3

## Pipeline decision

V3 starts from a verified **Silver V2** snapshot, treating it as the V3 Bronze-equivalent
input. Label discovery and approved News relabeling happen in Silver V3. Gold V3 then
creates new train/validation/test splits from the complete Silver V3 pool using the
existing deterministic, near-duplicate-aware splitter. V1/V2 snapshots remain immutable.

V3 does not clean or otherwise change article text. Silver V3 changes only approved
labels and records the Silver V2 lineage. If text-cleaning rules are later changed,
that should be specified and audited separately.

## Taxonomy policy and News subtype review

The Colab runner writes a keyword report with class-specific document-frequency terms,
News-vs-other-class keyword overlap, class counts, and a small set of article excerpts.
These are **discovery signals**, not automatic labels. Keywords cannot reliably tell
whether an article's main topic is politics, sport, a crime, or general current affairs.
Review full articles and use a written label guide before selecting or changing the
candidate taxonomy in `src/vietnamese_topic_classification/config/domains_v3.yaml`.

The selected single-label policy prioritizes existing specialist labels when the main
topic fits them. Use `Sports` for sports articles rather than a parallel `News_Sports`;
use `Finance`, `Health`, `Business_and_Industrial`, and other existing labels where
appropriate. The three provisional News sublabels are `News_Politics`,
`News_Crime_Disaster`, and `News_Current_Affairs`. Apply `Law_and_Government` to legal,
administrative, policy, and public-service content; reserve `News_Politics` for
political developments such as elections, parties, and political disputes. `News_Current_Affairs`
is a residual category, not a default for uncertain records. Rebuild the review template
after taxonomy changes.

## Colab workflow

The V3 notebooks follow the V1/V2 stage layout:
[Silver](../notebooks/v3/colab_silver_v3_runner.ipynb),
[Gold](../notebooks/v3/colab_gold_v3_runner.ipynb),
[Modeling](../notebooks/v3/colab_modeling_v3_runner.ipynb), and the combined
[end-to-end notebook](../notebooks/v3/colab_end_to_end_v3.ipynb). Mount the shared
Drive with write access only when publishing:

1. Select the exact published Silver V2 quality report. The runner verifies its
   checksums and Silver V1 lineage before reading it.
2. Run keyword analysis. The report is saved under
   `04_reports/silver/v3/news_keyword_analysis_v3_*.json`.
3. Review the deterministic **up-to-300-row News pilot** in
   `04_reports/silver/v3/news_pilot_v3_*.jsonl`. Use it to inspect actual article
   types, resolve boundaries, and finalize a non-overlapping label guide and taxonomy.
   The pilot file cannot be used to publish Silver V3.
4. Optionally set Colab to use a GPU, enable `INSTALL_LOCAL_MODEL_DEPS=True`, then
   set `RUN_LOCAL_MODEL_LABELING=True`. The runner downloads the public
   `Qwen/Qwen2.5-3B-Instruct` weights and runs inference locally in the Colab runtime;
   it does not call an LLM API or require an API key. Model weights require internet
   access. Check the model license before use. The output is checkpointed and can be
   resumed with `RESUME_LOCAL_MODEL_LABELING=True` only when it is still an
   unreviewed candidate file from the same model and taxonomy.
5. Local LLM output is only a **candidate**. Every row remains
   `approved=false` and `needs_review=true`; long inputs that exceed configured text
   or token limits are explicitly marked as truncated. Review the complete article,
   correct the label if needed, enter reviewer/reason, then set `approved=true` and
   `needs_review=false`. The candidate file includes article text; keep it in your
   authorized Drive and do not commit/share it publicly.
6. If not using the local model, enable `CREATE_FULL_REVIEW_TEMPLATE=True` to create
   a blank file for manual/LLM-assisted review and complete the same approval fields.
7. Set `PUBLISH_SILVER_V3=True` only after review. Silver V3 validates that all News
   IDs have exactly one approved decision, writes immutable JSONL plus audit/report,
   and verifies the published files.
8. In the Gold runner, set `PUBLISH_GOLD_V3=True` to split Silver V3 into new
   train/validation/test sets. The manifest records V3 taxonomy, split configuration,
   checksums, and lineage.
9. In the Modeling runner, select the exact Gold V3 manifest. The TF-IDF Logistic
   Regression and LinearSVC baselines train on train data and are compared on
   validation Macro-F1; Gold test remains unevaluated.

The end-to-end notebook defaults its publish and modeling flags to `False`. The
notebooks refuse partial or stale publications and never overwrite V1/V2. Gold test
data must remain held out from model training and model selection. Because taxonomy exploration/relabeling occurs
before the Gold split, the source pool (including records later assigned to test)
influences taxonomy development; document that limitation in reported results.
The source manifest currently marks the dataset license as `unverified`; keep source
data and article-bearing review output out of Git until usage rights are confirmed.

## CLI equivalents

```bash
# Inspect Silver V2 labels/keywords; this does not change data.
python scripts/13_build_silver_v3.py analyze \
  --data-lake-root "/path/to/lake" \
  --silver-v2-report "silver_quality_v2_YOUR_SNAPSHOT.json"

# Create a deterministic pilot for taxonomy and guideline review.
python scripts/13_build_silver_v3.py pilot \
  --data-lake-root "/path/to/lake" \
  --silver-v2-report "silver_quality_v2_YOUR_SNAPSHOT.json" \
  --output "/path/to/lake/04_reports/silver/v3/news_pilot.jsonl" \
  --sample-size 300 --seed 42

# After freezing taxonomy, prepare the full editable News review input.
# The output must not already exist.
python scripts/13_build_silver_v3.py prepare \
  --data-lake-root "/path/to/lake" \
  --silver-v2-report "silver_quality_v2_YOUR_SNAPSHOT.json" \
  --output "/path/to/lake/04_reports/silver/v3/news_review.jsonl"

# Optional: generate local-model suggestions (CUDA GPU required; no API key).
# The output remains unapproved and includes article text for review.
python scripts/15_label_news_v3_local.py \
  --data-lake-root "/path/to/lake" \
  --silver-v2-report "silver_quality_v2_YOUR_SNAPSHOT.json" \
  --output "/path/to/lake/04_reports/silver/v3/news_review_decisions_v3.jsonl"
# Resume an interrupted, unreviewed candidate file by adding --resume.

# After review, publish Silver V3.
python scripts/13_build_silver_v3.py build \
  --data-lake-root "/path/to/lake" \
  --silver-v2-report "silver_quality_v2_YOUR_SNAPSHOT.json" \
  --decisions "/path/to/lake/04_reports/silver/v3/news_review.jsonl"

# Split the complete Silver V3 pool.
python scripts/14_build_gold_v3.py \
  --data-lake-root "/path/to/lake" \
  --silver-v3-report "silver_quality_v3_YOUR_SNAPSHOT.json"
```

The modeling equivalent is `scripts/06_train_baseline.py`; set
`--manifest-name` to the exact Gold V3 manifest and use a separate output root such as
`experiments/tfidf_baselines_v3/`. The training runner reads class names from that
verified manifest, so V3's 27-class evaluation is not fixed to V1/V2's 25 classes.

`analyze` and `prepare` write only the requested review artifacts; they do not
publish Silver or Gold. `build` is write-once and requires complete, human-approved
decisions. Use the exact report filenames printed by each successful command.