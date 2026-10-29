# Modeling skeleton (Phase 0)

This branch only introduces the module boundaries requested for text-classification
experiments. **It does not train a model, fit TF-IDF, evaluate predictions, create
artifacts, change Gold, or add scikit-learn as a runtime dependency.**

## Layout

- `training/model.py`: typed split and experiment-result contracts.
- `training/config.py`: validated, immutable top-level experiment config.
- `training/dataset.py`: future verified `GoldReader` adapter.
- `training/factory.py`: future features + classifier construction.
- `training/runner.py`: future training/validation orchestration.
- `training/artifacts.py`: future immutable artifact publication.
- `evaluation/model.py`: typed classification-metrics contract.
- `evaluation/metrics.py`: future pure multiclass metrics.
- `evaluation/report.py`: future report publication.

Each unfinished public entrypoint intentionally raises `NotImplementedError`.
It must not silently claim training, evaluation, or publication succeeded.

## Contract example

```python
from vietnamese_topic_classification.training import ExperimentConfig

config = ExperimentConfig(
    name="tfidf_logreg_v1",
    gold_manifest_name="gold_manifest_v1_8cadceaf8fd0_cf811f9b2c.json",
)
print(config.features.kind, config.classifier.kind)
```

Do not call `run_experiment()` yet; the training engine has not been implemented.
The split adapter must use `GoldReader.verify()`, retain Gold's stable
`label_id`, and preserve train/validation/test boundaries. TF-IDF must fit
on train only, validation selects the experiment, and the test split remains
untouched until a separate final evaluation step.

`factory.py` is the planned extension point. A future
`Provider + BaseEngine` design can be introduced behind it without forcing
that abstraction into the initial sklearn baseline.

## Test

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_modeling_skeleton.py -q
ruff check src tests
```

No Google Drive mount or external ML dependency is needed for skeleton tests.
