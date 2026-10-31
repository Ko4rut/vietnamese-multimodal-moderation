"""Train TF-IDF baselines on verified Gold V1/V2 (never evaluate held-out test)."""

from __future__ import annotations

from pathlib import Path

from vietnamese_topic_classification.config import get_domain_labels
from vietnamese_topic_classification.evaluation.metrics import calculate_metrics

from .artifacts import save_artifacts
from .config import ExperimentConfig
from .dataset import load_gold_training_splits
from .factory import build_pipeline
from .model import ExperimentResult


def run_experiment(
    config: ExperimentConfig,
    *,
    data_lake_root: Path,
    output_root: Path,
) -> ExperimentResult:
    """Train on Gold train, score validation, and publish a write-once artifact."""
    if config.selection_metric != "macro_f1":
        raise ValueError("TF-IDF baseline modeling only supports macro_f1 selection")
    train, validation = load_gold_training_splits(
        data_lake_root=data_lake_root,
        manifest_name=config.gold_manifest_name,
    )
    pipeline = build_pipeline(config)
    pipeline.fit(train["texts"], train["labels"])
    predictions = pipeline.predict(validation["texts"])
    report = calculate_metrics(
        validation["labels"], predictions, label_names=get_domain_labels()
    )
    result: ExperimentResult = {
        "experiment_name": config.name,
        "gold_manifest_name": config.gold_manifest_name,
        "validation_metrics": {
            key: report[key] for key in ("accuracy", "macro_f1", "weighted_f1")
        },
        "artifact_dir": "",
        "validation_report": report,
    }
    path = save_artifacts(
        pipeline, result, config, output_root=output_root, data_lake_root=data_lake_root
    )
    result["artifact_dir"] = str(path)
    return result
