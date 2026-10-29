"""Offline, dependency-light contract checks for the modeling skeleton."""

from pathlib import Path

import pytest

from vietnamese_topic_classification.evaluation import ClassificationMetrics
from vietnamese_topic_classification.evaluation.metrics import calculate_metrics
from vietnamese_topic_classification.evaluation.report import write_evaluation_report
from vietnamese_topic_classification.training import (
    ClassifierConfig,
    ExperimentConfig,
    FeatureConfig,
)
from vietnamese_topic_classification.training.artifacts import save_artifacts
from vietnamese_topic_classification.training.dataset import load_gold_splits
from vietnamese_topic_classification.training.factory import build_pipeline
from vietnamese_topic_classification.training.runner import run_experiment


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="tfidf_logreg_v1",
        gold_manifest_name="gold_manifest_v1_sample.json",
    )


def test_config_defaults_and_separate_parameter_mappings() -> None:
    config = _config()
    assert config.features.kind == "tfidf"
    assert config.classifier.kind == "logistic_regression"
    assert config.selection_metric == "macro_f1"
    assert FeatureConfig().params is not FeatureConfig().params
    assert ClassifierConfig().params is not ClassifierConfig().params


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "has spaces"])
def test_unsafe_experiment_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        ExperimentConfig(name=name, gold_manifest_name="gold.json")


@pytest.mark.parametrize("manifest", ["../gold.json", "folder/gold.json", "bad.txt", ""])
def test_gold_manifest_must_be_a_filename(manifest: str) -> None:
    with pytest.raises(ValueError, match="gold_manifest_name"):
        ExperimentConfig(name="baseline", gold_manifest_name=manifest)


def test_config_seed_is_validated() -> None:
    with pytest.raises(ValueError, match="seed"):
        ExperimentConfig(name="baseline", gold_manifest_name="gold.json", seed=-1)


def test_entrypoints_explicitly_mark_unimplemented(tmp_path: Path) -> None:
    config = _config()
    with pytest.raises(NotImplementedError, match="Skeleton only"):
        load_gold_splits(data_lake_root=tmp_path, manifest_name=config.gold_manifest_name)
    with pytest.raises(NotImplementedError, match="Skeleton only"):
        build_pipeline(config)
    with pytest.raises(NotImplementedError, match="Skeleton only"):
        run_experiment(config, data_lake_root=tmp_path, output_root=tmp_path)
    result = {
        "experiment_name": config.name,
        "gold_manifest_name": config.gold_manifest_name,
        "validation_metrics": {},
        "artifact_dir": str(tmp_path),
    }
    with pytest.raises(NotImplementedError, match="Skeleton only"):
        save_artifacts(object(), result, config, output_root=tmp_path)
    with pytest.raises(NotImplementedError, match="Skeleton only"):
        calculate_metrics([0], [0], label_names=["topic"])
    report: ClassificationMetrics = {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "per_class_f1": {"topic": 1.0},
        "confusion_matrix": [[1]],
    }
    with pytest.raises(NotImplementedError, match="Skeleton only"):
        write_evaluation_report(report, output_path=tmp_path / "metrics.json")
    assert not (tmp_path / "metrics.json").exists()
