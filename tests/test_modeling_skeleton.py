"""Configuration contracts for the implemented modeling baseline."""

import pytest

from vietnamese_topic_classification.training import (
    ClassifierConfig,
    ExperimentConfig,
    FeatureConfig,
)


def _config():
    return ExperimentConfig(name="tfidf_logreg_v1", gold_manifest_name="gold_manifest_v1_sample.json")


def test_defaults():
    config = _config()
    assert config.features.kind == "tfidf"
    assert config.classifier.kind == "logistic_regression"
    assert config.selection_metric == "macro_f1"
    assert FeatureConfig().params is not FeatureConfig().params
    assert ClassifierConfig().params is not ClassifierConfig().params


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "has spaces"])
def test_unsafe_experiment_names(name):
    with pytest.raises(ValueError, match="name"):
        ExperimentConfig(name=name, gold_manifest_name="gold.json")


@pytest.mark.parametrize("manifest", ["../gold.json", "folder/gold.json", "bad.txt", ""])
def test_bad_manifest(manifest):
    with pytest.raises(ValueError, match="gold_manifest_name"):
        ExperimentConfig(name="baseline", gold_manifest_name=manifest)


def test_negative_seed():
    with pytest.raises(ValueError, match="seed"):
        ExperimentConfig(name="baseline", gold_manifest_name="gold.json", seed=-1)
