"""Modeling V2 uses the existing factory/runner on a verified Gold V2 snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gold_v2 import _published_source

from vietnamese_topic_classification.data.gold import GoldReader, GoldV2Pipeline
from vietnamese_topic_classification.training import (
    ClassifierConfig,
    ExperimentConfig,
    FeatureConfig,
)
from vietnamese_topic_classification.training.runner import run_experiment


@pytest.mark.parametrize("algorithm", ["logistic_regression", "linear_svc"])
def test_v2_baseline_trains_without_touching_v1_or_test(tmp_path, algorithm):
    silver = _published_source(tmp_path)
    gold_v2 = GoldV2Pipeline(silver).run()
    manifest_name = Path(gold_v2["paths"]["manifest"]).name
    reader = GoldReader(tmp_path, manifest_name)
    assert reader.version == "v2"
    reader.verify()

    target = tmp_path / "experiments" / "tfidf_baselines_v2"
    config = ExperimentConfig(
        name=f"tfidf_{algorithm}_v2",
        gold_manifest_name=manifest_name,
        seed=42,
        features=FeatureConfig(params={
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_features": 2000,
        }),
        classifier=ClassifierConfig(kind=algorithm, params={"class_weight": None}),
    )
    result = run_experiment(config, data_lake_root=tmp_path, output_root=target)
    path = Path(result["artifact_dir"])
    assert path.parent == target
    assert (path / "model.joblib").stat().st_size > 0
    saved_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "validation_metrics.json").read_text(encoding="utf-8"))
    assert saved_config["gold_manifest_name"] == manifest_name
    assert metadata["lineage"]["gold_source"]["silver_version"] == "v2"
    assert metadata["test_evaluated"] is False
    assert len(metrics["per_class_f1"]) == 25
    assert len(metrics["confusion_matrix"]) == 25
    assert 0 <= metrics["macro_f1"] <= 1
    assert not (tmp_path / "experiments" / "tfidf_baselines_v1").exists()

    with pytest.raises(FileExistsError):
        run_experiment(config, data_lake_root=tmp_path, output_root=target)
