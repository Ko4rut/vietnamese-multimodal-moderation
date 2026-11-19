from __future__ import annotations

import json
from pathlib import Path

from test_news_v3 import _decisions_for_news, _published_silver_v2

from vietnamese_topic_classification.data.gold import GoldV3Pipeline
from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.silver.v3 import SilverV3Pipeline
from vietnamese_topic_classification.training import (
    ClassifierConfig,
    ExperimentConfig,
    FeatureConfig,
)
from vietnamese_topic_classification.training.runner import run_experiment


def test_v3_baseline_uses_gold_v3_label_mapping_and_never_evaluates_test(
    tmp_path: Path,
) -> None:
    silver_v2 = _published_silver_v2(tmp_path)
    decisions = _decisions_for_news(silver_v2, tmp_path / "decisions.jsonl")
    silver_v3_result = SilverV3Pipeline(silver_v2, decisions).run()
    silver_v3 = SilverReader(
        tmp_path, Path(silver_v3_result["paths"]["report"]).name,
    )
    gold_v3_result = GoldV3Pipeline(silver_v3).run()
    manifest_name = Path(gold_v3_result["paths"]["manifest"]).name

    result = run_experiment(
        ExperimentConfig(
            name="tfidf_logreg_v3_test",
            gold_manifest_name=manifest_name,
            features=FeatureConfig(params={
                "ngram_range": (1, 2),
                "min_df": 1,
                "max_features": 1000,
            }),
            classifier=ClassifierConfig(
                kind="logistic_regression",
                params={"max_iter": 200},
            ),
        ),
        data_lake_root=tmp_path,
        output_root=tmp_path / "experiments" / "tfidf_baselines_v3",
    )

    report = result["validation_report"]
    assert len(report["per_class_f1"]) == 27
    assert len(report["confusion_matrix"]) == 27
    artifact_dir = Path(result["artifact_dir"])
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["lineage"]["gold_source"]["silver_version"] == "v3"
    assert metadata["test_evaluated"] is False
