"""Offline smoke tests for modeling and evaluation."""

import json

import joblib
import pytest

from vietnamese_topic_classification.evaluation.metrics import calculate_metrics
from vietnamese_topic_classification.evaluation.report import write_evaluation_report
from vietnamese_topic_classification.training import ClassifierConfig, ExperimentConfig
from vietnamese_topic_classification.training.artifacts import save_artifacts
from vietnamese_topic_classification.training.factory import build_pipeline


@pytest.mark.parametrize("classifier", ["logistic_regression", "linear_svc"])
def test_training_pipeline(classifier):
    config = ExperimentConfig(
        name="smoke", gold_manifest_name="gold.json",
        classifier=ClassifierConfig(kind=classifier, params={"max_iter": 100} if classifier == "logistic_regression" else {}),
    )
    pipeline = build_pipeline(config)
    texts = ["mèo và chó nuôi", "thú cưng trong nhà", "công nghệ máy tính", "phần mềm máy tính"]
    pipeline.set_params(tfidf__min_df=1)
    pipeline.fit(texts, [0, 0, 1, 1])
    assert len(pipeline.predict(["con mèo", "máy tính"])) == 2


def test_metrics_keep_canonical_order():
    scores = calculate_metrics([0, 1, 1], [0, 0, 1], label_names=["a", "b", "c"])
    assert len(scores["confusion_matrix"]) == 3
    assert list(scores["per_class_f1"]) == ["a", "b", "c"]
    assert scores["macro_f1"] < 1.0


def test_metric_errors():
    with pytest.raises(ValueError):
        calculate_metrics([], [], label_names=["a"])
    with pytest.raises(ValueError):
        calculate_metrics([0], [2], label_names=["a", "b"])


def test_reports_are_write_once(tmp_path):
    report = calculate_metrics([0], [0], label_names=["a"])
    output = tmp_path / "metrics.json"
    write_evaluation_report(report, output_path=output)
    assert json.loads(output.read_text())["macro_f1"] == 1
    with pytest.raises(FileExistsError):
        write_evaluation_report(report, output_path=output)


def test_artifact_publication(tmp_path):
    config = ExperimentConfig(name="baseline", gold_manifest_name="gold.json")
    model = build_pipeline(config)
    model.set_params(tfidf__min_df=1)
    model.fit(["thể thao", "bóng đá", "sách văn học", "tiểu thuyết"], [0, 0, 1, 1])
    result = {
        "experiment_name": config.name, "gold_manifest_name": config.gold_manifest_name,
        "validation_metrics": {"macro_f1": 1.0}, "artifact_dir": "",
    }
    path = save_artifacts(model, result, config, output_root=tmp_path)
    assert joblib.load(path / "model.joblib").predict(["bóng đá"]).shape == (1,)
    assert (path / "metadata.json").is_file()
    with pytest.raises(FileExistsError):
        save_artifacts(model, result, config, output_root=tmp_path)
