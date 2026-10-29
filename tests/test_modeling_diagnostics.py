"""Offline smoke tests of diagnostic figures and write-once publication."""

import json

import pytest

from vietnamese_topic_classification.evaluation.diagnostics import save_diagnostics
from vietnamese_topic_classification.training import ClassifierConfig, ExperimentConfig
from vietnamese_topic_classification.training.factory import build_pipeline


@pytest.mark.parametrize("classifier", ["logistic_regression", "linear_svc"])
def test_diagnostics_create_real_figures_and_history(tmp_path, classifier):
    texts = [
        "bóng đá", "đội tuyển", "trận đấu", "cầu thủ",
        "phần mềm", "máy tính", "công nghệ", "lập trình",
    ] * 3
    labels = [0] * 4 + [1] * 4
    labels *= 3
    validation_texts = ["bóng đá", "đội tuyển", "công nghệ", "lập trình"]
    validation_labels = [0, 0, 1, 1]
    train = {"ids": [str(i) for i in range(len(texts))], "texts": texts, "labels": labels}
    validation = {
        "ids": ["a", "b", "c", "d"],
        "texts": validation_texts,
        "labels": validation_labels,
    }
    config = ExperimentConfig(
        name="demo", gold_manifest_name="gold.json", classifier=ClassifierConfig(kind=classifier)
    )
    pipeline = build_pipeline(config)
    pipeline.set_params(tfidf__min_df=1)
    pipeline.fit(texts, labels)
    destination = tmp_path / "diagnostics_v1"
    report = save_diagnostics(
        pipeline, train, validation, label_names=["sports", "tech"],
        output_dir=destination, fractions=(0.5, 1.0),
    )
    assert len(report["learning_curve"]) == 2
    assert report["learning_curve"][-1]["train_size"] == len(texts)
    assert len(report["per_class"]) == 2
    assert report["solver_iterations"] is not None
    assert json.loads((destination / "diagnostics.json").read_text())["validation_rows"] == 4
    for name in (
        "train_vs_validation.png", "learning_curve_by_train_size.png",
        "correct_counts_by_class.png", "recall_by_class.png",
        "per_class_f1.png", "confusion_matrix_normalized.png",
    ):
        assert (destination / name).is_file(), name
        assert (destination / name).stat().st_size > 100
    with pytest.raises(FileExistsError):
        save_diagnostics(
            pipeline, train, validation, label_names=["sports", "tech"],
            output_dir=destination, fractions=(0.5, 1.0),
        )


def test_bad_fraction_is_rejected(tmp_path):
    config = ExperimentConfig(name="demo", gold_manifest_name="gold.json")
    pipeline = build_pipeline(config)
    with pytest.raises(ValueError, match="fractions"):
        save_diagnostics(
            pipeline,
            {"ids": ["1"], "texts": ["sports"], "labels": [0]},
            {"ids": ["2"], "texts": ["tech"], "labels": [1]},
            label_names=["sports", "tech"],
            output_dir=tmp_path / "charts",
            fractions=(1.0, 0.5),
        )
