"""Typed training contracts; Gold remains unchanged."""

from typing import TypedDict

from typing_extensions import NotRequired

from vietnamese_topic_classification.evaluation.model import ClassificationMetrics


class TextSplit(TypedDict):
    ids: list[str]
    texts: list[str]
    labels: list[int]


class DatasetSplits(TypedDict):
    train: TextSplit
    validation: TextSplit
    test: TextSplit


class ExperimentResult(TypedDict):
    experiment_name: str
    gold_manifest_name: str
    validation_metrics: dict[str, float]
    artifact_dir: str
    validation_report: NotRequired[ClassificationMetrics]
