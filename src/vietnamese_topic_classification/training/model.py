"""Typed training contracts; Gold's source schema remains unchanged."""

from __future__ import annotations

from typing import TypedDict


class TextSplit(TypedDict):
    """Aligned fields from exactly one immutable Gold partition."""

    ids: list[str]
    texts: list[str]
    labels: list[int]


class DatasetSplits(TypedDict):
    """Train, validation and test are kept separate to prevent leakage."""

    train: TextSplit
    validation: TextSplit
    test: TextSplit


class ExperimentResult(TypedDict):
    """Minimal metadata returned by a future experiment runner."""

    experiment_name: str
    gold_manifest_name: str
    validation_metrics: dict[str, float]
    artifact_dir: str
