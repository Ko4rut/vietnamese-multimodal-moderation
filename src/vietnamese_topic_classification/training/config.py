"""Typed experiment configuration; no YAML loading or estimator creation yet."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Future feature-extraction settings, e.g. TF-IDF."""

    kind: str = "tfidf"
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("Feature kind must not be empty")


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """Future estimator settings, e.g. logistic regression or LinearSVC."""

    kind: str = "logistic_regression"
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("Classifier kind must not be empty")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Identity, Gold snapshot and model selection settings for a single run."""

    name: str
    gold_manifest_name: str
    features: FeatureConfig = field(default_factory=FeatureConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    seed: int = 42
    selection_metric: str = "macro_f1"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name):
            raise ValueError("Experiment name must be a safe alphanumeric slug")
        manifest = self.gold_manifest_name
        if (
            not isinstance(manifest, str)
            or not manifest.endswith(".json")
            or Path(manifest).name != manifest
            or "\\" in manifest
        ):
            raise ValueError("gold_manifest_name must be a JSON filename, not a path")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not self.selection_metric.strip():
            raise ValueError("selection_metric must not be empty")
