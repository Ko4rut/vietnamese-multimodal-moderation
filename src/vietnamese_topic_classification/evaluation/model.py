"""Typed evaluation outputs for a single-label, multiclass experiment."""

from __future__ import annotations

from typing import TypedDict


class ClassificationMetrics(TypedDict):
    """Planned classification report; scores and matrix are not yet computed."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
