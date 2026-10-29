"""Deterministic multiclass metrics for the canonical domain label order."""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .model import ClassificationMetrics


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    label_names: Sequence[str],
) -> ClassificationMetrics:
    """Include zero-support classes in macro-F1 and the confusion matrix."""
    labels = list(range(len(label_names)))
    if not labels or len(set(label_names)) != len(labels):
        raise ValueError("label_names must be nonempty and unique")
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Predictions and targets must have equal, nonzero lengths")
    if any(type(int(v)) is not int or int(v) not in labels for v in [*y_true, *y_pred]):
        raise ValueError("Predicted or reference labels outside the canonical range")
    true = [int(v) for v in y_true]
    pred = [int(v) for v in y_pred]
    class_f1 = f1_score(true, pred, average=None, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(
            f1_score(true, pred, average="weighted", labels=labels, zero_division=0)
        ),
        "per_class_f1": {name: float(score) for name, score in zip(label_names, class_f1)},
        "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
    }
