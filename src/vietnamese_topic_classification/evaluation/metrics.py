"""Pure metric-calculation boundary; estimator and I/O independent."""

from __future__ import annotations

from typing import Sequence

from .model import ClassificationMetrics


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    label_names: Sequence[str],
) -> ClassificationMetrics:
    """Compute future multiclass scores from predictions and canonical labels."""
    raise NotImplementedError("Skeleton only: implement classification metrics")
