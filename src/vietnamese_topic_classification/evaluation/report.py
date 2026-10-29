"""Evaluation report serialization boundary; no files are written yet."""

from __future__ import annotations

from pathlib import Path

from .model import ClassificationMetrics


def write_evaluation_report(
    metrics: ClassificationMetrics,
    *,
    output_path: Path,
) -> Path:
    """Eventually publish JSON metrics without overwriting an existing report."""
    raise NotImplementedError("Skeleton only: implement evaluation report publication")
