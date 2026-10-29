"""One-experiment orchestration boundary; no training implementation yet."""

from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig
from .model import ExperimentResult


def run_experiment(
    config: ExperimentConfig,
    *,
    data_lake_root: Path,
    output_root: Path,
) -> ExperimentResult:
    """Train on Gold train, select via validation, and publish immutable artifacts.

    Final test evaluation must remain a separate, explicitly invoked step.
    """
    raise NotImplementedError("Skeleton only: implement the experiment runner")
