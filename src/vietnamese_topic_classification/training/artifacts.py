"""Immutable experiment artifact publication boundary."""

from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig
from .model import ExperimentResult


def save_artifacts(
    model: object,
    result: ExperimentResult,
    config: ExperimentConfig,
    *,
    output_root: Path,
) -> Path:
    """Eventually save fitted pipeline, effective config and source lineage safely."""
    raise NotImplementedError("Skeleton only: implement artifact persistence")
