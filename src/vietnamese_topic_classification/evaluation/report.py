"""Write-once JSON evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

from .model import ClassificationMetrics


def write_evaluation_report(
    metrics: ClassificationMetrics,
    *,
    output_path: Path,
) -> Path:
    """Create one readable report without overwriting an existing result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return output_path
