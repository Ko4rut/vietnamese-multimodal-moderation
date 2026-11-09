"""Shared matplotlib style (same palette as ``notebooks/v1/eda/01_silver_eda.ipynb``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"blue": "#2878B5", "teal": "#009F93", "red": "#D7646C", "amber": "#D59432"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.figsize": (11, 5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_figure(fig: Any, path: Path) -> Path:
    """Save as PNG and close; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def horizontal_boxplot(ax: Any, data: list[Any]) -> dict[str, Any]:
    """Boxplot drawn horizontally across matplotlib versions (``vert`` is deprecated)."""
    try:
        return ax.boxplot(data, orientation="horizontal", showfliers=False, patch_artist=True)
    except TypeError:  # matplotlib < 3.10
        return ax.boxplot(data, vert=False, showfliers=False, patch_artist=True)
