"""Label distribution, representativeness checks, length profile and signature terms."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats

from .hf_parquet import EMPTY_LABEL, NULL_LABEL
from .plots import COLORS, apply_style, horizontal_boxplot, save_figure

LENGTH_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def merge_counts(per_file: Sequence[Mapping[str, int]]) -> Counter:
    total: Counter = Counter()
    for counts in per_file:
        total.update(counts)
    return total


def label_kind(label: str, known: Sequence[str]) -> str:
    if label == NULL_LABEL:
        return "null_value"
    if label == EMPTY_LABEL:
        return "empty_value"
    return "known" if label in known else "unknown"


def counts_table(counts: Mapping[str, int], known: Sequence[str]) -> pd.DataFrame:
    """Every label value seen (including null/empty/unknown) plus known labels with 0."""
    merged = {label: 0 for label in known}
    merged.update({k: int(v) for k, v in counts.items()})
    total = sum(merged.values())
    table = pd.DataFrame({"label": list(merged), "count": list(merged.values())})
    table["pct"] = 100 * table["count"] / total if total else 0.0
    table = table.sort_values(["count", "label"], ascending=[False, True]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table["kind"] = [label_kind(x, known) for x in table["label"]]
    return table


def max_min_ratio(table: pd.DataFrame) -> float | None:
    present = table.loc[(table["kind"] == "known") & (table["count"] > 0), "count"]
    return float(present.max() / present.min()) if len(present) else None


def file_position_table(
    scans: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> pd.DataFrame:
    """Row percentage of each label inside every shard (shard order = file position)."""
    rows = []
    for scan in scans:
        n = scan["meta"]["num_rows"]
        row: dict[str, Any] = {"file_index": scan["meta"]["index"], "rows": n}
        for label in labels:
            row[label] = 100 * scan["counts"].get(label, 0) / n if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def position_drift(positions: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """Per label: min/max share across shards and the correlation with file position."""
    out = []
    for label in labels:
        series = positions[label].to_numpy(float)
        corr = (
            float(np.corrcoef(positions["file_index"], series)[0, 1])
            if series.std() > 0 and len(series) > 2
            else 0.0
        )
        out.append(
            {
                "label": label,
                "pct_min": series.min(),
                "pct_max": series.max(),
                "pct_std": series.std(),
                "corr_with_file_index": corr,
            }
        )
    return pd.DataFrame(out).sort_values("pct_std", ascending=False).reset_index(drop=True)


def fig_full_label_distribution(table: pd.DataFrame, path: Path) -> Path:
    apply_style()
    data = table[table["count"] > 0].sort_values("count")
    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.35)))
    median = data["count"].median()
    bars = ax.barh(
        data["label"], data["count"],
        color=[COLORS["teal"] if v < median else COLORS["blue"] for v in data["count"]],
    )
    ax.bar_label(bars, labels=[f"{p:.2f}%" for p in data["pct"]], padding=3, fontsize=8)
    ax.set_xlim(0, data["count"].max() * 1.15)
    ax.set(xlabel="Số bài", title="Toàn bộ dataset · phân bổ nhãn domain")
    return save_figure(fig, path)


def fig_label_by_file_position(
    positions: pd.DataFrame, labels: Sequence[str], path: Path
) -> Path:
    apply_style()
    matrix = positions[list(labels)].to_numpy(float).T
    fig, ax = plt.subplots(figsize=(13, max(6, len(labels) * 0.34)))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Vị trí file (shard)")
    ax.set_title("Tỉ lệ % của từng nhãn trong mỗi shard (mỗi cột cộng lại = 100%)")
    fig.colorbar(image, ax=ax, label="% bài trong shard", fraction=0.025)
    return save_figure(fig, path)


def compare_distributions(
    reference: Mapping[str, int], other: Mapping[str, int], labels: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare ``other`` against ``reference``: percentage gaps, TVD, chi-square."""
    ref = np.array([reference.get(x, 0) for x in labels], float)
    obs = np.array([other.get(x, 0) for x in labels], float)
    if ref.sum() == 0 or obs.sum() == 0:
        raise ValueError("Cannot compare empty distributions")
    p, q = ref / ref.sum(), obs / obs.sum()
    table = pd.DataFrame(
        {
            "label": list(labels),
            "reference_pct": 100 * p,
            "other_pct": 100 * q,
            "diff_pct_points": 100 * (q - p),
            "ratio": np.where(p > 0, q / np.where(p > 0, p, 1), np.nan),
        }
    )
    expected = np.maximum(p, 1e-12)
    expected = expected / expected.sum() * obs.sum()
    chi2, pvalue = stats.chisquare(obs, f_exp=expected)
    summary = {
        "n_other": float(obs.sum()),
        "total_variation_distance": float(0.5 * np.abs(p - q).sum()),
        "max_abs_gap_pct_points": float(100 * np.abs(p - q).max()),
        "chi2": float(chi2),
        "dof": float(len(labels) - 1),
        "p_value": float(pvalue),
    }
    return table.sort_values("diff_pct_points", key=np.abs, ascending=False).reset_index(
        drop=True
    ), summary


def fig_paired_distribution(
    table: pd.DataFrame, path: Path, *, reference_name: str, other_name: str, title: str
) -> Path:
    apply_style()
    data = table.sort_values("reference_pct")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.38)))
    ax.barh(y + 0.2, data["reference_pct"], height=0.4, color=COLORS["blue"], label=reference_name)
    ax.barh(y - 0.2, data["other_pct"], height=0.4, color=COLORS["amber"], label=other_name)
    ax.set_yticks(y, data["label"], fontsize=8)
    ax.set(xlabel="% bài", title=title)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def length_profile(sample: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """Chars and whitespace-separated 'tiếng' per label (a tiếng is not a model token)."""
    rows = []
    for label in labels:
        part = sample[sample["domain"] == label]
        if part.empty:
            continue
        row: dict[str, Any] = {"label": label, "n": len(part)}
        for column, prefix in (("n_chars", "chars"), ("n_tokens", "tieng")):
            for q in LENGTH_QUANTILES:
                row[f"{prefix}_p{int(q * 100)}"] = float(part[column].quantile(q))
        row["share_gt_256_tieng"] = float((part["n_tokens"] > 256).mean())
        row["share_gt_512_tieng"] = float((part["n_tokens"] > 512).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fig_length_by_label(sample: pd.DataFrame, labels: Sequence[str], path: Path) -> Path:
    apply_style()
    present = [x for x in labels if (sample["domain"] == x).any()]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, len(present) * 0.4)), sharey=True)
    for ax, column, name in (
        (axes[0], "n_chars", "Số ký tự (trục log)"),
        (axes[1], "n_tokens", "Số tiếng (tách theo khoảng trắng, trục log)"),
    ):
        data = [sample.loc[sample["domain"] == x, column].clip(lower=1) for x in present]
        box = horizontal_boxplot(ax, data)
        for patch in box["boxes"]:
            patch.set(facecolor=COLORS["teal"], alpha=0.65)
        ax.set_xscale("log")
        ax.set(xlabel=name)
    axes[0].set_yticks(range(1, len(present) + 1), present, fontsize=8)
    fig.suptitle("Độ dài bài theo nhãn (P5–P95 ẩn ngoại lệ)")
    return save_figure(fig, path)


def top_terms_by_label(
    X: sparse.csr_matrix, y: np.ndarray, feature_names: np.ndarray, classes: Sequence[str],
    *, top_k: int = 20,
) -> pd.DataFrame:
    """One-vs-rest chi-square style score on TF-IDF mass, positive association only.

    ``score = (O - E)^2 / E`` where O is the TF-IDF mass of a term inside the label and
    E the mass expected if the term were spread in proportion to label size.
    """
    k = len(classes)
    onehot = sparse.csr_matrix(
        (np.ones(len(y)), (y, np.arange(len(y)))), shape=(k, len(y)), dtype=np.float64
    )
    observed = np.asarray((onehot @ X).todense())
    term_total = observed.sum(axis=0)
    label_share = observed.sum(axis=1) / max(observed.sum(), 1e-12)
    expected = np.outer(label_share, term_total)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    score[observed <= expected] = 0.0
    rows = []
    for i, label in enumerate(classes):
        top = np.argsort(-score[i])[:top_k]
        for rank, j in enumerate(top, start=1):
            if score[i, j] <= 0:
                break
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "term": str(feature_names[j]),
                    "score": float(score[i, j]),
                    "lift": float(observed[i, j] / expected[i, j]),
                }
            )
    return pd.DataFrame(rows, columns=["label", "rank", "term", "score", "lift"])
