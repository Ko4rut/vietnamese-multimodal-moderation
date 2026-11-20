"""Matplotlib figures. Every function returns a Figure; ``FigureStore`` keeps PNG bytes."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOISE_COLOR = "#b0b0b0"


class FigureStore:
    """Collects figures as PNG bytes so they can be exported later (or not at all)."""

    def __init__(self, show: bool = True) -> None:
        self.png: dict[str, bytes] = {}
        self.show = show

    def add(self, name: str, fig: plt.Figure) -> None:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        self.png[name] = buffer.getvalue()
        if self.show:
            plt.show()
        plt.close(fig)

    def write(self, directory: Path) -> list[str]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in self.png.items():
            (directory / name).write_bytes(data)
        return sorted(self.png)


def _palette(n: int) -> list:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def length_histograms(lengths: pd.DataFrame, stats: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, column, title in ((axes[0], "chars", "Số ký tự"), (axes[1], "words", "Số từ (âm tiết)")):
        values = lengths[column].clip(lower=1)
        ax.hist(np.log10(values), bins=40, color="#4c78a8")
        ticks = [t for t in (1, 2, 3, 4, 5) if t <= np.log10(values.max()) + 0.5]
        ax.set_xticks(ticks, [f"{10 ** t:,.0f}" for t in ticks])
        for p, style in (("p50", "-"), ("p90", "--"), ("p99", ":")):
            ax.axvline(np.log10(max(stats[column][p], 1)), color="#e45756", ls=style, lw=1,
                       label=f"{p}={stats[column][p]:,.0f}")
        ax.set_title(f"{title} (trục log)")
        ax.set_ylabel("số bài")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def top_terms_bars(terms: dict[str, pd.DataFrame], top: int = 20) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, title) in zip(axes, (("unigram", "Từ đơn"), ("bigram", "Cụm 2 âm tiết"))):
        data = terms[name].head(top).iloc[::-1]
        ax.barh(data["term"], data["rate"] * 100, color="#4c78a8")
        ax.set_title(f"{title}: % bài chứa")
    fig.tight_layout()
    return fig


def variance_curve(ratio: np.ndarray) -> plt.Figure:
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(np.arange(1, len(cum) + 1), cum * 100, color="#4c78a8")
    ax.set_xlabel("số chiều SVD")
    ax.set_ylabel("% phương sai giải thích (tích lũy)")
    ax.set_title(f"LSA: {len(cum)} chiều giữ {cum[-1] * 100:.1f}% phương sai")
    ax.grid(alpha=0.3)
    return fig


def projections(pca: np.ndarray, tsne: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, pts, title in ((axes[0], pca, "PCA 2D"), (axes[1], tsne, "t-SNE 2D")):
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5, color="#4c78a8")
        ax.set_title(title + " (chưa tô màu cụm)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def kdistance_plot(kd: np.ndarray, k: int, eps: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(kd, color="#4c78a8")
    ax.axhline(eps, color="#e45756", ls="--", label=f"eps gợi ý = {eps:.3f}")
    ax.set_xlabel("điểm (sắp theo khoảng cách)")
    ax.set_ylabel(f"khoảng cách tới láng giềng thứ {k}")
    ax.set_title("k-distance: mật độ có đều không?")
    ax.legend()
    return fig


def cosine_hist(cos: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(cos, bins=60, color="#4c78a8")
    ax.axvline(cos.mean(), color="#e45756", ls="--", label=f"trung bình {cos.mean():.2f}")
    ax.set_xlabel("khoảng cách cosine giữa hai bài")
    ax.set_ylabel("số cặp")
    ax.legend()
    ax.set_title("Phân bố khoảng cách cặp bài")
    return fig


def k_curves_plot(curve: pd.DataFrame, stability: pd.DataFrame | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    axes[0].plot(curve["k"], curve["inertia"], "o-")
    axes[0].set_title("Elbow (KMeans inertia)")
    axes[1].plot(curve["k"], curve["silhouette"], "o-", color="#54a24b")
    axes[1].set_title("Silhouette cosine (KMeans)")
    axes[2].plot(curve["k"], curve["gmm_bic"], "o-", label="BIC")
    axes[2].plot(curve["k"], curve["gmm_aic"], "s--", label="AIC")
    axes[2].legend()
    axes[2].set_title("GaussianMixture (diag)")
    if stability is not None and len(stability):
        axes[3].plot(stability["k"], stability["ari_mean"], "o-", color="#b279a2")
        axes[3].set_ylim(0, 1.02)
    axes[3].set_title("Ổn định KMeans (ARI giữa seed)")
    for ax in axes:
        ax.set_xlabel("k")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _scatter_clusters(ax, coords: np.ndarray, labels: np.ndarray, size: float = 4) -> None:
    ids = sorted(int(c) for c in np.unique(labels) if c >= 0)
    colors = _palette(max(len(ids), 1))
    mask = labels < 0
    if mask.any():
        ax.scatter(coords[mask, 0], coords[mask, 1], s=size, color=NOISE_COLOR, alpha=0.5)
    for color, c in zip(colors, ids):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=size, color=color, alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])


def algorithm_grid(coords: np.ndarray, picks: Sequence[tuple[str, np.ndarray, float, str]],
                   cols: int = 4) -> plt.Figure:
    """scikit-learn style grid: each panel = one algorithm on the same t-SNE coordinates."""
    rows = max(1, int(np.ceil(len(picks) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, labels, seconds, extra) in zip(axes.ravel(), picks):
        ax.axis("on")
        _scatter_clusters(ax, coords, labels)
        ax.set_title(f"{title}\n{extra}", fontsize=8)
        ax.text(0.98, 0.02, f"{seconds:.2f}s", transform=ax.transAxes, ha="right", fontsize=9)
    fig.tight_layout()
    return fig


def main_clusters_plot(coords: np.ndarray, labels: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _scatter_clusters(ax, coords, labels, size=7)
    for c in sorted(int(c) for c in np.unique(labels) if c >= 0):
        centre = np.median(coords[labels == c], axis=0)
        ax.text(centre[0], centre[1], str(c), fontsize=13, weight="bold", ha="center",
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.7})
    ax.set_title(title)
    return fig


def cluster_sizes(summary: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(summary["cluster"].astype(str), summary["size"], color="#4c78a8")
    for bar, pct in zip(bars, summary["percent"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha="center",
                va="bottom", fontsize=8)
    ax.set_xlabel("cụm")
    ax.set_ylabel("số bài")
    ax.set_title(title)
    return fig


def neighbor_overlay(coords_news: np.ndarray, news_labels: np.ndarray, coords_neigh: np.ndarray,
                     neigh_domains: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _scatter_clusters(axes[0], coords_news, news_labels, size=6)
    axes[0].set_title("Chỉ News (màu = cụm)")
    axes[1].scatter(coords_news[:, 0], coords_news[:, 1], s=5, color="#cccccc", label="News")
    names = sorted(set(neigh_domains))
    for color, name in zip(_palette(len(names)), names):
        m = neigh_domains == name
        axes[1].scatter(coords_neigh[m, 0], coords_neigh[m, 1], s=7, color=color, alpha=0.7,
                        label=name)
    axes[1].legend(fontsize=7, markerscale=2)
    axes[1].set_title("News (xám) và nhãn láng giềng (t-SNE chung, chỉ để vẽ)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def overlap_heatmap(overlap: pd.DataFrame) -> plt.Figure:
    cols = [c for c in overlap.columns if c.startswith("knn_share_")]
    data = overlap[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 2, 0.5 * len(overlap) + 2))
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)), [c.replace("knn_share_", "") for c in cols], rotation=45,
                  ha="right")
    ax.set_yticks(range(len(overlap)), [f"cụm {c}" for c in overlap["cluster"]])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="tỉ trọng láng giềng gần nhất")
    ax.set_title("Mỗi cụm News 'giống' nhãn nào (k-NN cân bằng theo nhãn)")
    fig.tight_layout()
    return fig


def length_by_cluster_plot(words: pd.Series, labels: np.ndarray) -> plt.Figure:
    ids = sorted(int(c) for c in np.unique(labels) if c >= 0)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.boxplot([words[labels == c].clip(lower=1) for c in ids], tick_labels=[str(c) for c in ids],
               showfliers=False)
    ax.set_yscale("log")
    ax.set_xlabel("cụm")
    ax.set_ylabel("số từ (log)")
    ax.set_title("Độ dài bài theo cụm")
    return fig
