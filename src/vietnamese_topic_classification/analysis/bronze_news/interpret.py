"""Cluster interpretation: keywords, representatives, V3 label hints, neighbor overlap."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .clustering import NOISE

SEED_KEYWORDS_PATH = Path(__file__).with_name("seed_keywords.yaml")
HINT_NOTE = "gợi ý có điểm số từ keyword hạt giống, CẦN NGƯỜI KIỂM TRA, không phải nhãn"


def load_seed_keywords(path: Path | None = None) -> dict[str, list[str]]:
    data = yaml.safe_load((path or SEED_KEYWORDS_PATH).read_text(encoding="utf-8"))
    return {label: [str(t).strip().lower() for t in terms] for label, terms in data.items()}


def cluster_ids(labels: np.ndarray) -> list[int]:
    return [int(c) for c in np.unique(labels) if c != NOISE]


def cluster_keywords(X_tfidf: sparse.csr_matrix, labels: np.ndarray, feature_names: np.ndarray,
                     top_n: int = 20) -> dict[int, list[tuple[str, float]]]:
    """Top terms per cluster: mean TF-IDF of the cluster minus the corpus mean (c-TF-IDF style).

    Computed on the original sparse TF-IDF space, not on the SVD space.
    """
    overall = np.asarray(X_tfidf.mean(axis=0)).ravel()
    out: dict[int, list[tuple[str, float]]] = {}
    for c in cluster_ids(labels):
        mean = np.asarray(X_tfidf[labels == c].mean(axis=0)).ravel()
        score = mean - overall
        top = np.argsort(-score)[:top_n]
        out[c] = [(str(feature_names[i]), float(score[i])) for i in top if score[i] > 0]
    return out


def _seed_weight(seed: str, weights: dict[str, float]) -> float | None:
    """Weight a seed term earns in a cluster, or None when it does not match.

    One-syllable seed: the same unigram term. Two-syllable seed: the same bigram term (full
    weight) or, failing that, both syllables among the cluster's top terms (half weight),
    because Vietnamese words are often split into syllables by the tokenizer.
    """
    if seed in weights:
        return weights[seed]
    parts = seed.split()
    if len(parts) > 1 and all(p in weights for p in parts):
        return 0.5 * sum(weights[p] for p in parts)
    return None


def suggest_labels(keywords: dict[int, list[tuple[str, float]]], seeds: dict[str, list[str]],
                   top: int = 3) -> dict[int, list[dict[str, Any]]]:
    """Score = share of the cluster's keyword weight that matches each label's seed terms."""
    out: dict[int, list[dict[str, Any]]] = {}
    for c, kws in keywords.items():
        weights = dict(kws)
        total = sum(weights.values()) or 1.0
        scored = []
        for label, terms in seeds.items():
            hits = [(t, w) for t in dict.fromkeys(terms)
                    if (w := _seed_weight(t, weights)) is not None]
            if hits:
                scored.append({
                    "label": label,
                    "score": float(min(1.0, sum(w for _, w in hits) / total)),
                    "matched_terms": [t for t, _ in hits][:8],
                })
        scored.sort(key=lambda d: -d["score"])
        out[c] = scored[:top]
    return out


def centroids(X_lsa: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {c: normalize(X_lsa[labels == c].mean(axis=0, keepdims=True))[0]
            for c in cluster_ids(labels)}


def distance_to_centroid(X_lsa: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cosine distance of each point to its own cluster centroid (NaN for noise)."""
    out = np.full(len(labels), np.nan)
    for c, centre in centroids(X_lsa, labels).items():
        mask = labels == c
        out[mask] = 1.0 - normalize(X_lsa[mask]) @ centre
    return out


def representatives(distance: np.ndarray, labels: np.ndarray, per_cluster: int = 3
                    ) -> pd.DataFrame:
    """Row positions of the closest ('centre') and farthest ('edge') articles per cluster."""
    rows = []
    for c in cluster_ids(labels):
        idx = np.where(labels == c)[0]
        order = idx[np.argsort(distance[idx])]
        for rank, i in enumerate(order[:per_cluster], 1):
            rows.append({"cluster": c, "kind": "centre", "rank": rank, "row": int(i),
                         "distance": float(distance[i])})
        edge = [i for i in order[::-1][:per_cluster] if i not in set(order[:per_cluster])]
        for rank, i in enumerate(edge, 1):
            rows.append({"cluster": c, "kind": "edge", "rank": rank, "row": int(i),
                         "distance": float(distance[i])})
    return pd.DataFrame(rows, columns=["cluster", "kind", "rank", "row", "distance"])


def excerpt(text: str | None, chars: int) -> str:
    flat = " ".join((text or "").split())
    return flat[:chars] + ("…" if len(flat) > chars else "")


def length_by_cluster(lengths: pd.DataFrame, noise: pd.DataFrame, labels: np.ndarray,
                      short_words: int) -> pd.DataFrame:
    """Per-cluster length/noise profile; flags clusters that look like length/crawl artifacts."""
    rows = []
    for c in cluster_ids(labels):
        m = labels == c
        short = float((lengths.loc[m, "words"] < short_words).mean())
        noisy = float(noise.loc[m].any(axis=1).mean())
        rows.append({
            "cluster": c, "median_words": float(lengths.loc[m, "words"].median()),
            "median_chars": float(lengths.loc[m, "chars"].median()),
            "short_fraction": short, "noise_flag_fraction": noisy,
            "artifact_flag": bool(short >= 0.5 or noisy >= 0.3),
        })
    return pd.DataFrame(rows)


def neighbor_overlap(
    news_lsa: np.ndarray,
    news_labels: np.ndarray,
    neigh_lsa: np.ndarray,
    neigh_domains: Sequence[str],
    *,
    k: int = 15,
    news_name: str = "News",
) -> tuple[pd.DataFrame, np.ndarray]:
    """How much each News cluster looks like a neighbor label, in the News-fitted space.

    - ``nearest_label_by_centroid``: neighbor label with the highest cosine to the cluster centroid.
    - k-NN vote of every News article over the pool (News + neighbor sample), with votes
      weighted by 1/label size so a large label does not win just by volume.
      ``leak_rate`` = share of the cluster's articles whose weighted majority is a
      non-News label. This is a similarity of word usage, not evidence of mislabeling.

    Returns ``(per-cluster table, per-article majority label)``.
    """
    domains = np.asarray(list(neigh_domains))
    names = sorted(set(domains))
    pool = np.vstack([news_lsa, neigh_lsa])
    pool_labels = np.concatenate([np.full(len(news_lsa), news_name), domains])
    uniq, counts = np.unique(pool_labels, return_counts=True)
    weight = {u: 1.0 / c for u, c in zip(uniq, counts)}
    kk = min(k, len(pool) - 1)
    neigh_idx = NearestNeighbors(n_neighbors=kk + 1).fit(pool).kneighbors(news_lsa)[1][:, 1:]
    votes = np.zeros((len(news_lsa), len(uniq)))
    col = {u: i for i, u in enumerate(uniq)}
    label_col = np.array([col[n] for n in pool_labels])
    w = np.array([weight[n] for n in pool_labels])
    for r in range(neigh_idx.shape[1]):
        np.add.at(votes, (np.arange(len(news_lsa)), label_col[neigh_idx[:, r]]), w[neigh_idx[:, r]])
    votes /= votes.sum(axis=1, keepdims=True).clip(min=1e-12)
    majority = uniq[votes.argmax(axis=1)]

    neigh_centroids = {n: normalize(neigh_lsa[domains == n].mean(axis=0, keepdims=True))[0]
                       for n in names}
    rows = []
    for c, centre in centroids(news_lsa, news_labels).items():
        m = news_labels == c
        sims = {n: float(centre @ v) for n, v in neigh_centroids.items()}
        nearest = max(sims, key=sims.get)
        leak = majority[m] != news_name
        row: dict[str, Any] = {
            "cluster": c, "size": int(m.sum()),
            "nearest_label_by_centroid": nearest, "nearest_centroid_cosine": sims[nearest],
            "leak_rate": float(leak.mean()),
            "leak_top_label": (pd.Series(majority[m][leak]).mode().iat[0] if leak.any() else ""),
            "knn_share_news": float(votes[m, col[news_name]].mean()),
        }
        for n in names:
            row[f"knn_share_{n}"] = float(votes[m, col[n]].mean())
        rows.append(row)
    return pd.DataFrame(rows), majority


def build_cluster_summary(labels: np.ndarray, keywords: dict[int, list[tuple[str, float]]],
                          hints: dict[int, list[dict[str, Any]]], overlap: pd.DataFrame | None,
                          profile: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster (list-valued columns; see ``flatten_summary`` for CSV)."""
    total = len(labels)
    rows = []
    for c in cluster_ids(labels):
        size = int((labels == c).sum())
        rows.append({
            "cluster": c, "size": size, "percent": 100.0 * size / total,
            "keywords": [t for t, _ in keywords.get(c, [])],
            "label_hints": hints.get(c, []),
            "hint_note": HINT_NOTE,
        })
    out = pd.DataFrame(rows)
    out = out.merge(profile, on="cluster", how="left")
    if overlap is not None and len(overlap):
        extra = overlap.drop(columns=["size"]).rename(columns={
            "nearest_label_by_centroid": "nearest_neighbor_label"})
        out = out.merge(extra, on="cluster", how="left")
    else:
        out["nearest_neighbor_label"] = ""
        out["nearest_centroid_cosine"] = np.nan
        out["leak_rate"] = np.nan
        out["leak_top_label"] = ""
    return out


def flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """CSV-friendly version: keywords joined by '; ', hints as 'label:score'."""
    flat = summary.copy()
    flat["keywords"] = flat["keywords"].map(lambda ks: "; ".join(ks))
    flat["label_hints"] = flat["label_hints"].map(
        lambda hs: "; ".join(f"{h['label']}:{h['score']:.3f}" for h in hs))
    return flat
