"""Shape diagnostics of the LSA space: cluster tendency, density, 2-D projections."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .config import BronzeNewsConfig


def hopkins(X: np.ndarray, *, seed: int, repeats: int = 10, null: str = "permute",
            sample: int | None = None) -> tuple[float, float]:
    """Hopkins statistic (mean, std over repeats). ~0.5: no structure; -> 1: clusterable.

    ``null="permute"`` draws synthetic points by shuffling each column independently
    (keeps the marginals, destroys cross-column structure); ``"uniform"`` uses the
    bounding box. Both ignore correlations, so in high dimensions the statistic is
    inflated: read it as a rough sign, not as proof.
    """
    n = len(X)
    m = sample or int(min(max(30, n // 10), 500))
    m = min(m, n - 1)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    rng = np.random.default_rng(seed)
    values = []
    lo, hi = X.min(axis=0), X.max(axis=0)
    for _ in range(repeats):
        idx = rng.choice(n, size=m, replace=False)
        w = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]  # nearest *other* real point
        if null == "uniform":
            synthetic = rng.uniform(lo, hi, size=(m, X.shape[1]))
        else:
            rows = rng.integers(0, n, size=(m, X.shape[1]))
            synthetic = X[rows, np.arange(X.shape[1])]
        u = nn.kneighbors(synthetic, n_neighbors=1)[0][:, 0]
        values.append(float(u.sum() / (u.sum() + w.sum())))
    return float(np.mean(values)), float(np.std(values))


def kdistance(X: np.ndarray, k: int) -> np.ndarray:
    """Sorted (ascending) distance of every point to its k-th nearest neighbor."""
    k = min(k, len(X) - 1)
    dist = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, k]
    return np.sort(dist)


def knee_eps(sorted_dist: np.ndarray) -> float:
    """Knee of the k-distance curve: point farthest below the chord (suggested DBSCAN eps)."""
    n = len(sorted_dist)
    x = np.linspace(0.0, 1.0, n)
    span = sorted_dist[-1] - sorted_dist[0]
    y = (sorted_dist - sorted_dist[0]) / span if span > 0 else np.zeros(n)
    eps = float(sorted_dist[int(np.argmax(x - y))])
    lo, hi = np.percentile(sorted_dist, 50), np.percentile(sorted_dist, 99)
    return float(np.clip(eps, lo, hi))


def cosine_distance_sample(X: np.ndarray, *, n: int, seed: int) -> np.ndarray:
    """Cosine distances (1 - cos) of all pairs inside a seeded sample of rows."""
    rng = np.random.default_rng(seed)
    take = min(n, len(X))
    Z = normalize(X[np.sort(rng.choice(len(X), take, replace=False))])
    sim = Z @ Z.T
    return np.clip(1.0 - sim[np.triu_indices(take, k=1)], 0.0, 2.0)


def project_pca(X: np.ndarray, seed: int) -> np.ndarray:
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def project_tsne(X: np.ndarray, *, seed: int, perplexity: float = 30.0,
                 max_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """t-SNE (visualization only). Returns ``(coords, row_index)``; sampled if too large."""
    n = len(X)
    idx = np.arange(n)
    if max_points and n > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
    perp = float(min(perplexity, max(2.0, (len(idx) - 1) / 3.0)))
    coords = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed,
                  metric="cosine").fit_transform(X[idx])
    return coords, idx


def k_curves(X: np.ndarray, k_range: tuple[int, int], seed: int) -> pd.DataFrame:
    """KMeans inertia + silhouette(cosine) and GaussianMixture(diag) BIC/AIC per k."""
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X)
        sil = float(silhouette_score(X, km.labels_, metric="cosine",
                                     sample_size=min(len(X), 2500), random_state=seed))
        sizes = np.bincount(km.labels_, minlength=k)
        gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4,
                              random_state=seed, max_iter=100).fit(X)
        rows.append({
            "k": k, "inertia": float(km.inertia_), "silhouette": sil,
            "size_balance": float(sizes.min() / sizes.max()),
            "gmm_bic": float(gmm.bic(X)), "gmm_aic": float(gmm.aic(X)),
        })
    return pd.DataFrame(rows)


def diagnose(X: np.ndarray, cfg: BronzeNewsConfig) -> tuple[dict[str, Any], dict[str, np.ndarray],
                                                            pd.DataFrame]:
    """All numbers needed to choose algorithms. Returns (metrics, arrays, k_curve table)."""
    seed = cfg.seed
    h_full, h_full_sd = hopkins(X, seed=seed, repeats=cfg.hopkins_repeats, null="permute")
    low = X[:, : min(10, X.shape[1])]
    h_low, h_low_sd = hopkins(low, seed=seed, repeats=cfg.hopkins_repeats, null="uniform")
    kd = kdistance(X, cfg.kdist_k)
    cos = cosine_distance_sample(X, n=cfg.cosine_pairs_sample, seed=seed)
    q1, q3 = np.percentile(kd, [25, 75])
    curve = k_curves(X, cfg.k_range, seed)
    best_sil = curve.loc[curve["silhouette"].idxmax()] if len(curve) else None
    best_bic = curve.loc[curve["gmm_bic"].idxmin()] if len(curve) else None
    metrics: dict[str, Any] = {
        "n": int(len(X)), "dim": int(X.shape[1]),
        "hopkins_full": h_full, "hopkins_full_std": h_full_sd,
        "hopkins_top10_dims": h_low, "hopkins_top10_dims_std": h_low_sd,
        "hopkins": float(min(h_full, h_low)),
        "kdist_k": int(min(cfg.kdist_k, len(X) - 1)),
        "kdist_median": float(np.median(kd)),
        "kdist_p95": float(np.percentile(kd, 95)),
        "kdist_ratio_p95_p50": float(np.percentile(kd, 95) / max(np.median(kd), 1e-12)),
        "kdist_knee_eps": knee_eps(kd),
        "outlier_fraction": float((kd > q3 + 1.5 * (q3 - q1)).mean()),
        "cosdist_mean": float(cos.mean()), "cosdist_std": float(cos.std()),
        "cosdist_p5": float(np.percentile(cos, 5)), "cosdist_p95": float(np.percentile(cos, 95)),
        "cosdist_cv": float(cos.std() / max(cos.mean(), 1e-12)),
        "kmeans_best_k_silhouette": int(best_sil["k"]) if best_sil is not None else None,
        "kmeans_best_silhouette": float(best_sil["silhouette"]) if best_sil is not None else None,
        "kmeans_size_balance_at_best_k": (
            float(best_sil["size_balance"]) if best_sil is not None else None),
        "gmm_best_k_bic": int(best_bic["k"]) if best_bic is not None else None,
    }
    return metrics, {"kdistance": kd, "cosine_distances": cos}, curve


def shape_profile(m: dict[str, Any]) -> dict[str, Any]:
    """Turn raw numbers into labels using fixed, printed heuristic thresholds."""
    h = m["hopkins"]
    tendency = "rõ" if h >= 0.75 else "yếu" if h >= 0.60 else "không rõ"
    sil = m.get("kmeans_best_silhouette")
    separation = (
        "chưa rõ" if sil is None else "rõ" if sil >= 0.15 else "yếu" if sil >= 0.05 else "rất yếu"
    )
    uneven = bool(m["kdist_ratio_p95_p50"] >= 1.3 or m["outlier_fraction"] >= 0.05)
    balance = m.get("kmeans_size_balance_at_best_k")
    return {
        "tendency": tendency, "separation": separation, "uneven_density": uneven,
        "many_outliers": bool(m["outlier_fraction"] >= 0.05),
        "balanced_sizes": bool(balance is not None and balance >= 0.3),
        "high_dimensional": bool(m["dim"] >= 30),
    }


THRESHOLDS = (
    "Ngưỡng (heuristic, chỉ để gợi ý): Hopkins ≥ 0.75 rõ, 0.60–0.75 yếu, < 0.60 không rõ; "
    "silhouette KMeans ≥ 0.15 tách rõ, 0.05–0.15 yếu, < 0.05 rất yếu; mật độ không đều khi "
    "p95/p50 của k-distance ≥ 1.3 hoặc ≥ 5% điểm lẻ; cụm 'đều' khi cụm nhỏ nhất ≥ 30% cụm lớn nhất."
)


def interpret_shape(m: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    """Vietnamese bullet points generated from the numbers (no fixed conclusion)."""
    lines = [
        f"- **Xu hướng tạo cụm (Hopkins = {m['hopkins']:.2f}):** {profile['tendency']}. "
        + ("Dữ liệu gần như không có cấu trúc cụm rõ so với dữ liệu ngẫu nhiên → kết luận thận trọng; "
           "cụm tìm được nên coi là cách chia tiện lợi, không phải ranh giới thật."
           if profile["tendency"] == "không rõ" else
           "Có cấu trúc, nhưng Hopkins ở chiều cao thường bị thổi phồng nên chỉ dùng làm dấu hiệu."),
        f"- **Tách cụm (silhouette cosine tốt nhất của KMeans = "
        f"{m['kmeans_best_silhouette']:.3f} tại k={m['kmeans_best_k_silhouette']}):** "
        f"{profile['separation']}. "
        + ("Các cụm chồng lấn nhiều → kỳ vọng nhiều bài nằm giữa hai chủ đề."
           if profile["separation"] in ("yếu", "rất yếu") else "Các cụm khá tách nhau."),
        f"- **Mật độ (k-distance k={m['kdist_k']}):** p95/p50 = {m['kdist_ratio_p95_p50']:.2f}, "
        f"{100 * m['outlier_fraction']:.1f}% điểm lẻ → "
        + ("mật độ KHÔNG đều, có điểm thưa/lẻ → hợp HDBSCAN/OPTICS (chịu được nhiễu)."
           if profile["uneven_density"] else
           "mật độ tương đối đều, ít điểm lẻ → các thuật toán dựa trên tâm/phân cấp hợp lý."),
        f"- **Khoảng cách cosine giữa cặp bài:** trung bình {m['cosdist_mean']:.2f}, độ lệch chuẩn "
        f"{m['cosdist_std']:.2f}" + (" (khoảng cách tập trung hẹp: hiện tượng 'lời nguyền chiều cao', "
                                     "khoảng cách kém phân biệt)." if m["cosdist_cv"] < 0.15 else "."),
        f"- **Kích thước cụm:** KMeans tại k tốt nhất có cụm nhỏ/lớn = "
        f"{m['kmeans_size_balance_at_best_k']:.2f} → "
        + ("kích thước tương đối đều, hợp KMeans/Ward/GMM." if profile["balanced_sizes"] else
           "kích thước lệch nhiều → KMeans/Ward dễ chia sai; cân nhắc GMM, HDBSCAN."),
        f"- **BIC của GMM (diag) nhỏ nhất tại k={m['gmm_best_k_bic']}**, silhouette cao nhất tại "
        f"k={m['kmeans_best_k_silhouette']}: hai gợi ý số cụm độc lập (nếu lệch nhau thì số cụm "
        "'đúng' không rõ ràng).",
    ]
    return lines


def joint_tsne(X_news: np.ndarray, X_neigh: np.ndarray, *, seed: int, perplexity: float,
               max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One t-SNE over News + neighbor samples, for drawing only (clustering never uses it).

    Returns ``(coords_news, news_rows, coords_neigh, neigh_rows)``; each side is capped at
    ``max_points / 2`` rows (seeded) to keep Colab time bounded.
    """
    rng = np.random.default_rng(seed)
    cap = max(2, max_points // 2)

    def pick(n: int) -> np.ndarray:
        return np.arange(n) if n <= cap else np.sort(rng.choice(n, cap, replace=False))

    a, b = pick(len(X_news)), pick(len(X_neigh))
    coords, _ = project_tsne(np.vstack([X_news[a], X_neigh[b]]), seed=seed, perplexity=perplexity)
    return coords[: len(a)], a, coords[len(a):], b
