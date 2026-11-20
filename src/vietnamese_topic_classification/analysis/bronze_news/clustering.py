"""Algorithm registry, k sweep, metrics, stability and agreement for the LSA space."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
    estimate_bandwidth,
)
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from .config import BronzeNewsConfig

NOISE = -1


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    family: str
    needs_k: bool
    max_points: int | None  # slow algorithms are fit on a seeded subsample of this size
    seeded: bool            # result depends on the random seed
    build: Callable[[dict[str, Any], int], Any]
    note: str               # what the scikit-learn comparison table says about it


def _ward(p, seed):
    return AgglomerativeClustering(n_clusters=p["k"], linkage="ward")


def _agg(linkage):
    return lambda p, seed: AgglomerativeClustering(
        n_clusters=p["k"], linkage=linkage, metric="cosine")


REGISTRY: dict[str, AlgoSpec] = {s.name: s for s in [
    AlgoSpec("kmeans", "tâm cụm", True, None, True,
             lambda p, s: KMeans(n_clusters=p["k"], n_init=10, random_state=s),
             "cụm dạng cầu, kích thước đều; cần k"),
    AlgoSpec("minibatch_kmeans", "tâm cụm", True, None, True,
             lambda p, s: MiniBatchKMeans(n_clusters=p["k"], n_init=5, batch_size=1024,
                                          random_state=s),
             "như KMeans nhưng nhanh hơn cho dữ liệu rất lớn"),
    AlgoSpec("gmm", "phân phối", True, None, True,
             lambda p, s: GaussianMixture(n_components=p["k"], covariance_type="diag",
                                          reg_covar=1e-4, max_iter=100, random_state=s),
             "cụm elip, cho phép kích thước khác nhau; 'diag' vì full covariance quá nhiều tham số "
             "ở chiều 100"),
    AlgoSpec("birch", "tâm cụm", True, None, False,
             lambda p, s: Birch(n_clusters=p["k"], threshold=0.5),
             "dành cho dữ liệu lớn; ở n nhỏ chỉ để đối chứng"),
    AlgoSpec("ward", "phân cấp", True, None, False, _ward,
             "cụm đều, khoảng cách Euclid (≈ cosine sau L2)"),
    AlgoSpec("agglomerative_average_cosine", "phân cấp", True, None, False, _agg("average"),
             "linkage average + cosine: nhạy với điểm lẻ"),
    AlgoSpec("agglomerative_complete_cosine", "phân cấp", True, None, False, _agg("complete"),
             "linkage complete + cosine: cụm gọn, hay tạo cụm đều"),
    AlgoSpec("hdbscan", "mật độ", False, None, False,
             lambda p, s: HDBSCAN(min_cluster_size=p["min_cluster_size"]),
             "mật độ không đều, tự phát hiện điểm nhiễu; không cần k"),
    AlgoSpec("dbscan", "mật độ", False, None, False,
             lambda p, s: DBSCAN(eps=p["eps"], min_samples=p["min_samples"]),
             "mật độ đều; eps lấy từ k-distance"),
    AlgoSpec("optics", "mật độ", False, 1500, False,
             lambda p, s: OPTICS(min_samples=p["min_samples"], xi=p["xi"],
                                 min_cluster_size=p["min_cluster_size"]),
             "như DBSCAN nhưng nhiều mật độ; chậm → chạy trên mẫu"),
    AlgoSpec("spectral", "đồ thị", True, 2000, True,
             lambda p, s: SpectralClustering(n_clusters=p["k"], affinity="nearest_neighbors",
                                             n_neighbors=15, assign_labels="kmeans",
                                             random_state=s),
             "ít cụm, hình dạng không phẳng; chậm → chạy trên mẫu"),
    AlgoSpec("affinity_propagation", "đồ thị", False, 1000, True,
             lambda p, s: AffinityPropagation(damping=p["damping"], max_iter=200,
                                              convergence_iter=15, random_state=s),
             "không cần k, hợp nhiều cụm lệch; O(n²) → chạy trên mẫu nhỏ"),
    AlgoSpec("meanshift", "mật độ", False, 600, False,
             lambda p, s: MeanShift(bandwidth=p["bandwidth"], bin_seeding=True, max_iter=50),
             "ước lượng mật độ kernel: kém ở chiều cao; chậm → chạy trên mẫu nhỏ"),
]}


def plan_algorithms(cfg: BronzeNewsConfig, profile: dict[str, Any], n: int,
                    dim: int) -> list[dict[str, Any]]:
    """Decide, from the data's shape, which algorithms run. Every decision has a reason."""
    plan = []
    for name, spec in REGISTRY.items():
        if name not in cfg.algorithms:
            plan.append({"algorithm": name, "run": False, "reason": "tắt trong cấu hình ALGORITHMS"})
            continue
        run, reason = True, spec.note
        if name == "meanshift" and dim >= 30:
            run = False
            reason = (f"BỎ QUA: dữ liệu {dim} chiều; ước lượng mật độ kernel (bandwidth) mất ý nghĩa "
                      "ở chiều cao, thường ra 1 cụm hoặc mỗi điểm 1 cụm.")
        elif name in ("hdbscan", "dbscan", "optics"):
            reason += ("; ƯU TIÊN vì mật độ không đều / nhiều điểm lẻ" if profile["uneven_density"]
                       else "; mật độ khá đều nên chạy để đối chứng")
        elif name == "gmm" and not profile["balanced_sizes"]:
            reason += "; ƯU TIÊN vì cụm lệch kích thước"
        elif name in ("kmeans", "ward", "minibatch_kmeans") and profile["balanced_sizes"]:
            reason += "; ƯU TIÊN vì cụm tương đối đều"
        if run and spec.max_points and n > min(spec.max_points, cfg.slow_max_points):
            cap = min(spec.max_points, cfg.slow_max_points)
            reason += f" (chạy trên mẫu {cap} bài, gán phần còn lại theo láng giềng gần nhất)"
        plan.append({"algorithm": name, "run": run, "reason": reason})
    return plan


def _subsample(n: int, spec: AlgoSpec, cfg: BronzeNewsConfig) -> np.ndarray:
    cap = min(spec.max_points, cfg.slow_max_points) if spec.max_points else n
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(cfg.seed).choice(n, cap, replace=False))


def param_grid(name: str, cfg: BronzeNewsConfig, shape: dict[str, Any],
               X_fit: np.ndarray) -> list[dict[str, Any]]:
    n_fit = len(X_fit)
    k0, k1 = cfg.k_range
    spec = REGISTRY[name]
    if spec.needs_k:
        return [{"k": k} for k in range(k0, min(k1, n_fit - 1) + 1)]
    if name == "hdbscan":
        sizes = sorted({max(5, n_fit // 200), max(8, n_fit // 100), max(12, n_fit // 50)})
        return [{"min_cluster_size": m} for m in sizes]
    if name == "dbscan":
        eps = shape["kdist_knee_eps"]
        ms = max(3, shape["kdist_k"])
        return [{"eps": round(eps * f, 4), "min_samples": ms} for f in (0.9, 1.0, 1.1)]
    if name == "optics":
        return [{"min_samples": max(3, shape["kdist_k"]), "xi": xi,
                 "min_cluster_size": max(5, n_fit // 100)} for xi in (0.05, 0.02)]
    if name == "affinity_propagation":
        return [{"damping": 0.8}]
    if name == "meanshift":
        bw = estimate_bandwidth(X_fit, quantile=0.2, n_samples=min(300, n_fit),
                                random_state=cfg.seed)
        return [{"bandwidth": round(float(bw), 4) if bw > 0 else None}]
    raise KeyError(name)


def cluster_metrics(X: np.ndarray, labels: np.ndarray, seed: int = 42) -> dict[str, Any]:
    """Quality metrics ignoring noise points; NaN when fewer than 2 clusters."""
    labels = np.asarray(labels)
    mask = labels != NOISE
    ids = np.unique(labels[mask])
    sizes = [int((labels == c).sum()) for c in ids]
    out: dict[str, Any] = {
        "n_clusters": int(len(ids)),
        "noise_frac": float(1 - mask.mean()),
        "largest_frac": float(max(sizes) / len(labels)) if sizes else float("nan"),
        "smallest_frac": float(min(sizes) / len(labels)) if sizes else float("nan"),
        "silhouette_cosine": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }
    if len(ids) >= 2 and mask.sum() > len(ids):
        Xm, lm = X[mask], labels[mask]
        out["silhouette_cosine"] = float(silhouette_score(
            Xm, lm, metric="cosine", sample_size=min(len(Xm), 3000), random_state=seed))
        out["davies_bouldin"] = float(davies_bouldin_score(Xm, lm))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xm, lm))
    return out


def _fmt(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def config_id(name: str, params: dict[str, Any]) -> str:
    return f"{name}|{_fmt(params)}"


def _parse(text: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(text, str):
        return out
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, _, value = part.partition("=")
        for cast in (int, float):
            try:
                out[key] = cast(value)
                break
            except ValueError:
                continue
        else:
            out[key] = None if value == "None" else value
    return out


def _fit_labels(spec: AlgoSpec, params: dict[str, Any], X: np.ndarray, seed: int) -> np.ndarray:
    est = spec.build(params, seed)
    if isinstance(est, GaussianMixture):
        return est.fit(X).predict(X)
    return np.asarray(est.fit_predict(X))


def run_config(name: str, params: dict[str, Any], X: np.ndarray, cfg: BronzeNewsConfig,
               seed: int | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit one configuration; never raises. Slow algorithms use a seeded subsample."""
    seed = cfg.seed if seed is None else seed
    spec = REGISTRY[name]
    n = len(X)
    sub = _subsample(n, spec, cfg)
    Xf = X[sub]
    row: dict[str, Any] = {
        "algorithm": name, "family": spec.family, "params": _fmt(params),
        "n_fit": int(len(sub)), "k": params.get("k"),
    }
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels_fit = _fit_labels(spec, params, Xf, seed)
        row["seconds"] = round(time.perf_counter() - started, 3)
        if name == "affinity_propagation" and (labels_fit < 0).all():
            row.update(status="không hội tụ",
                       note="AffinityPropagation không hội tụ trong giới hạn vòng lặp")
            return None, row
        converge = [w for w in caught if "onverge" in str(w.message)]
        if converge:
            row["note"] = "cảnh báo hội tụ: " + str(converge[0].message)[:80]
        row.update(cluster_metrics(Xf, labels_fit, cfg.seed))
        if len(sub) == n:
            labels = labels_fit
        else:
            labels = KNeighborsClassifier(n_neighbors=1).fit(Xf, labels_fit).predict(X)
        row["status"] = "ok"
        return labels, row
    except Exception as exc:  # estimator errors are heterogeneous; report, do not abort the run
        row.update(seconds=round(time.perf_counter() - started, 3), status="lỗi",
                   note=f"{type(exc).__name__}: {str(exc)[:100]}")
        return None, row


def run_comparison(X: np.ndarray, cfg: BronzeNewsConfig, shape: dict[str, Any],
                   profile: dict[str, Any], *, log: Callable[[str], None] = print
                   ) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Run every planned algorithm/config. Returns (table, labels by config id, plan)."""
    n, dim = X.shape
    plan = plan_algorithms(cfg, profile, n, dim)
    rows: list[dict[str, Any]] = []
    labels: dict[str, np.ndarray] = {}
    slow_elapsed = 0.0
    for item in plan:
        name = item["algorithm"]
        spec = REGISTRY[name]
        if not item["run"]:
            log(f"⏭  {name}: {item['reason']}")
            rows.append({"config_id": name, "algorithm": name, "family": spec.family,
                         "status": "bỏ qua", "note": item["reason"]})
            continue
        grid = param_grid(name, cfg, shape, X[_subsample(n, spec, cfg)])
        t_algo = time.perf_counter()
        for params in grid:
            cid = config_id(name, params)
            if spec.max_points and slow_elapsed > cfg.time_budget_seconds:
                rows.append({"config_id": cid, "algorithm": name, "family": spec.family,
                             "params": _fmt(params), "status": "bỏ qua",
                             "note": "hết ngân sách thời gian cho thuật toán chậm"})
                continue
            lab, row = run_config(name, params, X, cfg)
            row["config_id"] = cid
            rows.append(row)
            if lab is not None:
                labels[cid] = lab
            if spec.max_points:
                slow_elapsed += row.get("seconds", 0.0)
        log(f"✔ {name}: {len(grid)} cấu hình, {time.perf_counter() - t_algo:.1f}s")
    table = pd.DataFrame(rows)
    first = ["config_id", "algorithm", "family", "params", "k", "n_fit", "n_clusters", "noise_frac",
             "silhouette_cosine", "davies_bouldin", "calinski_harabasz", "largest_frac",
             "smallest_frac", "seconds", "status", "note"]
    table = table.reindex(columns=first + [c for c in table.columns if c not in first])
    return table, labels, plan


def rank_configs(table: pd.DataFrame, cfg: BronzeNewsConfig) -> pd.DataFrame:
    """Composite rank (0..1, higher is better) of silhouette, Davies-Bouldin and CH.

    Only 'ok' rows with >= 2 clusters are ranked; noise-heavy rows are kept but flagged.
    """
    out = table.copy()
    out["rank_score"] = np.nan
    ok = out[(out["status"] == "ok") & (out["n_clusters"] >= 2)]
    if len(ok):
        ranks = pd.concat([
            ok["silhouette_cosine"].rank(pct=True),
            (-ok["davies_bouldin"]).rank(pct=True),
            ok["calinski_harabasz"].rank(pct=True),
        ], axis=1)
        out.loc[ok.index, "rank_score"] = ranks.mean(axis=1)
    out["eligible_main"] = (
        out["rank_score"].notna()
        & (out["n_clusters"] >= cfg.main_min_clusters)
        & (out["n_clusters"] <= max(cfg.k_range[1], cfg.main_min_clusters))
        & (out["noise_frac"] <= cfg.main_max_noise)
        & (out["smallest_frac"] >= cfg.main_min_cluster_frac)
    )
    return out.sort_values("rank_score", ascending=False, na_position="last").reset_index(drop=True)


def seed_stability(name: str, params: dict[str, Any], X: np.ndarray,
                   cfg: BronzeNewsConfig) -> dict[str, Any]:
    """Mean pairwise ARI of the same configuration over several seeds (1.0 if deterministic)."""
    if not REGISTRY[name].seeded:
        return {"stability_ari": 1.0, "stability_note": "xác định (không phụ thuộc seed)"}
    runs = []
    for i in range(cfg.stability_seeds):
        lab, _ = run_config(name, params, X, cfg, seed=cfg.seed + 1 + i)
        if lab is not None:
            runs.append(lab)
    pairs = [adjusted_rand_score(runs[i], runs[j])
             for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return {"stability_ari": float(np.mean(pairs)) if pairs else float("nan"),
            "stability_note": f"ARI trung bình giữa {len(runs)} seed"}


def kmeans_stability_curve(X: np.ndarray, cfg: BronzeNewsConfig) -> pd.DataFrame:
    """KMeans (n_init=1, so the seed matters) ARI across seeds for each k."""
    rows = []
    for k in range(cfg.k_range[0], min(cfg.k_range[1], len(X) - 1) + 1):
        runs = [KMeans(n_clusters=k, n_init=1, random_state=cfg.seed + i).fit_predict(X)
                for i in range(cfg.stability_seeds)]
        pairs = [adjusted_rand_score(runs[i], runs[j])
                 for i in range(len(runs)) for j in range(i + 1, len(runs))]
        rows.append({"k": k, "ari_mean": float(np.mean(pairs)), "ari_min": float(np.min(pairs))})
    return pd.DataFrame(rows)


def choose_main(ranked: pd.DataFrame, X: np.ndarray, cfg: BronzeNewsConfig
                ) -> list[dict[str, Any]]:
    """Pick up to ``n_main`` configs: eligible, high rank, stable, from different algorithms."""
    picks: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in ranked[ranked["eligible_main"]].itertuples():
        if row.algorithm in used:
            continue
        stability = seed_stability(row.algorithm, _parse(row.params), X, cfg)
        ari = stability["stability_ari"]
        if ari == ari and ari < cfg.main_min_stability:  # NaN = not measurable -> passes
            continue
        picks.append({
            "config_id": row.config_id, "algorithm": row.algorithm, "params": row.params,
            "rank_score": float(row.rank_score), "n_clusters": int(row.n_clusters),
            "silhouette_cosine": float(row.silhouette_cosine), **stability,
            "criteria": (
                f"{cfg.main_min_clusters}-{max(cfg.k_range[1], cfg.main_min_clusters)} cụm, "
                f"nhiễu ≤ {cfg.main_max_noise:.0%}, cụm nhỏ nhất ≥ {cfg.main_min_cluster_frac:.0%}, "
                "xếp hạng tổng hợp (silhouette, Davies-Bouldin, Calinski-Harabasz) cao nhất "
                f"trong thuật toán chưa được chọn, ổn định seed ≥ {cfg.main_min_stability}"),
        })
        used.add(row.algorithm)
        if len(picks) >= cfg.n_main:
            break
    if not picks:
        usable = ranked[ranked["rank_score"].notna()]
        if usable.empty:
            raise ValueError("Không có cấu hình phân cụm nào hợp lệ (>= 2 cụm).")
        row = next(usable.itertuples())
        picks.append({
            "config_id": row.config_id, "algorithm": row.algorithm, "params": row.params,
            "rank_score": float(row.rank_score), "n_clusters": int(row.n_clusters),
            "silhouette_cosine": float(row.silhouette_cosine),
            **seed_stability(row.algorithm, _parse(row.params), X, cfg),
            "criteria": "KHÔNG cấu hình nào đạt đủ tiêu chí; lấy cấu hình xếp hạng tổng hợp cao nhất "
                        "(cần xem xét kỹ)",
        })
    return picks


def agreement(labels: dict[str, np.ndarray], ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise ARI and NMI between chosen configs (noise counts as its own group)."""
    ari = pd.DataFrame(index=ids, columns=ids, dtype=float)
    nmi = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for a in ids:
        for b in ids:
            ari.loc[a, b] = adjusted_rand_score(labels[a], labels[b])
            nmi.loc[a, b] = normalized_mutual_info_score(labels[a], labels[b])
    return ari, nmi
