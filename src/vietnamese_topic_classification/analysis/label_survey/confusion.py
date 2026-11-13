"""Out-of-fold TF-IDF + Logistic Regression evidence of label confusion (analysis model).

The model is a *measuring instrument*, not a product: every document is predicted by a
model that never saw it, so confusion reflects how similar label contents are.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from vietnamese_topic_classification.evaluation.metrics import calculate_metrics
from vietnamese_topic_classification.training.config import (
    ClassifierConfig,
    ExperimentConfig,
    FeatureConfig,
)
from vietnamese_topic_classification.training.factory import build_pipeline

from .config import SurveyConfig
from .plots import COLORS, apply_style, save_figure


def order_labels(present: Sequence[str], canonical: Sequence[str]) -> list[str]:
    """Canonical (domains_v1) order first, then unexpected labels alphabetically."""
    known = [x for x in canonical if x in set(present)]
    return known + sorted(set(present) - set(canonical))


def select_analysis_subset(
    sample: pd.DataFrame, canonical: Sequence[str], cfg: SurveyConfig
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """At most ``per_class_cap`` random documents per label (seeded).

    Capping stops big labels from dominating; confusion then mirrors content similarity
    rather than label size. Labels too small for stratified CV are reported and skipped.
    """
    valid = sample[sample["domain"].notna() & (sample["domain"].str.strip() != "")]
    counts = valid["domain"].value_counts()
    minimum = 2 * cfg.n_folds
    skipped = {str(k): int(v) for k, v in counts.items() if v < minimum}
    classes = order_labels([k for k, v in counts.items() if v >= minimum], canonical)
    rng = np.random.default_rng(cfg.seed)
    parts = []
    for label in classes:
        part = valid[valid["domain"] == label]
        if len(part) > cfg.per_class_cap:
            part = part.iloc[np.sort(rng.choice(len(part), cfg.per_class_cap, replace=False))]
        parts.append(part)
    subset = pd.concat(parts, ignore_index=True)
    return subset, classes, skipped


def build_estimators(cfg: SurveyConfig) -> tuple[Any, Any]:
    """Vectorizer and classifier taken from the repo baseline factory (no re-implementation)."""
    experiment = ExperimentConfig(
        name="label-survey",
        gold_manifest_name="not_used.json",
        features=FeatureConfig(
            params={"ngram_range": (1, 2), "min_df": 2, "max_features": cfg.max_features,
                    "dtype": np.float32}
        ),
        classifier=ClassifierConfig(
            kind="logistic_regression", params={"max_iter": cfg.lr_max_iter}
        ),
        seed=cfg.seed,
    )
    pipeline = build_pipeline(experiment)
    return pipeline.named_steps["tfidf"], pipeline.named_steps["classifier"]


def vectorize(texts: Sequence[str], cfg: SurveyConfig, char_limit: int) -> tuple[Any, sparse.csr_matrix]:
    """Fit TF-IDF once on all analysis texts (labels are not used, see limitations)."""
    vectorizer, _ = build_estimators(cfg)
    clipped = [t[:char_limit] for t in texts]
    return vectorizer, vectorizer.fit_transform(clipped).tocsr()


def _fit_fold(classifier: Any, X: sparse.csr_matrix, y: np.ndarray, train: np.ndarray,
              test: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    model = clone(classifier).fit(X[train], y[train])
    local = model.predict_proba(X[test])
    proba = np.zeros((len(test), n_classes), dtype=np.float32)
    proba[:, model.classes_] = local
    return test, proba, time.perf_counter() - started


def oof_probabilities(X: sparse.csr_matrix, y: np.ndarray, n_classes: int, cfg: SurveyConfig,
                      *, label: str = "") -> np.ndarray:
    """Out-of-fold class probabilities, shape ``(n, n_classes)``."""
    from joblib import Parallel, delayed

    _, classifier = build_estimators(cfg)
    folds = list(StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(X, y))
    started = time.perf_counter()
    jobs = max(1, min(cfg.n_jobs, cfg.n_folds))
    results = Parallel(n_jobs=jobs)(
        delayed(_fit_fold)(classifier, X, y, tr, te, n_classes) for tr, te in folds
    )
    proba = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    for test, local, seconds in results:
        proba[test] = local
        print(f"  {label} fold: {seconds:,.0f}s", flush=True)
    print(f"  {label} OOF xong: {time.perf_counter() - started:,.0f}s", flush=True)
    return proba


def oof_frame(ids: Sequence[str], y: np.ndarray, proba: np.ndarray,
              classes: Sequence[str]) -> pd.DataFrame:
    """Per-document: given label, top-1/top-2 and their probabilities."""
    order = np.argsort(-proba, axis=1)[:, :2]
    rows = np.arange(len(y))
    names = np.array(classes, dtype=object)
    return pd.DataFrame(
        {
            "id": list(ids),
            "given_idx": y,
            "given": names[y],
            "given_prob": proba[rows, y],
            "top1_idx": order[:, 0],
            "top1": names[order[:, 0]],
            "top1_prob": proba[rows, order[:, 0]],
            "top2_idx": order[:, 1],
            "top2": names[order[:, 1]],
            "top2_prob": proba[rows, order[:, 1]],
        }
    )


def confusion_counts(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (y, pred), 1)
    return matrix


def row_normalize(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros(counts.shape, float), where=totals > 0)


def symmetric_confusion(normalized: np.ndarray) -> np.ndarray:
    """``(C[a,b] + C[b,a]) / 2`` of a row-normalized matrix; diagonal set to 0."""
    sym = (normalized + normalized.T) / 2
    np.fill_diagonal(sym, 0.0)
    return sym


def centroid_cosine(X: sparse.csr_matrix, y: np.ndarray, k: int) -> np.ndarray:
    """Cosine between the mean TF-IDF vectors of every pair of labels."""
    onehot = sparse.csr_matrix(
        (np.ones(len(y)), (y, np.arange(len(y)))), shape=(k, len(y)), dtype=np.float64
    )
    sums = np.asarray((onehot @ X).todense())
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    unit = np.divide(sums, norms, out=np.zeros_like(sums), where=norms > 0)
    return unit @ unit.T


def top2_ambiguity(oof: pd.DataFrame, k: int, margin: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Count documents whose top-2 set is {a,b}, given label is a or b, and p1-p2 < margin."""
    given, t1, t2 = (oof[c].to_numpy() for c in ("given_idx", "top1_idx", "top2_idx"))
    close = (oof["top1_prob"].to_numpy() - oof["top2_prob"].to_numpy()) < margin
    mask = close & ((given == t1) | (given == t2))
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (np.minimum(t1[mask], t2[mask]), np.maximum(t1[mask], t2[mask])), 1)
    return counts, np.bincount(given, minlength=k)


def pair_table(counts: np.ndarray, oof: pd.DataFrame, cosine: np.ndarray,
               classes: Sequence[str], margin: float = 0.2) -> pd.DataFrame:
    """All label pairs ranked by symmetric confusion (several independent signals)."""
    k = len(classes)
    normalized = row_normalize(counts)
    sym = symmetric_confusion(normalized)
    ambiguous, support = top2_ambiguity(oof, k, margin)
    rows = []
    for a in range(k):
        for b in range(a + 1, k):
            n_pair = support[a] + support[b]
            rows.append(
                {
                    "label_a": classes[a],
                    "label_b": classes[b],
                    "sym_confusion": sym[a, b],
                    "rate_a_to_b": normalized[a, b],
                    "rate_b_to_a": normalized[b, a],
                    "n_a_to_b": int(counts[a, b]),
                    "n_b_to_a": int(counts[b, a]),
                    "top2_ambiguity": ambiguous[a, b] / n_pair if n_pair else 0.0,
                    "top2_ambiguity_n": int(ambiguous[a, b]),
                    "centroid_cosine": float(cosine[a, b]),
                }
            )
    table = pd.DataFrame(rows).sort_values("sym_confusion", ascending=False)
    table = table.reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def label_linkage(sym: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Average-linkage tree; distance = 1 - sym/max(sym) (0 diagonal)."""
    top = sym.max()
    similarity = sym / top if top > 0 else np.zeros_like(sym)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    tree = hierarchy.linkage(squareform(distance, checks=False), method="average")
    return tree, [int(i) for i in hierarchy.leaves_list(tree)]


def per_label_metrics(y: np.ndarray, pred: np.ndarray, classes: Sequence[str]
                      ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-label precision/recall/F1 and overall metrics (repo ``calculate_metrics``)."""
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0
    )
    overall = calculate_metrics(y.tolist(), pred.tolist(), label_names=list(classes))
    table = pd.DataFrame(
        {"label": list(classes), "precision": precision, "recall": recall, "f1": f1,
         "support": support}
    )
    return table, {"accuracy": overall["accuracy"], "macro_f1": overall["macro_f1"],
                   "weighted_f1": overall["weighted_f1"]}


# ----------------------------------------------------------------------- figures


def fig_confusion_matrix(normalized: np.ndarray, classes: Sequence[str], order: Sequence[int],
                         path: Path) -> Path:
    apply_style()
    names = [classes[i] for i in order]
    matrix = normalized[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.3, float(matrix.max())))
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set(xlabel="Nhãn mô hình đoán (out-of-fold)", ylabel="Nhãn gốc (do máy gán)",
           title="Ma trận lẫn out-of-fold, chuẩn hóa theo hàng (xếp theo cụm trong dendrogram)")
    fig.colorbar(image, ax=ax, fraction=0.04, label="Tỉ lệ")
    return save_figure(fig, path)


def fig_top_confusing_pairs(pairs: pd.DataFrame, top_n: int, path: Path) -> Path:
    apply_style()
    data = pairs.head(top_n).iloc[::-1]
    names = [f"{a} ↔ {b}" for a, b in zip(data["label_a"], data["label_b"])]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(data))))
    bars = ax.barh(names, data["sym_confusion"], color=COLORS["red"])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in data["sym_confusion"]], padding=3, fontsize=8)
    ax.set(xlabel="sym_confusion = (C[a,b] + C[b,a]) / 2",
           title=f"{top_n} cặp nhãn dễ lẫn nhất")
    ax.set_xlim(0, float(data["sym_confusion"].max()) * 1.15 + 1e-9)
    return save_figure(fig, path)


def fig_dendrogram(tree: np.ndarray, classes: Sequence[str], path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(classes))))
    hierarchy.dendrogram(tree, labels=list(classes), orientation="right", ax=ax,
                         color_threshold=0, above_threshold_color=COLORS["blue"])
    ax.set(xlabel="Khoảng cách = 1 − sym_confusion chuẩn hóa",
           title="Cây gom nhóm nhãn (average linkage)")
    return save_figure(fig, path)


def fig_per_label_f1(metrics: pd.DataFrame, path: Path) -> Path:
    apply_style()
    data = metrics.sort_values("f1")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.34 * len(data))))
    bars = ax.barh(data["label"], data["f1"], color=COLORS["teal"])
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in data["f1"]], padding=3, fontsize=8)
    ax.set(xlim=(0, 1.05), xlabel="F1 out-of-fold", title="F1 theo nhãn (thấp nhất ở trên cùng)")
    ax.invert_yaxis()
    return save_figure(fig, path)


def fig_greedy_merge_curve(curve: pd.DataFrame, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve["step"], curve["macro_f1"], "o-", color=COLORS["blue"])
    for step, f1, label in zip(curve["step"], curve["macro_f1"], curve["merged"]):
        if step > 0:
            ax.annotate(label, (step, f1), textcoords="offset points", xytext=(4, -12),
                        fontsize=7, rotation=0)
    ax.set(xlabel="Số bước gộp tham lam", ylabel="Macro-F1 (ước lượng từ dự đoán cũ)",
           title="Macro-F1 theo từng bước gộp (gần như luôn tăng; tăng ≠ nên gộp)")
    ax.grid(alpha=0.3)
    return save_figure(fig, path)
