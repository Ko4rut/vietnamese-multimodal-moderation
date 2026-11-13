"""What-if label scenarios computed from stored out-of-fold predictions (no retraining).

Scenarios are *reference numbers*, not recommendations. Merging labels makes the task
easier, so scores almost always rise; dropping a label means the model can never
recognise that kind of text. Recomputing from old predictions is only a quick estimate.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .confusion import confusion_counts, row_normalize, symmetric_confusion


def _scores(y: np.ndarray, pred: np.ndarray, names: Sequence[str]) -> dict[str, object]:
    labels = np.unique(y)
    f1 = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    worst = int(np.argmin(f1))
    return {
        "n_labels": int(len(labels)),
        "macro_f1": float(f1.mean()),
        "min_f1": float(f1[worst]),
        "min_f1_label": names[int(labels[worst])],
        "accuracy": float(accuracy_score(y, pred)),
    }


def _partition(k: int, merged: Sequence[Sequence[int]]) -> list[list[int]]:
    """Partition of ``range(k)``: the given groups plus every other label alone."""
    used = {m for g in merged for m in g}
    return [sorted(g) for g in merged] + [[i] for i in range(k) if i not in used]


def _relabel(y: np.ndarray, pred: np.ndarray, parts: Sequence[Sequence[int]],
             classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map original label ids to part ids; a part's name joins its members with '+'."""
    table = np.zeros(len(classes), dtype=np.int64)
    for part_id, members in enumerate(parts):
        for m in members:
            table[m] = part_id
    names = ["+".join(classes[m] for m in members) for members in parts]
    return table[y], table[pred], names


def _pct(weights: Mapping[str, float], labels: Sequence[str]) -> float:
    return float(sum(weights.get(x, 0.0) for x in labels))


def run_scenarios(
    oof: pd.DataFrame,
    classes: Sequence[str],
    pairs: pd.DataFrame,
    *,
    full_pct: Mapping[str, float],
    top_pairs: int,
    greedy_steps: int = 8,
    lowest_f1_to_drop: int = 5,
    extra_drop: Sequence[str] = ("News", "Sensitive_Subjects"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(scenario table, greedy merge curve)``."""
    y = oof["given_idx"].to_numpy()
    pred = oof["top1_idx"].to_numpy()
    pred2 = oof["top2_idx"].to_numpy()
    k = len(classes)
    share = {label: float((y == i).mean()) for i, label in enumerate(classes)}
    base = _scores(y, pred, list(classes))
    rows: list[dict[str, object]] = []

    def add(scenario: str, description: str, scores: dict, **extra: object) -> None:
        rows.append({"scenario": scenario, "description": description, **scores,
                     "delta_macro_f1": scores["macro_f1"] - base["macro_f1"], **extra})

    add("S0", "Giữ nguyên toàn bộ nhãn (mốc so sánh)", base,
        affected_pct_analysis_set=0.0, pct_full_dataset_affected=0.0, merge_order="")

    # S1: merge one pair at a time -------------------------------------------------
    index = {label: i for i, label in enumerate(classes)}
    for r in pairs.head(top_pairs).itertuples():
        a, b = index[r.label_a], index[r.label_b]
        ty, tp, names = _relabel(y, pred, _partition(k, [[a, b]]), classes)
        add(f"S1: {r.label_a}+{r.label_b}", "Gộp riêng cặp này (nhãn gốc và nhãn đoán cùng gộp)",
            _scores(ty, tp, names),
            affected_pct_analysis_set=100 * (share[r.label_a] + share[r.label_b]),
            pct_full_dataset_affected=_pct(full_pct, [r.label_a, r.label_b]),
            merge_order=f"{r.label_a} + {r.label_b}")

    # S2: greedy merge, recomputing confusion after every merge -------------------
    parts = _partition(k, [])
    curve = [{"step": 0, "merged": "(không gộp)", "macro_f1": base["macro_f1"],
              "min_f1": base["min_f1"], "n_labels": base["n_labels"]}]
    order: list[str] = []
    for step in range(1, min(greedy_steps, k - 1) + 1):
        cy, cp, names = _relabel(y, pred, parts, classes)
        sym = symmetric_confusion(row_normalize(confusion_counts(cy, cp, len(parts))))
        i, j = (int(v) for v in np.unravel_index(int(np.argmax(sym)), sym.shape))
        if sym[i, j] <= 0:
            break
        order.append(f"{names[i]} + {names[j]}")
        merged_members = sorted(parts[i] + parts[j])
        parts = [p for n_, p in enumerate(parts) if n_ not in (i, j)] + [merged_members]
        ty, tp, new_names = _relabel(y, pred, parts, classes)
        scores = _scores(ty, tp, new_names)
        involved = [classes[m] for p in parts if len(p) > 1 for m in p]
        curve.append({"step": step, "merged": "+".join(classes[m] for m in merged_members),
                      "macro_f1": scores["macro_f1"], "min_f1": scores["min_f1"],
                      "n_labels": scores["n_labels"]})
        add(f"S2: bước {step}", "Gộp tham lam (tính lại ma trận lẫn sau mỗi lần gộp)", scores,
            affected_pct_analysis_set=100 * sum(share[x] for x in involved),
            pct_full_dataset_affected=_pct(full_pct, involved),
            merge_order=" → ".join(order))

    # S3: drop labels ---------------------------------------------------------------
    f1_all = f1_score(y, pred, labels=list(range(k)), average=None, zero_division=0)
    lowest = [classes[i] for i in np.argsort(f1_all)[:lowest_f1_to_drop]]
    for label in dict.fromkeys([*lowest, *[x for x in extra_drop if x in index]]):
        dropped = index[label]
        keep = y != dropped
        fixed = np.where(pred == dropped, pred2, pred)[keep]
        scores = _scores(y[keep], fixed, list(classes))
        add(f"S3: bỏ {label}", "Bỏ bài mang nhãn này; bài bị đoán thành nhãn này dùng top-2",
            scores,
            affected_pct_analysis_set=100 * share[label],
            pct_full_dataset_affected=full_pct.get(label, 0.0),
            pct_remaining_docs_repredicted=100 * float((pred[keep] == dropped).mean()),
            merge_order="")
    return pd.DataFrame(rows), pd.DataFrame(curve)
