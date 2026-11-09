"""Signs of wrong labels: confident out-of-fold disagreement and conflicting duplicates.

Everything here is a *suspicion*: the analysis model can be wrong too.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import pandas as pd


def suspect_mask(oof: pd.DataFrame, threshold: float) -> pd.Series:
    """Model's top-1 differs from the given label with probability >= threshold."""
    return (oof["top1_idx"] != oof["given_idx"]) & (oof["top1_prob"] >= threshold)


def mismatch_table(oof: pd.DataFrame, classes: Sequence[str], threshold: float) -> pd.DataFrame:
    """Per label: confident-mismatch rate and the label it is most often read as."""
    mask = suspect_mask(oof, threshold)
    rows = []
    for idx, label in enumerate(classes):
        part = oof["given_idx"] == idx
        n = int(part.sum())
        flagged = oof[part & mask]
        if len(flagged):
            counts = flagged["top1"].value_counts()
            top_label, top_n = str(counts.index[0]), int(counts.iloc[0])
        else:
            top_label, top_n = "", 0
        rows.append(
            {
                "label": label,
                "n": n,
                "n_confident_mismatch": len(flagged),
                "confident_mismatch_rate": len(flagged) / n if n else 0.0,
                "most_often_read_as": top_label,
                "most_often_read_as_share": top_n / len(flagged) if len(flagged) else 0.0,
                "mean_prob_of_given_label": float(oof.loc[part, "given_prob"].mean()) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("confident_mismatch_rate", ascending=False).reset_index(
        drop=True
    )


def suspected_examples(oof: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Suspected mislabels, most confident first (ids and labels only, no text)."""
    flagged = oof[suspect_mask(oof, threshold)]
    return flagged.sort_values("top1_prob", ascending=False)[
        ["id", "given", "top1", "top1_prob", "given_prob", "top2", "top2_prob"]
    ].reset_index(drop=True)


def conflicting_duplicates(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Identical normalized texts that carry different labels (ids and labels only)."""
    labelled = sample[sample["domain"].notna() & (sample["domain"].str.strip() != "")]
    grouped = labelled.groupby("text_sha")["domain"]
    sizes, distinct = grouped.transform("size"), grouped.transform("nunique")
    duplicated = labelled[sizes > 1]
    conflicting = labelled[(sizes > 1) & (distinct > 1)]
    group_number = pd.factorize(conflicting["text_sha"])[0] + 1
    table = pd.DataFrame(
        {
            "group": group_number,
            "id": conflicting["id"].to_numpy(),
            "domain": conflicting["domain"].to_numpy(),
        }
    ).sort_values(["group", "domain"]).reset_index(drop=True)
    pair_counter: Counter = Counter()
    for _, labels in table.groupby("group")["domain"]:
        pair_counter[" | ".join(sorted(set(labels)))] += 1
    summary = {
        "rows_in_exact_duplicate_groups": int(len(duplicated)),
        "exact_duplicate_groups": int(duplicated["text_sha"].nunique()),
        "conflicting_groups": int(table["group"].nunique()) if len(table) else 0,
        "rows_in_conflicting_groups": int(len(table)),
        "most_common_conflicts": pair_counter.most_common(10),
    }
    return table, summary
