"""Descriptive statistics and data-quality checks for the News rows (report only)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from vietnamese_topic_classification.analysis.news_v3 import build_keyword_report
from vietnamese_topic_classification.data.silver.v1.transformer import SilverTransformer

from .features import EMAIL_RE, URL_RE, to_documents

PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
PERCENTILES = (50, 90, 95, 99)


def normalize(text: str | None) -> str:
    return SilverTransformer.normalize_text(text) if isinstance(text, str) else ""


def text_digest(text: str | None) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def prepare_frame(news: pd.DataFrame) -> pd.DataFrame:
    """Add normalized text, its sha256 and an ``is_empty`` flag (the input is not changed)."""
    frame = news.copy()
    frame["id"] = frame["id"].map(lambda v: "" if v is None else str(v))
    frame["text_null"] = frame["text"].map(lambda v: not isinstance(v, str))
    frame["norm"] = frame["text"].map(normalize)
    frame["is_empty"] = frame["norm"].str.len() == 0
    frame["sha256"] = frame["norm"].map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    return frame


def exact_duplicate_mask(frame: pd.DataFrame) -> pd.Series:
    """True for the 2nd, 3rd... copy of an identical (normalized, non-empty) text."""
    dup = frame.loc[~frame["is_empty"], "sha256"].duplicated(keep="first")
    return dup.reindex(frame.index, fill_value=False)


def quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    nonempty = frame[~frame["is_empty"]]
    ids = frame["id"]
    dup_ids = ids[ids != ""].duplicated(keep=False)
    groups = nonempty.groupby("sha256").size()
    return {
        "news_articles": int(n),
        "text_null": int(frame["text_null"].sum()),
        "text_empty_or_blank": int(frame["is_empty"].sum()) - int(frame["text_null"].sum()),
        "text_unusable_total": int(frame["is_empty"].sum()),
        "id_missing": int((ids == "").sum()),
        "id_duplicated_rows": int(dup_ids.sum()),
        "id_duplicated_distinct": int(ids[ids != ""][dup_ids].nunique()),
        "exact_duplicate_groups": int((groups > 1).sum()),
        "exact_duplicate_extra_copies": int((groups - 1).clip(lower=0).sum()),
        "exact_duplicate_rate": float((groups - 1).clip(lower=0).sum() / max(1, n)),
    }


def near_duplicate_report(
    texts: Sequence[str], *, threshold: float, sample_size: int, seed: int
) -> dict[str, Any]:
    """Pairs with cosine TF-IDF >= threshold inside a bounded seeded sample (report only)."""
    n = len(texts)
    take = min(sample_size, n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=take, replace=False)) if take < n else np.arange(n)
    docs = to_documents([texts[i] for i in idx])
    keep = [i for i, d in enumerate(docs) if d]
    if len(keep) < 2:
        return {"sample_size": int(take), "pairs": 0, "documents_involved": 0,
                "threshold": threshold, "rate_in_sample": 0.0}
    X = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True).fit_transform(
        [docs[i] for i in keep]).tocsr()
    pairs, involved = 0, set()
    for start in range(0, X.shape[0], 500):
        sims = (X[start:start + 500] @ X.T).tocoo()
        mask = (sims.data >= threshold) & (sims.row + start < sims.col)
        pairs += int(mask.sum())
        involved.update((sims.row[mask] + start).tolist())
        involved.update(sims.col[mask].tolist())
    return {
        "sample_size": int(take),
        "documents_compared": int(X.shape[0]),
        "pairs": int(pairs),
        "documents_involved": len(involved),
        "threshold": threshold,
        "rate_in_sample": float(len(involved) / X.shape[0]),
    }


def length_table(norm_texts: Sequence[str]) -> pd.DataFrame:
    """Characters and whitespace-separated words (~Vietnamese syllables)."""
    return pd.DataFrame({
        "chars": [len(t) for t in norm_texts],
        "words": [len(t.split()) for t in norm_texts],
    })


def length_stats(lengths: pd.DataFrame, *, short_words: int, long_words: int) -> dict[str, Any]:
    out: dict[str, Any] = {"documents": int(len(lengths))}
    for column in ("chars", "words"):
        values = lengths[column].to_numpy()
        out[column] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            **{f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES},
            "min": int(values.min()) if len(values) else 0,
            "max": int(values.max()) if len(values) else 0,
        }
    out["short_threshold_words"] = short_words
    out["long_threshold_words"] = long_words
    out["short_documents"] = int((lengths["words"] < short_words).sum())
    out["long_documents"] = int((lengths["words"] > long_words).sum())
    return out


def noise_flags(norm_texts: Sequence[str]) -> pd.DataFrame:
    """Per-document crawl-noise indicators (booleans)."""
    rows = []
    for text in norm_texts:
        rows.append({
            "has_url": bool(URL_RE.search(text)),
            "has_email": bool(EMAIL_RE.search(text)),
            "has_phone": bool(PHONE_RE.search(text)),
            "odd_chars": _odd_char_ratio(text) > 0.01,
        })
    return pd.DataFrame(rows, columns=["has_url", "has_email", "has_phone", "odd_chars"])


def _odd_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    odd = sum(
        1 for ch in text
        if ch == "�" or unicodedata.category(ch) in {"Co", "Cn", "Cs"}
        or (unicodedata.category(ch) == "Cc" and ch not in "\n\t")
        or unicodedata.category(ch) == "Cf"
    )
    return odd / len(text)


def boilerplate_sentences(
    norm_texts: Sequence[str], *, min_docs: int | None = None, top: int = 10
) -> list[tuple[str, int]]:
    """Sentences/lines that repeat across many *different* articles (crawl boilerplate).

    The sentences are article text: show them in the notebook only, never in summaries.
    """
    n = len(norm_texts)
    threshold = min_docs if min_docs is not None else max(5, int(0.01 * n))
    counter: Counter[str] = Counter()
    for text in norm_texts:
        sentences = {s.strip().lower() for s in _SENTENCE_SPLIT.split(text)}
        counter.update(s for s in sentences if 15 <= len(s) <= 300)
    return [(s, c) for s, c in counter.most_common(top * 3) if c >= threshold][:top]


def noise_report(flags: pd.DataFrame, boilerplate: list[tuple[str, int]]) -> dict[str, Any]:
    n = max(1, len(flags))
    out: dict[str, Any] = {"documents": int(len(flags))}
    for column in flags.columns:
        out[column] = int(flags[column].sum())
        out[f"{column}_rate"] = float(flags[column].sum() / n)
    out["any_noise_flag"] = int(flags.any(axis=1).sum())
    out["boilerplate_sentences_repeated"] = len(boilerplate)
    out["boilerplate_max_documents"] = int(boilerplate[0][1]) if boilerplate else 0
    return out


def _document_frequency(docs: Sequence[str], ngram: tuple[int, int]) -> pd.Series:
    vec = CountVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=ngram, binary=True,
                          min_df=1)
    X = vec.fit_transform(docs)
    df = np.asarray(X.sum(axis=0)).ravel()
    return pd.Series(df, index=vec.get_feature_names_out()).sort_values(ascending=False)


def top_terms(texts: Sequence[str | None], *, top: int = 25) -> dict[str, pd.DataFrame]:
    """Top unigrams / bigrams by document frequency (stopwords removed)."""
    docs = [d for d in to_documents(texts) if d]
    result = {}
    for name, ngram in (("unigram", (1, 1)), ("bigram", (2, 2))):
        series = _document_frequency(docs, ngram).head(top)
        result[name] = pd.DataFrame({
            "term": series.index, "document_frequency": series.to_numpy(),
            "rate": series.to_numpy() / max(1, len(docs)),
        })
    return result


def compare_news_with_neighbors(
    news_texts: Sequence[str | None],
    neighbors: pd.DataFrame,
    *,
    top: int = 15,
    min_df: int = 3,
) -> dict[str, Any]:
    """Which terms are shared with / specific to each neighbor label (document frequency).

    Reuses ``news_v3.build_keyword_report`` for the vocabulary-overlap (Jaccard) numbers.
    These are discovery hints, not labels.
    """
    news_docs = [d for d in to_documents(news_texts) if d]
    news_df = _document_frequency(news_docs, (1, 1))
    news_rate = news_df / max(1, len(news_docs))
    out: dict[str, Any] = {"labels": {}}
    records = [{"domain": "News", "text": t} for t in news_texts if isinstance(t, str) and t.strip()]
    records += [
        {"domain": r.domain, "text": r.text}
        for r in neighbors.itertuples() if isinstance(r.text, str) and r.text.strip()
    ]
    overlap = build_keyword_report(records, top_n=top, examples_per_domain=0)[
        "news_vs_other_class_keyword_overlap"]
    for label, group in neighbors.groupby("domain"):
        docs = [d for d in to_documents(group["text"].tolist()) if d]
        if not docs:
            continue
        other_df = _document_frequency(docs, (1, 1))
        other_rate = other_df / len(docs)
        terms = news_rate.index.union(other_rate.index)
        a = news_rate.reindex(terms, fill_value=0.0)
        b = other_rate.reindex(terms, fill_value=0.0)
        support = (news_df.reindex(terms, fill_value=0) >= min_df) | (
            other_df.reindex(terms, fill_value=0) >= min_df)
        diff = (a - b)[support]
        shared = (np.minimum(a, b))[support].sort_values(ascending=False).head(top)
        out["labels"][label] = {
            "documents": len(docs),
            "shared_terms": [t for t in shared.index],
            "news_specific_terms": [t for t in diff.sort_values(ascending=False).head(top).index],
            "neighbor_specific_terms": [t for t in diff.sort_values().head(top).index],
            "vocabulary_jaccard": float(overlap.get(label, {}).get("shared_term_jaccard", np.nan)),
        }
    return out


def drop_summary(frame: pd.DataFrame, drop_exact: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rows used for clustering: never empty text; exact duplicates only if requested."""
    mask_empty = frame["is_empty"]
    mask_dup = exact_duplicate_mask(frame) if drop_exact else pd.Series(False, index=frame.index)
    keep = ~(mask_empty | mask_dup)
    info = {
        "input": int(len(frame)),
        "dropped_empty": int(mask_empty.sum()),
        "dropped_exact_duplicates": int((mask_dup & ~mask_empty).sum()),
        "kept": int(keep.sum()),
    }
    return frame[keep].reset_index(drop=True), info
