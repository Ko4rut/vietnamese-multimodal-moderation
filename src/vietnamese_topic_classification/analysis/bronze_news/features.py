"""Text cleaning, syllable tokens, TF-IDF and LSA (TruncatedSVD + L2 normalization)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from vietnamese_topic_classification.analysis.news_v3 import _STOPWORDS as _BASE_STOPWORDS
from vietnamese_topic_classification.analysis.news_v3 import _TOKEN
from vietnamese_topic_classification.data.silver.v1.transformer import SilverTransformer

from .config import BronzeNewsConfig

# Extends news_v3._STOPWORDS with common Vietnamese function words and news boilerplate.
STOPWORDS = frozenset(_BASE_STOPWORDS) | frozenset(
    "cũng để nên vì bởi nếu mà thế đến ra lên xuống ở do qua bị tới lại đây đấy nào ai gì "
    "đều vẫn chỉ cả mỗi hay hoặc tuy song mới từng phải thể việc cách lần nhất hết luôn "
    "khác cùng giữa ngoài trên dưới bên thành làm nói biết cho ông bà anh chị em họ tôi "
    "chúng mình kia ấy đó thêm nữa lúc vậy thật đúng cần muốn thấy ra vừa còn đang "
    "bằng nhưng tuy nhiên tức đồng hơn".split()
)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def clean_text(text: str | None) -> str:
    """NFC + whitespace normalization (SilverTransformer), lowercase, drop URLs/emails."""
    if not text:
        return ""
    normalized = SilverTransformer.normalize_text(text).lower()
    return EMAIL_RE.sub(" ", URL_RE.sub(" ", normalized))


def tokenize(text: str | None) -> list[str]:
    """Vietnamese syllables (letters only, so numbers/punctuation vanish), no stopwords."""
    return [t for t in _TOKEN.findall(clean_text(text)) if t not in STOPWORDS]


def to_documents(texts: Sequence[str | None], char_limit: int | None = None) -> list[str]:
    """Space-joined syllable strings: the exact input of the TF-IDF vectorizer."""
    docs = []
    for text in texts:
        if text and char_limit:
            text = text[:char_limit]
        docs.append(" ".join(tokenize(text)))
    return docs


@dataclass
class FeatureSpace:
    """Fitted TF-IDF + LSA transform (fit on News only)."""

    vectorizer: TfidfVectorizer = field(repr=False)
    svd: TruncatedSVD = field(repr=False)
    normalizer: Normalizer = field(repr=False)
    char_limit: int
    X_tfidf: sparse.csr_matrix = field(repr=False)
    X_lsa: np.ndarray = field(repr=False)
    kept: np.ndarray = field(repr=False)  # bool mask over the input texts
    explained_variance_ratio: np.ndarray = field(repr=False)

    @property
    def feature_names(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()

    def matrix_stats(self) -> dict[str, float | int]:
        rows, cols = self.X_tfidf.shape
        return {
            "documents": int(rows),
            "tfidf_features": int(cols),
            "tfidf_nonzeros": int(self.X_tfidf.nnz),
            "tfidf_density": float(self.X_tfidf.nnz / max(1, rows * cols)),
            "tfidf_avg_terms_per_doc": float(self.X_tfidf.nnz / max(1, rows)),
            "lsa_dimensions": int(self.X_lsa.shape[1]),
            "lsa_explained_variance": float(self.explained_variance_ratio.sum()),
            "documents_dropped_without_features": int((~self.kept).sum()),
        }

    def transform(self, texts: Sequence[str | None]) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
        """Project other texts into this *News-fitted* space (no refit).

        Returns ``(X_tfidf, X_lsa, has_features)``; rows with no known term stay zero.
        """
        X = self.vectorizer.transform(to_documents(texts, self.char_limit)).tocsr()
        lsa = self.normalizer.transform(self.svd.transform(X))
        return X, lsa, np.asarray(X.getnnz(axis=1) > 0)


def fit_feature_space(texts: Sequence[str | None], cfg: BronzeNewsConfig) -> FeatureSpace:
    """TF-IDF (1-2 grams, sublinear tf) -> TruncatedSVD (LSA) -> L2 normalization.

    Documents that end up with no feature (e.g. only stopwords/rare terms) are dropped
    and flagged in ``kept``: they cannot be placed in a cosine space.
    """
    docs = to_documents(texts, cfg.text_char_limit)
    vectorizer = TfidfVectorizer(
        token_pattern=r"\S+",
        lowercase=False,
        ngram_range=tuple(cfg.ngram_range),
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        sublinear_tf=True,
        dtype=np.float64,
    )
    X = vectorizer.fit_transform(docs).tocsr()
    kept = np.asarray(X.getnnz(axis=1) > 0)
    X = X[kept]
    n_components = int(min(cfg.svd_components, X.shape[1] - 1, X.shape[0] - 1))
    if n_components < 2:
        raise ValueError(
            f"Quá ít dữ liệu để giảm chiều: {X.shape[0]} bài x {X.shape[1]} đặc trưng"
        )
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=cfg.seed)
    reduced = svd.fit_transform(X)
    normalizer = Normalizer(copy=False)
    X_lsa = normalizer.fit_transform(reduced)
    return FeatureSpace(
        vectorizer=vectorizer,
        svd=svd,
        normalizer=normalizer,
        char_limit=cfg.text_char_limit,
        X_tfidf=X,
        X_lsa=X_lsa,
        kept=kept,
        explained_variance_ratio=svd.explained_variance_ratio_,
    )
