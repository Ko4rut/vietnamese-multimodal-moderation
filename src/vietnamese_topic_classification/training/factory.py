"""Build an unfitted sklearn pipeline; TF-IDF is fit only by Pipeline.fit()."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import ExperimentConfig


def build_pipeline(config: ExperimentConfig) -> Pipeline:
    """Construct the requested, unfitted TF-IDF + linear classifier baseline."""
    if config.features.kind != "tfidf":
        raise ValueError(f"Unsupported feature extractor: {config.features.kind!r}")
    feature_params: dict[str, object] = {"ngram_range": (1, 2), "min_df": 2}
    feature_params.update(config.features.params)
    vectorizer = TfidfVectorizer(**feature_params)

    classifier_params: dict[str, object] = {}
    classifier_params.update(config.classifier.params)
    if config.classifier.kind == "logistic_regression":
        classifier_params.setdefault("max_iter", 1000)
        classifier_params.setdefault("random_state", config.seed)
        classifier = LogisticRegression(**classifier_params)
    elif config.classifier.kind == "linear_svc":
        classifier_params.setdefault("random_state", config.seed)
        classifier = LinearSVC(**classifier_params)
    else:
        raise ValueError(f"Unsupported classifier: {config.classifier.kind!r}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
