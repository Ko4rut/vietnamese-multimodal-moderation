"""Run configuration, deterministic run id and config hash."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

NEIGHBOR_LABELS = (
    "Law_and_Government",
    "Sensitive_Subjects",
    "People_and_Society",
    "Business_and_Industrial",
    "Sports",
)

ALGORITHM_NAMES = (
    "kmeans",
    "minibatch_kmeans",
    "gmm",
    "birch",
    "ward",
    "agglomerative_average_cosine",
    "agglomerative_complete_cosine",
    "hdbscan",
    "dbscan",
    "optics",
    "spectral",
    "affinity_propagation",
    "meanshift",
)


@dataclass(frozen=True)
class BronzeNewsConfig:
    """Every parameter that can change the numbers of a run."""

    seed: int = 42
    news_label: str = "News"
    neighbor_labels: tuple[str, ...] = NEIGHBOR_LABELS
    include_neighbor_labels: bool = True
    neighbor_per_label: int = 600
    # --- EDA / data quality
    drop_exact_duplicates: bool = True
    near_dup_sample: int = 3000
    near_dup_threshold: float = 0.9
    short_words: int = 30
    long_words: int = 3000
    top_terms: int = 25
    # --- features (TF-IDF + LSA)
    text_char_limit: int = 8000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 3
    max_df: float = 0.6
    max_features: int = 30_000
    svd_components: int = 100
    # --- shape diagnostics
    tsne_max_points: int = 4000
    tsne_perplexity: float = 30.0
    hopkins_repeats: int = 10
    kdist_k: int = 10
    cosine_pairs_sample: int = 1500
    # --- clustering
    k_range: tuple[int, int] = (3, 15)
    algorithms: tuple[str, ...] = ALGORITHM_NAMES
    slow_max_points: int = 1500
    time_budget_seconds: float = 300.0
    stability_seeds: int = 5
    main_min_clusters: int = 5
    main_max_noise: float = 0.30
    main_min_cluster_frac: float = 0.01
    main_min_stability: float = 0.5
    n_main: int = 2
    # --- interpretation
    keywords_per_cluster: int = 20
    representatives_per_cluster: int = 3
    excerpt_chars: int = 500
    knn_k: int = 15
    overlay_max_points: int = 3000
    n_jobs: int = field(default=1, compare=False)

    def __post_init__(self) -> None:
        for name in ("seed",):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        positive = (
            "neighbor_per_label", "near_dup_sample", "short_words", "long_words", "top_terms",
            "text_char_limit", "min_df", "max_features", "svd_components", "tsne_max_points",
            "hopkins_repeats", "kdist_k", "cosine_pairs_sample", "slow_max_points",
            "stability_seeds", "main_min_clusters", "n_main", "keywords_per_cluster",
            "representatives_per_cluster", "excerpt_chars", "knn_k", "overlay_max_points",
        )
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.near_dup_threshold <= 1:
            raise ValueError("near_dup_threshold must be in (0, 1]")
        if not 0 < self.max_df <= 1:
            raise ValueError("max_df must be in (0, 1]")
        if len(self.ngram_range) != 2 or not 1 <= self.ngram_range[0] <= self.ngram_range[1]:
            raise ValueError("ngram_range must be (min_n, max_n) with 1 <= min_n <= max_n")
        if len(self.k_range) != 2 or not 2 <= self.k_range[0] <= self.k_range[1]:
            raise ValueError("k_range must be (k_min, k_max) with 2 <= k_min <= k_max")
        unknown = set(self.algorithms) - set(ALGORITHM_NAMES)
        if unknown:
            raise ValueError(f"Unknown algorithms: {sorted(unknown)}")
        if self.time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive")
        if not 0 <= self.main_max_noise < 1:
            raise ValueError("main_max_noise must be in [0, 1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BronzeNewsConfig":
        """Build from notebook globals (UPPER_CASE keys; unknown keys are ignored).

        ``ALGORITHMS`` may be a ``{name: bool}`` dict (notebook style) or a sequence.
        """
        names = {f.name for f in fields(cls)}
        picked: dict[str, Any] = {}
        for key, value in values.items():
            lower = key.lower()
            if lower not in names:
                continue
            if lower == "algorithms" and isinstance(value, Mapping):
                value = tuple(name for name, on in value.items() if on)
            elif isinstance(value, list):
                value = tuple(value)
            picked[lower] = value
        return cls(**picked)

    def config_hash(self, extra: Mapping[str, Any] | None = None) -> str:
        data = asdict(self)
        # Speed / machine dependent knobs never define the result.
        data.pop("n_jobs")
        data.pop("time_budget_seconds")
        data["extra"] = dict(extra or {})
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def run_id(self, bronze_sha256: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", bronze_sha256):
            raise ValueError("bronze_sha256 must be a 64-character hex digest")
        return (
            f"bronze_news_{bronze_sha256[:8]}_seed{self.seed}"
            f"_{self.config_hash({'bronze_sha256': bronze_sha256})[:8]}"
        )
