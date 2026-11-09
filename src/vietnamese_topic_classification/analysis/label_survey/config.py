"""Survey configuration, deterministic run id and config hash."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

DEFAULT_DATASET = "VTSNLP/vietnamese_curated_dataset"
DEFAULT_REVISION = "b81fcce58945970117a1b56d50ec81be2628a5c3"


@dataclass(frozen=True)
class SurveyConfig:
    """Every parameter that can change the numbers of a survey run."""

    hf_dataset: str = DEFAULT_DATASET
    hf_revision: str = DEFAULT_REVISION
    seed: int = 42
    survey_target_rows: int = 300_000
    max_download_gb: float = 4.0
    # None = choose automatically from the parquet metadata (see ``plan_sample``).
    rows_per_row_group: int | None = None
    per_class_cap: int = 4_000
    text_char_limit: int = 5_000
    head_chars: int = 1_000
    n_folds: int = 5
    top_pairs: int = 15
    mismatch_prob: float = 0.70
    audit_per_class: int = 12
    audit_per_pair: int = 10
    audit_part_c: int = 30
    run_head_vs_full: bool = True
    # Same TF-IDF defaults as ``scripts/06_train_baseline.py`` (max_features=100000).
    max_features: int = 100_000
    # Bounded iterations keep Colab CPU time predictable; documented as a limitation.
    lr_max_iter: int = 300
    # Parallel HTTP range requests / worker processes (do not change the numbers).
    http_workers: int = 24
    n_jobs: int = 2

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.hf_revision):
            raise ValueError("hf_revision must be a pinned 40-character commit SHA")
        positive = {
            "survey_target_rows": self.survey_target_rows,
            "per_class_cap": self.per_class_cap,
            "text_char_limit": self.text_char_limit,
            "head_chars": self.head_chars,
            "top_pairs": self.top_pairs,
            "audit_per_class": self.audit_per_class,
            "audit_per_pair": self.audit_per_pair,
            "max_features": self.max_features,
            "lr_max_iter": self.lr_max_iter,
            "http_workers": self.http_workers,
            "n_jobs": self.n_jobs,
        }
        for name, value in positive.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.n_folds) is not int or self.n_folds < 2:
            raise ValueError("n_folds must be an integer >= 2")
        if not 0 < self.max_download_gb <= 64:
            raise ValueError("max_download_gb must be in (0, 64]")
        if not 0.5 <= self.mismatch_prob < 1:
            raise ValueError("mismatch_prob must be in [0.5, 1)")
        if self.rows_per_row_group is not None and self.rows_per_row_group < 1:
            raise ValueError("rows_per_row_group must be positive or None")
        if self.head_chars > self.text_char_limit:
            raise ValueError("head_chars must not exceed text_char_limit")

    @property
    def stored_text_chars(self) -> int:
        """Characters of each text kept in the sample cache (enough for every use)."""
        return max(self.text_char_limit, self.head_chars, 700)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SurveyConfig":
        """Build from notebook globals; keys are UPPER_CASE, unknown keys are ignored."""
        names = {f.name for f in fields(cls)}
        picked = {k.lower(): v for k, v in values.items() if k.lower() in names}
        return cls(**picked)

    def _digest(self, data: Mapping[str, Any]) -> str:
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def config_hash(self, extra: Mapping[str, Any] | None = None) -> str:
        """Hash of everything that affects results (``extra``: e.g. Bronze ID digest)."""
        data = asdict(self)
        # Parallelism changes speed, never the numbers.
        data.pop("http_workers"), data.pop("n_jobs")
        data["extra"] = dict(extra or {})
        return self._digest(data)

    def sample_hash(self, extra: Mapping[str, Any] | None = None) -> str:
        """Hash of what determines the downloaded sample (cache key of the sample)."""
        data = {
            "dataset": self.hf_dataset,
            "revision": self.hf_revision,
            "seed": self.seed,
            "rows": self.survey_target_rows,
            "max_gb": self.max_download_gb,
            "rows_per_row_group": self.rows_per_row_group,
            "stored_chars": self.stored_text_chars,
            "extra": dict(extra or {}),
        }
        return self._digest(data)

    def run_id(self, extra: Mapping[str, Any] | None = None) -> str:
        return (
            f"survey_{self.hf_revision[:8]}_seed{self.seed}"
            f"_n{self.survey_target_rows}_{self.config_hash(extra)[:8]}"
        )
