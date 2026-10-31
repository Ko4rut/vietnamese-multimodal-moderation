"""Immutable Gold V1 settings; model-specific tokenization is deliberately excluded."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Canonical Data Lake folder shared by Gold publication and reading.
GOLD_DIR_NAME = "03_gold"


@dataclass(frozen=True)
class GoldConfig:
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    similarity_threshold: float = 0.9
    shingle_words: int = 5
    signature_buckets: int = 16
    max_candidates_per_record: int = 2000

    def __post_init__(self) -> None:
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(not 0 < ratio < 1 for ratio in ratios):
            raise ValueError("Gold split ratios must all be between 0 and 1")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("Gold split ratios must sum to 1")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Gold seed must be a nonnegative integer")
        if not 0 < self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if not 2 <= self.shingle_words <= 10:
            raise ValueError("shingle_words must be in [2, 10]")
        if not 2 <= self.signature_buckets <= 64:
            raise ValueError("signature_buckets must be in [2, 64]")
        if self.max_candidates_per_record < 1:
            raise ValueError("max_candidates_per_record must be positive")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)
