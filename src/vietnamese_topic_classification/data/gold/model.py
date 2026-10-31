"""Typed contracts for Gold ML-ready rows, leakage audit and published metadata.

GoldRecord extends the Silver row contract with a stable 0-based label_id.
No model-specific TF-IDF features, embeddings, or token IDs are stored here.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from vietnamese_topic_classification.data.silver.v1.model import SilverRecord

GoldSplitName = Literal["train", "validation", "test"]


class GoldRecord(SilverRecord):
    label_id: int


class GoldSource(TypedDict):
    silver_file: str
    silver_quality_report: str
    silver_sha256: str


class GoldSplitFile(TypedDict):
    file_name: str
    rows: int
    sha256: str


class GoldCandidateAudit(TypedDict):
    method: str
    exhaustive: bool
    candidates_checked: int
    matched_pairs: int
    groups: int


class GoldSplitAudit(GoldCandidateAudit):
    split_rows: dict[str, int]
    split_class_counts: dict[str, dict[str, int]]
    target_ratios: dict[str, float]
    actual_ratios: dict[str, float]
    note: str


class GoldManifest(TypedDict):
    format_version: int
    pipeline_version: str
    source: GoldSource
    total_rows: int
    label_to_id: dict[str, int]
    split_config: dict[str, int | float]
    files: dict[str, GoldSplitFile]
    quality_report: str
    created_utc: str


class GoldQualityReport(TypedDict):
    pipeline_version: str
    source: GoldSource
    input_rows: int
    split_rows: dict[str, int]
    split_class_counts: dict[str, dict[str, int]]
    total_class_counts: dict[str, int]
    actual_ratios: dict[str, float]
    target_ratios: dict[str, float]
    near_duplicate_audit: GoldCandidateAudit
    limitations: str
    created_utc: str


class GoldPipelineResult(TypedDict):
    paths: dict[str, str]
    report: GoldQualityReport
    manifest: GoldManifest
