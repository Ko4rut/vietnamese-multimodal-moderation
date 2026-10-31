"""Typed contracts for Silver rows and quality-report JSON.

The canonical dataclass ValidationResult is defined here; the existing
silver.validator import path still re-exports it for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class SilverRecord(TypedDict):
    id: str
    text: str
    domain: str
    text_length: int
    source_snapshot: str
    quality_flags: list[str]


class SilverRejectedRecord(TypedDict):
    source_snapshot: str
    source_line: int
    reason: str
    record: dict[str, Any]


class SilverOutputFiles(TypedDict):
    silver: str


class SilverPublishedFiles(SilverOutputFiles, total=False):
    rejected: str
    report: str


class SilverQualityReport(TypedDict):
    """Minimum fields used by SilverReader to pin a verified snapshot."""

    output_files: SilverOutputFiles
    silver_sha256: str
    accepted_rows: int


class SilverTextLengthStats(TypedDict):
    min: int | None
    max: int | None
    mean: float | None


class SilverPublishedReport(SilverQualityReport, total=False):
    """Additional fields emitted by SilverPipeline, absent in minimal fixtures."""

    pipeline_version: str
    source_snapshot: str
    source_manifest: str
    source_sha256: str
    input_rows: int
    rejected_rows: int
    rejection_reasons: dict[str, int]
    accepted_domain_counts: dict[str, int]
    text_length: SilverTextLengthStats
    normalization: str
    deduplication: str
    rejected_sha256: str
    created_utc: str


class SilverRunResult(TypedDict):
    paths: dict[str, str]
    report: SilverPublishedReport


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None
