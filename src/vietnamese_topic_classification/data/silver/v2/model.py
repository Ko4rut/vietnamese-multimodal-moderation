"""Typed contracts for Silver V2 suffix cleanup and publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

SilverV2Action = Literal["unchanged", "cleaned", "review", "guardrail"]


@dataclass(frozen=True, slots=True)
class SilverV2CleaningDecision:
    text: str
    action: SilverV2Action
    signals: tuple[str, ...]
    removed_chars: int
    removed_preview: str


class SilverV2AuditRow(TypedDict):
    id: str
    domain: str
    action: SilverV2Action
    signals: list[str]
    before_chars: int
    after_chars: int
    removed_chars: int
    removed_preview: str


class SilverV2Report(TypedDict):
    pipeline_version: str
    ruleset_version: str
    cleaning_config: dict[str, int | float]
    source_report: str
    source_silver_sha256: str
    source_rows: int
    input_rows: int
    accepted_rows: int
    rejected_rows: int
    rejected_reasons: dict[str, int]
    cleaned_rows: int
    review_rows: int
    guardrail_rows: int
    detected_rows: int
    removed_total_chars: int
    signal_counts: dict[str, int]
    accepted_domain_counts: dict[str, int]
    output_files: dict[str, str]
    silver_sha256: str
    rejected_sha256: str
    audit_sha256: str
    created_utc: str


class SilverV2RunResult(TypedDict):
    paths: dict[str, str]
    report: SilverV2Report
    dry_run: bool
