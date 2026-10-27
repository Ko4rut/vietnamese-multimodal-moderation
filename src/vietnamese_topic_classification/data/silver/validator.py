"""Validate the VTSNLP source schema without altering the input record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vietnamese_topic_classification.config import get_domain_labels


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None


class SilverValidator:
    """Validate ID, text and source label; do not claim labels are ground truth."""

    def __init__(self, domains: tuple[str, ...] | None = None) -> None:
        self.domains = frozenset(domains if domains is not None else get_domain_labels())
        if not self.domains:
            raise ValueError("domains must not be empty")

    def validate(self, record: Mapping[str, Any]) -> ValidationResult:
        if not isinstance(record.get("id"), str) or not record["id"].strip():
            return ValidationResult(False, "invalid_id")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            return ValidationResult(False, "invalid_text")
        if not isinstance(record.get("domain"), str) or record["domain"] not in self.domains:
            return ValidationResult(False, "invalid_domain")
        return ValidationResult(True)
