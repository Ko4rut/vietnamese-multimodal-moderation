"""Validate ML-ready Gold records without altering their text."""

from __future__ import annotations

import hashlib
from typing import Any

from vietnamese_topic_classification.config import get_domain_labels

from .exceptions import GoldValidationError
from .model import GoldRecord


class GoldValidator:
    def __init__(self) -> None:
        self.label_to_id = {name: index for index, name in enumerate(get_domain_labels())}

    def convert(self, record: dict[str, Any], line_number: int) -> GoldRecord:
        if not isinstance(record, dict):
            raise GoldValidationError(f"Row {line_number}: expected JSON object")
        rid, text, domain = (record.get(key) for key in ("id", "text", "domain"))
        if not isinstance(rid, str) or not rid.strip():
            raise GoldValidationError(f"Row {line_number}: invalid id")
        if not isinstance(text, str) or not text.strip():
            raise GoldValidationError(f"Row {line_number}: invalid text")
        if not isinstance(domain, str) or domain not in self.label_to_id:
            raise GoldValidationError(f"Row {line_number}: unknown domain: {domain!r}")
        if type(record.get("text_length")) is not int or record["text_length"] != len(text):
            raise GoldValidationError(f"Row {line_number}: incorrect text_length")
        if not isinstance(record.get("source_snapshot"), str) or not record["source_snapshot"]:
            raise GoldValidationError(f"Row {line_number}: missing source_snapshot")
        if not isinstance(record.get("quality_flags"), list):
            raise GoldValidationError(f"Row {line_number}: quality_flags must be a list")
        if any(not isinstance(flag, str) for flag in record["quality_flags"]):
            raise GoldValidationError(f"Row {line_number}: nonstring quality flag")
        return {
            "id": rid,
            "text": text,
            "domain": domain,
            "label_id": self.label_to_id[domain],
            "text_length": len(text),
            "source_snapshot": record["source_snapshot"],
            "quality_flags": record["quality_flags"],
        }

    def convert_all(self, source: list[dict[str, Any]]) -> list[GoldRecord]:
        ids: set[str] = set()
        hashes: set[str] = set()
        rows: list[GoldRecord] = []
        for index, raw in enumerate(source, start=1):
            row = self.convert(raw, index)
            normalized_id = row["id"].strip()
            if normalized_id in ids:
                raise GoldValidationError(f"Row {index}: duplicate Silver ID {normalized_id!r}")
            ids.add(normalized_id)
            text_sha = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            if text_sha in hashes:
                raise GoldValidationError(f"Row {index}: exact duplicate Silver text")
            hashes.add(text_sha)
            rows.append(row)
        return rows
