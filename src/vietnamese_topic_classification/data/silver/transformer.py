"""Conservative and deterministic text normalization for Silver V1."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping


class SilverTransformer:
    """Preserve semantic content and paragraphs; no correction or truncation."""

    @staticmethod
    def normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[^\S\n]+", " ", normalized)
        return normalized.strip()

    def transform(self, record: Mapping[str, Any], source_snapshot: str) -> dict[str, Any]:
        text = self.normalize_text(record["text"])
        return {
            "id": record["id"],
            "text": text,
            "domain": record["domain"],
            "text_length": len(text),
            "source_snapshot": source_snapshot,
            "quality_flags": [],
        }
