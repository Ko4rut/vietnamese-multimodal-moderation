"""Typed contracts for Bronze raw rows, source manifests and ingestion reports.

These TypedDicts describe JSON payloads; they do not validate untrusted input
at runtime. Bronze intentionally preserves potentially missing/invalid values.
"""

from __future__ import annotations

from typing import Any, TypedDict


class BronzeRecord(TypedDict):
    """Source fields before Silver schema validation (values may be None)."""

    id: Any
    text: Any
    domain: Any


class BronzeManifest(TypedDict):
    """Fields required to locate and verify a Bronze JSONL snapshot."""

    file_name: str
    sha256: str
    sample_size: int
    file_size_bytes: int


class BronzeSourceFields(TypedDict):
    id: str
    text: str
    domain: str


class BronzeSampling(TypedDict):
    method: str
    seed: int
    shuffle_buffer: int
    note: str


class BronzePublishedManifest(BronzeManifest, total=False):
    """Additional provenance fields emitted by BronzeIngestor."""

    dataset: str
    source_url: str
    revision: str
    split: str
    source_fields: BronzeSourceFields
    sampling: BronzeSampling
    license_status: str
    label_provenance: str | None
    created_utc: str


class BronzeIngestionReport(TypedDict):
    sample_size: int
    domain_counts: dict[str, int]
    missing_field_counts: dict[str, int]
    schema_validation: str
    deduplication: str
    near_duplicate_check: str
    created_utc: str


class BronzeIngestResult(TypedDict):
    count: int
    paths: dict[str, str]
    sha256: str
    domain_counts: dict[str, int]
    missing_field_counts: dict[str, int]
