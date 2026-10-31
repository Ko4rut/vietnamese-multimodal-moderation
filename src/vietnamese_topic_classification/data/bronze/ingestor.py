"""Reproducible, non-destructive Bronze ingestion from the pinned VTSNLP source.

Bronze preserves the three source fields without cleaning, filtering or relabeling.
Google Drive mounting/authentication is the caller's responsibility.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Mapping

from vietnamese_topic_classification.config import load_domain_config

from .model import (
    BronzeIngestionReport,
    BronzeIngestResult,
    BronzePublishedManifest,
    BronzeRecord,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_verified(source: Path, destination: Path) -> None:
    """Copy to a staging file, verify content, then publish without overwriting."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".uploading")
    if staging.exists():
        raise FileExistsError(f"Stale upload artifact exists: {staging}")
    try:
        shutil.copyfile(source, staging)
        if source.stat().st_size != staging.stat().st_size:
            raise IOError(f"Size mismatch while copying to {destination}")
        if _sha256_file(source) != _sha256_file(staging):
            raise IOError(f"Checksum mismatch while copying to {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            staging.unlink()


@dataclass(frozen=True)
class BronzeConfig:
    output_root: Path
    sample_size: int = 50_000
    seed: int = 42
    shuffle_buffer: int = 10_000
    local_temp_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise ValueError("sample_size must be positive")
        if self.shuffle_buffer < 2:
            raise ValueError("shuffle_buffer must be >= 2")


class BronzeIngestor:
    """Take a pinned, shuffled stream window and save an immutable Bronze snapshot."""

    def __init__(self, config: BronzeConfig) -> None:
        self.config = config
        self.source = load_domain_config()["dataset"]
        self.revision = self.source["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Dataset revision must be a pinned 40-character SHA")

    def _source_rows(self) -> Iterator[Mapping[str, Any]]:
        # Lazy import keeps offline package/unit-test installation lightweight.
        from datasets import load_dataset

        stream = load_dataset(
            self.source["id"],
            split=self.source["split"],
            revision=self.revision,
            streaming=True,
        )
        stream = stream.shuffle(
            seed=self.config.seed,
            buffer_size=self.config.shuffle_buffer,
        )
        return iter(stream)

    def run(self) -> BronzeIngestResult:
        root = Path(self.config.output_root).expanduser()
        suffix = (
            f"{self.revision[:10]}_seed{self.config.seed}"
            f"_buf{self.config.shuffle_buffer}_n{self.config.sample_size}"
        )
        outputs = {
            "bronze": root / "01_bronze" / f"vtsnlp_raw_sample_{suffix}.jsonl",
            "manifest": root / "00_source_metadata" / f"source_manifest_{suffix}.json",
            "report": root / "04_reports" / f"bronze_ingestion_{suffix}.json",
        }
        existing = [str(p) for p in outputs.values() if p.exists()]
        if existing:
            raise FileExistsError("Snapshot already exists; not overwriting: " + ", ".join(existing))

        domain_counts: Counter[str] = Counter()
        missing_fields: Counter[str] = Counter()
        count = 0

        # Assemble and verify locally before moving artifacts into mounted Drive.
        with tempfile.TemporaryDirectory(
            prefix="vtc_bronze_",
            dir=self.config.local_temp_dir,
        ) as temp:
            local = Path(temp)
            local_data = local / outputs["bronze"].name
            digest = hashlib.sha256()
            with local_data.open("wb") as handle:
                for row in islice(self._source_rows(), self.config.sample_size):
                    if not isinstance(row, Mapping):
                        raise TypeError(f"Source row {count} is not a mapping")
                    record: BronzeRecord = {
                        "id": row.get(self.source["id_column"]),
                        "text": row.get(self.source["text_column"]),
                        "domain": row.get(self.source["label_column"]),
                    }
                    for field, value in record.items():
                        if value is None:
                            missing_fields[field] += 1
                    domain = record["domain"]
                    domain_counts[str(domain) if domain is not None else "<missing>"] += 1
                    encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                    handle.write(encoded)
                    digest.update(encoded)
                    count += 1

            if count != self.config.sample_size:
                raise ValueError(
                    f"Source ended after {count} records; expected {self.config.sample_size}. "
                    "No Bronze artifacts were published."
                )

            created = datetime.now(timezone.utc).isoformat()
            manifest: BronzePublishedManifest = {
                "dataset": self.source["id"],
                "source_url": f"https://huggingface.co/datasets/{self.source['id']}",
                "revision": self.revision,
                "split": self.source["split"],
                "source_fields": {
                    "id": self.source["id_column"],
                    "text": self.source["text_column"],
                    "domain": self.source["label_column"],
                },
                "sample_size": count,
                "sampling": {
                    "method": "buffer_shuffle_then_first_n",
                    "seed": self.config.seed,
                    "shuffle_buffer": self.config.shuffle_buffer,
                    "note": "A streaming-buffer sample is not guaranteed to represent the entire corpus.",
                },
                "license_status": self.source.get("license_status", "unverified"),
                "label_provenance": self.source.get("label_provenance"),
                "file_name": outputs["bronze"].name,
                "file_size_bytes": local_data.stat().st_size,
                "sha256": digest.hexdigest(),
                "created_utc": created,
            }
            report: BronzeIngestionReport = {
                "sample_size": count,
                "domain_counts": dict(sorted(domain_counts.items())),
                "missing_field_counts": dict(sorted(missing_fields.items())),
                "schema_validation": "NOT_PERFORMED_BRONZE_PRESERVES_RAW_FIELDS",
                "deduplication": "NOT_PERFORMED",
                "near_duplicate_check": "NOT_PERFORMED",
                "created_utc": created,
            }
            for key, data in (("manifest", manifest), ("report", report)):
                (local / outputs[key].name).write_text(
                    json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            published: list[Path] = []
            try:
                for key in ("bronze", "manifest", "report"):
                    _publish_verified(local / outputs[key].name, outputs[key])
                    published.append(outputs[key])
            except Exception:
                # Do not leave behind a misleadingly completed snapshot.
                for path in published:
                    path.unlink(missing_ok=True)
                raise

        return {
            "count": count,
            "paths": {key: str(path) for key, path in outputs.items()},
            "sha256": manifest["sha256"],
            "domain_counts": report["domain_counts"],
            "missing_field_counts": report["missing_field_counts"],
        }
