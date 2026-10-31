"""Reproducible Bronze -> Silver V1 streaming pipeline with safe publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vietnamese_topic_classification.data.bronze import BronzeReader
from vietnamese_topic_classification.data.silver.exceptions import SilverPublicationError
from vietnamese_topic_classification.data.silver.v1.model import (
    SilverPublishedReport,
    SilverRecord,
    SilverRejectedRecord,
    SilverRunResult,
)
from vietnamese_topic_classification.data.silver.v1.transformer import SilverTransformer
from vietnamese_topic_classification.data.silver.v1.validator import SilverValidator

SILVER_VERSION = "v1"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(handle: Any, item: SilverRecord | SilverRejectedRecord) -> None:
    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _publish(source: Path, destination: Path) -> None:
    """Verify a staged copy; never replace an existing artifact."""
    if destination.exists():
        raise SilverPublicationError(f"Silver artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + ".uploading")
    if staged.exists():
        raise SilverPublicationError(f"Stale Silver upload exists: {staged}")
    try:
        shutil.copyfile(source, staged)
        if source.stat().st_size != staged.stat().st_size or _hash_file(source) != _hash_file(staged):
            raise SilverPublicationError(f"Copy verification failed: {destination}")
        staged.rename(destination)
    finally:
        staged.unlink(missing_ok=True)


class SilverPipeline:
    """Run deterministic Silver V1 processing from one pinned BronzeReader."""

    def __init__(
        self,
        reader: BronzeReader,
        output_root: Path | None = None,
        validator: SilverValidator | None = None,
        transformer: SilverTransformer | None = None,
    ) -> None:
        self.reader = reader
        self.root = Path(output_root) if output_root is not None else reader.root
        self.validator = validator if validator is not None else SilverValidator()
        self.transformer = transformer if transformer is not None else SilverTransformer()

    def run(self) -> SilverRunResult:
        """Verify source, write accepted/rejected JSONL and publish an audit report."""
        source_name = self.reader.data_path.stem
        suffix = source_name.removeprefix("vtsnlp_raw_sample_")
        output = {
            "silver": self.root / "02_silver" / SILVER_VERSION / f"vtsnlp_silver_{SILVER_VERSION}_{suffix}.jsonl",
            "rejected": self.root / "02_silver" / SILVER_VERSION / f"vtsnlp_rejected_{SILVER_VERSION}_{suffix}.jsonl",
            "report": self.root / "05_reports" / "silver" / SILVER_VERSION / f"silver_quality_{SILVER_VERSION}_{suffix}.json",
        }
        legacy_paths = [
            self.root / "02_silver" / output["silver"].name,
            self.root / "02_silver" / output["rejected"].name,
            self.root / "05_reports" / output["report"].name,
        ]
        if any(path.exists() or path.with_name(path.name + ".uploading").exists()
               for path in (*output.values(), *legacy_paths)):
            raise SilverPublicationError("Silver snapshot already exists or has stale upload files")

        verification = self.reader.verify()
        counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        length_total = 0
        min_length: int | None = None
        max_length = 0
        seen_ids: set[str] = set()
        text_domain_by_digest: dict[str, str] = {}

        with tempfile.TemporaryDirectory(prefix="vtc_silver_") as temp_dir:
            tmp = Path(temp_dir)
            local = {key: tmp / path.name for key, path in output.items()}
            with local["silver"].open("w", encoding="utf-8", newline="\n") as accepted, local[
                "rejected"
            ].open("w", encoding="utf-8", newline="\n") as rejected:
                for line_number, record in enumerate(self.reader.iter_records(), start=1):
                    counts["input"] += 1
                    validation = self.validator.validate(record)
                    reason = validation.reason
                    transformed = None
                    if validation.valid:
                        transformed = self.transformer.transform(record, self.reader.data_path.name)
                        if not transformed["text"]:
                            reason = "empty_after_normalization"
                        elif record["id"].strip() in seen_ids:
                            reason = "duplicate_id"
                        else:
                            text_digest = hashlib.sha256(transformed["text"].encode("utf-8")).hexdigest()
                            if text_digest in text_domain_by_digest:
                                previous_domain = text_domain_by_digest[text_digest]
                                reason = (
                                    "duplicate_text_conflicting_domain"
                                    if previous_domain != record["domain"]
                                    else "duplicate_text"
                                )
                    if reason is not None:
                        counts["rejected"] += 1
                        counts[f"reason:{reason}"] += 1
                        _write_jsonl(rejected, SilverRejectedRecord({
                            "source_snapshot": self.reader.data_path.name,
                            "source_line": line_number,
                            "reason": reason,
                            "record": record,
                        }))
                        continue

                    # Duplicates keep the first valid representative, in source order.
                    assert transformed is not None
                    _write_jsonl(accepted, transformed)
                    counts["accepted"] += 1
                    seen_ids.add(record["id"].strip())
                    text_domain_by_digest[text_digest] = record["domain"]
                    label_counts[record["domain"]] += 1
                    size = transformed["text_length"]
                    length_total += size
                    min_length = size if min_length is None else min(min_length, size)
                    max_length = max(max_length, size)

            if counts["input"] != verification["records"]:
                raise SilverPublicationError("Input count changed after Bronze verification")

            report: SilverPublishedReport = {
                "pipeline_version": SILVER_VERSION,
                "source_snapshot": self.reader.data_path.name,
                "source_manifest": self.reader.manifest_path.name,
                "source_sha256": verification["sha256"],
                "input_rows": counts["input"],
                "accepted_rows": counts["accepted"],
                "rejected_rows": counts["rejected"],
                "rejection_reasons": dict(sorted((key.removeprefix("reason:"), value) for key, value in counts.items() if key.startswith("reason:"))),
                "accepted_domain_counts": dict(sorted(label_counts.items())),
                "text_length": {
                    "min": min_length,
                    "max": max_length if counts["accepted"] else None,
                    "mean": length_total / counts["accepted"] if counts["accepted"] else None,
                },
                "normalization": "NFC; CRLF/CR to LF; collapse horizontal whitespace; strip edges",
                "deduplication": "first valid ID and normalized exact text retained; domain conflicts rejected",
                "output_files": {key: path.name for key, path in output.items()},
                "silver_sha256": _hash_file(local["silver"]),
                "rejected_sha256": _hash_file(local["rejected"]),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            local["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            published: list[Path] = []
            try:
                for name in ("silver", "rejected", "report"):
                    _publish(local[name], output[name])
                    published.append(output[name])
            except Exception:
                for path in published:
                    path.unlink(missing_ok=True)
                raise

        return {"paths": {key: str(path) for key, path in output.items()}, "report": report}
