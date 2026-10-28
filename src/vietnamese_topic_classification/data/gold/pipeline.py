"""Verify Silver and publish deterministic, immutable Gold V1 ML-ready splits."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vietnamese_topic_classification.data.silver import SilverReader

from .config import GoldConfig
from .exceptions import GoldPublicationError, GoldValidationError
from .model import (
    GoldManifest,
    GoldPipelineResult,
    GoldQualityReport,
    GoldSource,
    GoldSplitFile,
)
from .splitter import SPLITS, GoldSplitter
from .validator import GoldValidator

VERSION = "v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: GoldManifest | GoldQualityReport) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _publish(source: Path, destination: Path) -> None:
    """Use staged, checksum-verified copy and refuse pre-existing destinations."""
    if destination.exists():
        raise GoldPublicationError(f"Refusing to overwrite Gold artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + ".uploading")
    if staged.exists():
        raise GoldPublicationError(f"Stale Gold upload exists: {staged}")
    try:
        shutil.copyfile(source, staged)
        if (
            source.stat().st_size != staged.stat().st_size
            or _sha256_file(source) != _sha256_file(staged)
        ):
            raise GoldPublicationError(f"Copy verification failed: {destination}")
        if destination.exists():
            raise GoldPublicationError(f"Destination appeared during upload: {destination}")
        staged.rename(destination)
    finally:
        staged.unlink(missing_ok=True)


class GoldPipeline:
    """Build label-encoded, group-aware train/validation/test from one pinned Silver snapshot."""

    def __init__(
        self,
        reader: SilverReader,
        config: GoldConfig | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.reader = reader
        self.config = config or GoldConfig()
        self.root = Path(output_root) if output_root is not None else reader.root

    def run(self) -> GoldPipelineResult:
        source = self.reader.verify()
        if source["records"] == 0:
            raise GoldValidationError("Silver has no accepted records; cannot build Gold")
        validator = GoldValidator()
        params = {
            "source_sha256": source["sha256"],
            "config": self.config.as_dict(),
            "label_to_id": validator.label_to_id,
            "pipeline_version": VERSION,
        }
        config_hash = hashlib.sha256(
            json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        suffix = f"{source['sha256'][:12]}_{config_hash}"
        paths = {
            split: self.root / "03_gold" / f"gold_{split}_{VERSION}_{suffix}.jsonl"
            for split in SPLITS
        }
        paths["report"] = self.root / "05_reports" / f"gold_quality_{VERSION}_{suffix}.json"
        paths["manifest"] = (
            self.root / "03_gold" / "manifests" / f"gold_manifest_{VERSION}_{suffix}.json"
        )
        if any(
            path.exists() or path.with_name(path.name + ".uploading").exists()
            for path in paths.values()
        ):
            raise GoldPublicationError("Gold artifacts already exist or have stale upload files")

        rows = validator.convert_all(list(self.reader.iter_records()))
        if len(rows) != source["records"]:
            raise GoldValidationError("Silver record count changed after verification")
        indices, audit = GoldSplitter(self.config).split(rows)
        created = datetime.now(timezone.utc).isoformat()
        source_metadata: GoldSource = {
            "silver_file": self.reader.data_path.name,
            "silver_quality_report": self.reader.report_path.name,
            "silver_sha256": source["sha256"],
        }

        with tempfile.TemporaryDirectory(prefix="vtc_gold_") as temp:
            local = {name: Path(temp) / target.name for name, target in paths.items()}
            files: dict[str, GoldSplitFile] = {}
            for split in SPLITS:
                with local[split].open("w", encoding="utf-8", newline="\n") as handle:
                    for index in indices[split]:
                        handle.write(json.dumps(rows[index], ensure_ascii=False, sort_keys=True))
                        handle.write("\n")
                files[split] = {
                    "file_name": paths[split].name,
                    "rows": len(indices[split]),
                    "sha256": _sha256_file(local[split]),
                }
            if sum(entry["rows"] for entry in files.values()) != len(rows):
                raise GoldPublicationError("Gold outputs do not sum to Silver accepted rows")

            report: GoldQualityReport = {
                "pipeline_version": VERSION,
                "source": source_metadata,
                "input_rows": len(rows),
                "split_rows": audit["split_rows"],
                "split_class_counts": audit["split_class_counts"],
                "total_class_counts": dict(sorted(Counter(row["domain"] for row in rows).items())),
                "actual_ratios": audit["actual_ratios"],
                "target_ratios": audit["target_ratios"],
                "near_duplicate_audit": {
                    key: audit[key]
                    for key in ("method", "exhaustive", "candidates_checked", "matched_pairs", "groups")
                },
                "limitations": audit["note"],
                "created_utc": created,
            }
            manifest: GoldManifest = {
                "format_version": 1,
                "pipeline_version": VERSION,
                "source": source_metadata,
                "total_rows": len(rows),
                "label_to_id": validator.label_to_id,
                "split_config": self.config.as_dict(),
                "files": files,
                "quality_report": paths["report"].name,
                "created_utc": created,
            }
            _write_json(local["report"], report)
            _write_json(local["manifest"], manifest)

            published: list[Path] = []
            try:
                # The final manifest acts as the publication marker.
                for name in (*SPLITS, "report", "manifest"):
                    _publish(local[name], paths[name])
                    published.append(paths[name])
            except Exception:
                for finished in reversed(published):
                    finished.unlink(missing_ok=True)
                raise

        return {
            "paths": {name: str(path) for name, path in paths.items()},
            "report": report,
            "manifest": manifest,
        }
