"""Build immutable Gold V2 from a verified Silver V2 quality report.

Reuses the V1 Gold splitter and validator, but refuses Silver V1 inputs.
Published V1 artifacts remain untouched.
"""

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

from .config import GOLD_DIR_NAME, GoldConfig
from .exceptions import GoldPublicationError, GoldValidationError
from .splitter import SPLITS, GoldSplitter
from .validator import GoldValidator

VERSION = "v2"


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _publish(source: Path, dest: Path) -> None:
    if dest.exists():
        raise GoldPublicationError(f"Refusing to overwrite Gold V2: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staged = dest.with_name(dest.name + ".uploading")
    if staged.exists():
        raise GoldPublicationError(f"Stale upload: {staged}")
    try:
        shutil.copyfile(source, staged)
        if _digest(source) != _digest(staged) or source.stat().st_size != staged.stat().st_size:
            raise GoldPublicationError(f"Gold V2 publication checksum mismatch: {dest}")
        if dest.exists():
            raise GoldPublicationError(f"Gold V2 appeared during publish: {dest}")
        staged.rename(dest)
    finally:
        staged.unlink(missing_ok=True)


class GoldV2Pipeline:
    """Gold V2 has an immutable, explicitly versioned and auditable source contract."""

    def __init__(
        self,
        reader: SilverReader,
        config: GoldConfig | None = None,
        output_root: Path | None = None,
        *,
        allow_unreviewed: bool = False,
    ) -> None:
        self.reader = reader
        self.config = config or GoldConfig()
        self.root = Path(output_root) if output_root is not None else reader.root
        self.allow_unreviewed = allow_unreviewed

    def _source(self) -> tuple[dict[str, Any], dict[str, Any]]:
        report = self.reader.report
        if report.get("pipeline_version") != "v2":
            raise GoldValidationError("Gold V2 requires a Silver V2 quality report")
        if report.get("ruleset_version") is None:
            raise GoldValidationError("Silver V2 must identify its cleaning ruleset")
        counts = ("input_rows", "accepted_rows", "rejected_rows", "cleaned_rows",
                  "review_rows", "guardrail_rows")
        if any(type(report.get(key)) is not int or report[key] < 0 for key in counts):
            raise GoldValidationError("Silver V2 report has invalid counts")
        if report["input_rows"] != report["accepted_rows"] + report["rejected_rows"]:
            raise GoldValidationError("Silver V2 input/accepted/rejected counts differ")
        files = report.get("output_files")
        if not isinstance(files, dict) or not all(k in files for k in ("audit", "silver")):
            raise GoldValidationError("Silver V2 audit file is missing from report")
        name = files["audit"]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".jsonl"):
            raise GoldValidationError("Unsafe Silver V2 audit filename")
        expected_hash = report.get("audit_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise GoldValidationError("Silver V2 audit checksum is missing")
        audit_path = self.reader.root / "05_reports" / name
        if not audit_path.is_file() or _digest(audit_path) != expected_hash:
            raise GoldValidationError("Silver V2 cleaning audit is absent or altered")
        if not self.allow_unreviewed and (report["review_rows"] or report["guardrail_rows"]):
            raise GoldValidationError(
                "Silver V2 has review/guardrail rows; audit these before creating Gold V2. "
                "Use --allow-unreviewed only after explicitly accepting them."
            )
        verified = self.reader.verify()
        if verified["records"] != report["accepted_rows"] or not verified["records"]:
            raise GoldValidationError("Silver V2 accepted rows do not match verified file")
        return report, verified

    def planned_paths(self, source_sha256: str) -> dict[str, Path]:
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise ValueError("Expected a verified Silver V2 SHA256")
        options = {
            "source_sha256": source_sha256,
            "config": self.config.as_dict(),
            "label_mapping": GoldValidator().label_to_id,
            "version": VERSION,
            "allow_unreviewed": self.allow_unreviewed,
        }
        fingerprint = hashlib.sha256(
            json.dumps(options, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:10]
        suffix = f"{source_sha256[:12]}_{fingerprint}"
        paths = {
            split: self.root / GOLD_DIR_NAME / f"gold_{split}_{VERSION}_{suffix}.jsonl"
            for split in SPLITS
        }
        paths["report"] = self.root / "05_reports" / f"gold_quality_{VERSION}_{suffix}.json"
        paths["manifest"] = (
            self.root / GOLD_DIR_NAME / "manifests" / f"gold_manifest_{VERSION}_{suffix}.json"
        )
        return paths

    def run(self) -> dict[str, Any]:
        source_report, verified = self._source()
        paths = self.planned_paths(verified["sha256"])
        if any(
            path.exists() or path.with_name(path.name + ".uploading").exists()
            for path in paths.values()
        ):
            raise GoldPublicationError("Gold V2 artifacts already exist or have stale uploads")
        validator = GoldValidator()
        rows = validator.convert_all(list(self.reader.iter_records()))
        if len(rows) != verified["records"]:
            raise GoldValidationError("Silver V2 changed while reading")
        flags = Counter(flag for row in rows for flag in row["quality_flags"])
        pending = sum(
            any(flag in ("silver_v2:review", "silver_v2:guardrail")
                for flag in row["quality_flags"]) for row in rows
        )
        if pending and not self.allow_unreviewed:
            raise GoldValidationError("Silver V2 records still have unreviewed quality flags")
        allocations, audit = GoldSplitter(self.config).split(rows)
        source = {
            "silver_file": self.reader.data_path.name,
            "silver_quality_report": self.reader.report_path.name,
            "silver_sha256": verified["sha256"],
        }
        created = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory(prefix="vtc_gold_v2_") as temp:
            locals_ = {key: Path(temp) / path.name for key, path in paths.items()}
            split_files = {}
            for split in SPLITS:
                with locals_[split].open("w", encoding="utf-8", newline="\n") as stream:
                    for index in allocations[split]:
                        stream.write(json.dumps(rows[index], ensure_ascii=False, sort_keys=True))
                        stream.write("\n")
                split_files[split] = {
                    "file_name": paths[split].name,
                    "rows": len(allocations[split]),
                    "sha256": _digest(locals_[split]),
                }
            report = {
                "pipeline_version": VERSION,
                "source": source,
                "source_silver_v2": {
                    "ruleset_version": source_report["ruleset_version"],
                    "cleaning_audit": source_report["output_files"]["audit"],
                    "cleaning_audit_sha256": source_report["audit_sha256"],
                    "cleaned_rows": source_report["cleaned_rows"],
                    "review_rows": source_report["review_rows"],
                    "guardrail_rows": source_report["guardrail_rows"],
                    "allow_unreviewed": self.allow_unreviewed,
                },
                "input_rows": len(rows),
                "split_rows": audit["split_rows"],
                "split_class_counts": audit["split_class_counts"],
                "total_class_counts": dict(sorted(Counter(row["domain"] for row in rows).items())),
                "quality_flag_counts": dict(sorted(flags.items())),
                "actual_ratios": audit["actual_ratios"],
                "target_ratios": audit["target_ratios"],
                "near_duplicate_audit": {
                    key: audit[key]
                    for key in ("method", "exhaustive", "candidates_checked", "matched_pairs",
                                "groups")
                },
                "limitations": audit["note"],
                "created_utc": created,
            }
            # format_version=1 retains compatibility with the existing GoldReader,
            # while pipeline_version=v2 differentiates the new snapshot.
            manifest = {
                "format_version": 1,
                "pipeline_version": VERSION,
                "source": source,
                "source_silver_v2": report["source_silver_v2"],
                "total_rows": len(rows),
                "label_to_id": validator.label_to_id,
                "split_config": self.config.as_dict(),
                "files": split_files,
                "quality_report": paths["report"].name,
                "created_utc": created,
            }
            for name, obj in (("report", report), ("manifest", manifest)):
                locals_[name].write_text(
                    json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
            created_paths: list[Path] = []
            try:
                for name in (*SPLITS, "report", "manifest"):
                    _publish(locals_[name], paths[name])
                    created_paths.append(paths[name])
            except Exception:
                for path in reversed(created_paths):
                    path.unlink(missing_ok=True)
                raise
        return {
            "paths": {key: str(path) for key, path in paths.items()},
            "report": report, "manifest": manifest,
        }
