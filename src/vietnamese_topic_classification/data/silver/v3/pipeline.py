"""Create an immutable Silver V3 label-refinement snapshot from Silver V2."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vietnamese_topic_classification.config import get_v3_domain_labels
from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.silver.exceptions import (
    SilverPublicationError,
    SilverReportError,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(source: Path, destination: Path) -> None:
    if destination.exists():
        raise SilverPublicationError(f"Refusing to overwrite Silver V3 artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + ".uploading")
    if staged.exists():
        raise SilverPublicationError(f"Stale Silver V3 upload exists: {staged}")
    try:
        shutil.copyfile(source, staged)
        if source.stat().st_size != staged.stat().st_size or _sha256_file(source) != _sha256_file(staged):
            raise SilverPublicationError(f"Silver V3 copy verification failed: {destination}")
        if destination.exists():
            raise SilverPublicationError(f"Silver V3 destination appeared during upload: {destination}")
        staged.rename(destination)
    finally:
        staged.unlink(missing_ok=True)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class SilverV3Pipeline:
    """Relabel News only after a human has approved every decision."""

    version = "v3"

    def __init__(
        self,
        source: SilverReader,
        decisions_file: Path | None = None,
        *,
        output_root: Path | None = None,
    ) -> None:
        self.source = source
        self.decisions_file = Path(decisions_file) if decisions_file is not None else None
        self.root = Path(output_root) if output_root is not None else source.root
        if source.report.get("pipeline_version") != "v2":
            raise SilverReportError("Silver V3 requires a verified Silver V2 source")
        if source.report_path.parent != source.root / "04_reports" / "silver" / "v2":
            raise SilverReportError("Silver V2 report must be under 04_reports/silver/v2")

    def verify_source(self) -> dict[str, Any]:
        """Verify Silver V2 artifacts and its Silver V1 lineage before processing."""
        from vietnamese_topic_classification.data.gold.v2 import GoldV2Pipeline

        return GoldV2Pipeline(self.source).verify_silver_publication()

    def _load_decisions(self) -> tuple[dict[str, dict[str, Any]], str]:
        if self.decisions_file is None:
            raise ValueError("A reviewed decision file is required to build Silver V3")
        allowed = set(get_v3_domain_labels())
        decisions: dict[str, dict[str, Any]] = {}
        try:
            with self.decisions_file.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid decision JSONL at line {line_number}") from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"Decision line {line_number} must be a JSON object")
                    record_id = row.get("id")
                    label = row.get("label")
                    if not isinstance(record_id, str) or not record_id.strip():
                        raise ValueError(f"Decision line {line_number}: missing id")
                    if record_id in decisions:
                        raise ValueError(f"Duplicate decision id: {record_id}")
                    if not isinstance(label, str) or label not in allowed or label == "News":
                        raise ValueError(f"Decision {record_id}: unknown V3 label {label!r}")
                    if row.get("approved") is not True:
                        raise ValueError(f"Decision {record_id}: not human-approved")
                    if row.get("needs_review") is not False:
                        raise ValueError(f"Decision {record_id}: still needs review")
                    reviewer = row.get("reviewer")
                    reason = row.get("reason")
                    if not isinstance(reviewer, str) or not reviewer.strip():
                        raise ValueError(f"Decision {record_id}: reviewer is required")
                    if not isinstance(reason, str) or not reason.strip():
                        raise ValueError(f"Decision {record_id}: reason is required")
                    decisions[record_id] = {
                        "label": label,
                        "reviewer": reviewer.strip()[:200],
                        "reason": reason.strip()[:500],
                    }
        except OSError as exc:
            raise ValueError(f"Cannot read decision file: {self.decisions_file}") from exc
        return decisions, _sha256_file(self.decisions_file)

    def planned_paths(self, source_sha256: str, decisions_sha256: str) -> dict[str, Path]:
        params = {
            "source_sha256": source_sha256,
            "source_report_sha256": _sha256_file(self.source.report_path),
            "decisions_sha256": decisions_sha256,
            "labels": get_v3_domain_labels(),
            "pipeline_version": self.version,
        }
        config_hash = hashlib.sha256(
            json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        suffix = f"{source_sha256[:12]}_{config_hash}"
        return {
            "silver": self.root / "02_silver" / "v3" / f"vtsnlp_silver_v3_{suffix}.jsonl",
            "audit": self.root / "04_reports" / "silver" / "v3"
            / f"silver_audit_v3_{suffix}.jsonl",
            "decisions": self.root / "04_reports" / "silver" / "v3"
            / f"silver_v3_decisions_{suffix}.jsonl",
            "report": self.root / "04_reports" / "silver" / "v3"
            / f"silver_quality_v3_{suffix}.json",
        }

    def run(self) -> dict[str, Any]:
        source_verified = self.verify_source()
        decisions, decisions_sha256 = self._load_decisions()
        paths = self.planned_paths(source_verified["sha256"], decisions_sha256)
        existing = [path for path in paths.values() if path.exists()]
        stale = [
            path for path in paths.values()
            if path.with_name(path.name + ".uploading").exists()
        ]
        if existing or stale:
            raise SilverPublicationError(
                "Silver V3 artifacts already exist or have stale upload files"
            )

        rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        news_ids: set[str] = set()
        domain_counts: Counter[str] = Counter()
        source_v2_name = self.source.data_path.name
        for line_number, source_row in enumerate(self.source.iter_records(), 1):
            record_id = source_row.get("id")
            text = source_row.get("text")
            domain = source_row.get("domain")
            if (
                not isinstance(record_id, str)
                or not record_id.strip()
                or not isinstance(text, str)
                or not text.strip()
                or not isinstance(domain, str)
                or not isinstance(source_row.get("quality_flags"), list)
            ):
                raise ValueError(f"Invalid Silver V2 record at line {line_number}")
            if source_row.get("text_length") != len(text):
                raise ValueError(f"Incorrect text_length in Silver V2 at line {line_number}")
            if any(not isinstance(flag, str) for flag in source_row["quality_flags"]):
                raise ValueError(f"Invalid quality_flags in Silver V2 at line {line_number}")
            target = domain
            if domain == "News":
                news_ids.add(record_id)
                decision = decisions.get(record_id)
                if decision is None:
                    raise ValueError(f"News record is missing an approved decision: {record_id}")
                target = decision["label"]
                audit_rows.append({
                    "id": record_id,
                    "source_domain": domain,
                    "target_domain": target,
                    "reviewer": decision["reviewer"],
                    "reason": decision["reason"],
                })
            elif domain not in get_v3_domain_labels():
                raise ValueError(f"Silver V2 domain cannot map to V3: {domain!r}")

            flags = list(source_row["quality_flags"])
            if domain == "News":
                if "silver_v3:news_relabelled" not in flags:
                    flags.append("silver_v3:news_relabelled")
            row = {
                "id": record_id,
                "text": text,
                "domain": target,
                "text_length": len(text),
                "source_snapshot": source_row.get("source_snapshot", source_v2_name),
                "quality_flags": flags,
            }
            rows.append(row)
            domain_counts[target] += 1

        extra_ids = set(decisions) - news_ids
        if extra_ids:
            raise ValueError(f"Decisions include non-News or unknown IDs: {len(extra_ids)}")
        if news_ids != set(decisions):
            raise ValueError("Every Silver V2 News record must have exactly one approved decision")
        if not rows:
            raise ValueError("Silver V2 contains no accepted records")

        source_report_sha256 = _sha256_file(self.source.report_path)
        created = datetime.now(timezone.utc).isoformat()
        source = {
            "silver_v2_file": source_v2_name,
            "silver_v2_report": self.source.report_path.name,
            "silver_v2_sha256": source_verified["sha256"],
            "silver_v2_report_sha256": source_report_sha256,
            "silver_v1_report": self.source.report["source_report"],
            "silver_v1_sha256": self.source.report["source_silver_sha256"],
            "decisions_file": paths["decisions"].name,
            "decisions_sha256": decisions_sha256,
        }

        with tempfile.TemporaryDirectory(prefix="vtc_silver_v3_") as temp:
            local = {name: Path(temp) / path.name for name, path in paths.items()}
            with local["silver"].open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            with local["audit"].open("w", encoding="utf-8", newline="\n") as handle:
                for row in audit_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            shutil.copyfile(self.decisions_file, local["decisions"])
            if _sha256_file(local["decisions"]) != decisions_sha256:
                raise SilverPublicationError("Decision file changed while publishing Silver V3")
            report = {
                "pipeline_version": "v3",
                "ruleset_version": "silver-v3-reviewed-labels-v1",
                "source": source,
                "source_rows": source_verified["records"],
                "input_rows": len(rows),
                "accepted_rows": len(rows),
                "rejected_rows": 0,
                "relabeled_rows": len(news_ids),
                "accepted_domain_counts": dict(sorted(domain_counts.items())),
                "output_files": {name: path.name for name, path in paths.items()},
                "silver_sha256": _sha256_file(local["silver"]),
                "audit_sha256": _sha256_file(local["audit"]),
                "decisions_sha256": decisions_sha256,
                "created_utc": created,
                "limitations": (
                    "News labels were changed only from complete, human-approved decisions. "
                    "Silver V2 texts, IDs, non-News labels, and record membership are preserved."
                ),
            }
            _write_json(local["report"], report)
            published: list[Path] = []
            try:
                for name in ("silver", "audit", "decisions", "report"):
                    _publish(local[name], paths[name])
                    published.append(paths[name])
            except Exception:
                for path in reversed(published):
                    path.unlink(missing_ok=True)
                raise

        return {
            "paths": {name: str(path) for name, path in paths.items()},
            "report": report,
        }

    @staticmethod
    def verify_reader(reader: SilverReader) -> dict[str, Any]:
        """Verify published V3 artifacts using a reader addressed by its V3 report."""
        from vietnamese_topic_classification.data.gold.v2 import GoldV2Pipeline

        report = reader.report
        if report.get("pipeline_version") != "v3":
            raise SilverReportError("Expected a Silver V3 quality report")
        if reader.report_path.parent != reader.root / "04_reports" / "silver" / "v3":
            raise SilverReportError("Silver V3 report must be under 04_reports/silver/v3")
        verified = reader.verify()
        for path in (reader.data_path, reader.report_path):
            if path.with_name(path.name + ".uploading").exists():
                raise SilverReportError(f"Unfinished Silver V3 upload: {path}")
        outputs = report.get("output_files")
        if not isinstance(outputs, dict) or set(outputs) != {
            "silver", "audit", "decisions", "report"
        }:
            raise SilverReportError("Silver V3 report must declare all publication files")
        if outputs["silver"] != reader.data_path.name or outputs["report"] != reader.report_path.name:
            raise SilverReportError("Silver V3 publication filenames are inconsistent")
        if report.get("accepted_rows") != verified["records"]:
            raise SilverReportError("Silver V3 accepted row count mismatch")
        if report.get("input_rows") != report.get("accepted_rows"):
            raise SilverReportError("Silver V3 must preserve all accepted source rows")

        for key in ("audit", "decisions"):
            name = outputs[key]
            if not isinstance(name, str) or Path(name).name != name or "\\" in name:
                raise SilverReportError(f"Invalid Silver V3 {key} filename")
            path = reader.root / "04_reports" / "silver" / "v3" / name
            expected_hash = report.get(f"{key}_sha256")
            if (
                not isinstance(expected_hash, str)
                or not path.is_file()
                or _sha256_file(path) != expected_hash
                or path.with_name(path.name + ".uploading").exists()
            ):
                raise SilverReportError(f"Silver V3 {key} is missing or has a checksum mismatch")
        audit_path = reader.root / "04_reports" / "silver" / "v3" / outputs["audit"]
        with audit_path.open("rb") as handle:
            audit_rows = sum(
                chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b"")
            )
        if audit_rows != report.get("relabeled_rows"):
            raise SilverReportError("Silver V3 audit count differs from relabeled row count")

        source = report.get("source")
        if not isinstance(source, dict):
            raise SilverReportError("Silver V3 report is missing source lineage")
        source_name = source.get("silver_v2_report")
        if not isinstance(source_name, str) or Path(source_name).name != source_name:
            raise SilverReportError("Silver V3 has an invalid Silver V2 source report")
        source_reader = SilverReader(reader.root, source_name)
        source_verified = GoldV2Pipeline(source_reader).verify_silver_publication()
        if (
            source_verified["sha256"] != source.get("silver_v2_sha256")
            or _sha256_file(source_reader.report_path) != source.get("silver_v2_report_sha256")
            or source_reader.data_path.name != source.get("silver_v2_file")
            or source_reader.report.get("source_report") != source.get("silver_v1_report")
            or source_reader.report.get("source_silver_sha256") != source.get("silver_v1_sha256")
        ):
            raise SilverReportError("Silver V3 Silver V2 lineage does not match verified source")
        if source_verified["records"] != report.get("source_rows"):
            raise SilverReportError("Silver V3 source row count mismatch")
        if report.get("input_rows") != source_verified["records"]:
            raise SilverReportError("Silver V3 input count differs from Silver V2")
        return verified
