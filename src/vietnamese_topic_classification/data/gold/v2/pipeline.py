"""Build an isolated Gold V2 snapshot from a complete Silver V2 publication."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from vietnamese_topic_classification.data.gold.exceptions import GoldValidationError
from vietnamese_topic_classification.data.gold.model import GoldSource
from vietnamese_topic_classification.data.gold.pipeline import GoldPipeline, _sha256_file
from vietnamese_topic_classification.data.silver import SilverReader


def _safe_jsonl_name(name: object) -> bool:
    return (
        isinstance(name, str)
        and name.endswith(".jsonl")
        and Path(name).name == name
        and "\\" not in name
    )


class GoldV2Pipeline(GoldPipeline):
    """Reuse the V1 group-aware split algorithm without mutating V1 artifacts."""

    version = "v2"

    def __init__(self, reader: SilverReader, **kwargs: object) -> None:
        super().__init__(reader, **kwargs)
        if reader.report.get("pipeline_version") != "v2":
            raise GoldValidationError("Gold V2 requires a Silver V2 quality report")
        if reader.report_path.parent != reader.root / "04_reports" / "silver" / "v2":
            raise GoldValidationError("Silver V2 report must be under 04_reports/silver/v2")

    def verify_silver_publication(self) -> dict[str, object]:
        """Fail closed when the Silver V2 publication/audit is partial or tampered."""
        report = self.reader.report
        verified = self.reader.verify()
        files = report.get("output_files")
        if not isinstance(files, dict) or set(files) != {"silver", "rejected", "audit", "report"}:
            raise GoldValidationError("Silver V2 report must declare all four publication files")
        for key in ("silver", "rejected", "audit"):
            if not _safe_jsonl_name(files.get(key)):
                raise GoldValidationError(f"Invalid Silver V2 {key} filename")
        report_name = files.get("report")
        if (
            not isinstance(report_name, str)
            or not report_name.endswith(".json")
            or Path(report_name).name != report_name
            or "\\" in report_name
            or report_name != self.reader.report_path.name
        ):
            raise GoldValidationError("Silver V2 report filename is not canonical")
        if files["silver"] != self.reader.data_path.name:
            raise GoldValidationError("Silver V2 accepted file differs from quality report")
        expected = {
            "rejected": self.reader.root / "02_silver" / "v2" / files["rejected"],
            "audit": self.reader.root / "04_reports" / "silver" / "v2" / files["audit"],
        }
        for kind, path in expected.items():
            digest = report.get(f"{kind}_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise GoldValidationError(f"Silver V2 missing valid {kind} checksum")
            if not path.is_file() or path.with_name(path.name + ".uploading").exists():
                raise GoldValidationError(f"Partial Silver V2 publication: {kind}: {path}")
            if _sha256_file(path) != digest:
                raise GoldValidationError(f"Silver V2 {kind} SHA256 mismatch: {path}")
        for path in (*expected.values(), self.reader.data_path, self.reader.report_path):
            if path.with_name(path.name + ".uploading").exists():
                raise GoldValidationError(f"Unfinished Silver V2 upload: {path}")
        input_rows = report.get("input_rows")
        accepted_rows = report.get("accepted_rows")
        rejected_rows = report.get("rejected_rows")
        if any(type(n) is not int or n < 0 for n in (input_rows, accepted_rows, rejected_rows)):
            raise GoldValidationError("Invalid Silver V2 publication row counts")
        if input_rows != accepted_rows + rejected_rows:
            raise GoldValidationError("Silver V2 accepted/rejected rows do not sum to input")
        if accepted_rows != verified["records"]:
            raise GoldValidationError("Silver V2 accepted rows disagree with verified JSONL")
        with expected["rejected"].open("rb") as handle:
            rejected_count = sum(chunk.count(b"\n") for chunk in iter(
                lambda: handle.read(1024 * 1024), b""
            ))
        if rejected_count != rejected_rows:
            raise GoldValidationError("Silver V2 rejected row count mismatch")
        source_name = report.get("source_report")
        source_sha = report.get("source_silver_sha256")
        if (
            not isinstance(source_name, str)
            or not source_name.endswith(".json")
            or Path(source_name).name != source_name
            or "\\" in source_name
            or not isinstance(source_sha, str)
            or not re.fullmatch(r"[a-f0-9]{64}", source_sha)
        ):
            raise GoldValidationError("Silver V2 missing valid Silver V1 lineage")
        source_reader = SilverReader(self.reader.root, source_name)
        if source_reader.report.get("pipeline_version") not in (None, "v1"):
            raise GoldValidationError("Silver V2 source report must be Silver V1")
        source_verified = source_reader.verify()
        if source_verified["sha256"] != source_sha:
            raise GoldValidationError("Silver V2 source SHA256 differs from verified Silver V1")
        if report.get("source_rows") != source_verified["records"]:
            raise GoldValidationError("Silver V2 source row count differs from Silver V1")
        if input_rows != source_verified["records"]:
            raise GoldValidationError("Silver V2 input rows disagree with Silver V1")
        return verified

    def _source_metadata(self, verified: dict[str, object]) -> GoldSource:
        source = super()._source_metadata(verified)
        source["silver_version"] = "v2"
        source["silver_report_sha256"] = _sha256_file(self.reader.report_path)
        source["silver_v1_report"] = self.reader.report["source_report"]
        source["silver_v1_sha256"] = self.reader.report["source_silver_sha256"]
        return source

    def run(self):
        self.verify_silver_publication()
        return super().run()
