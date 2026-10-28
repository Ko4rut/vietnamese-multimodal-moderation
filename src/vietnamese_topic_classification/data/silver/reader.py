"""Read an immutable Silver JSONL snapshot addressed by its quality report."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

from vietnamese_topic_classification.data.base_reader import BaseReader

from .exceptions import (
    SilverException,
    SilverFileNotFoundError,
    SilverFormatError,
    SilverIntegrityError,
    SilverReportError,
)


class SilverReader(BaseReader):
    """Read a verified Silver snapshot from a mounted or synchronized Data Lake.

    The quality report pins the accepted JSONL filename, checksum and row count.
    This reader does not mutate data, normalize text or access Google Drive APIs.
    """

    def __init__(self, data_lake_root: Path, report_name: str) -> None:
        self.root = Path(data_lake_root).expanduser()
        if (
            not isinstance(report_name, str)
            or not report_name.endswith(".json")
            or Path(report_name).name != report_name
            or "\\" in report_name
        ):
            raise SilverReportError("report_name must be a JSON filename, not a path")

        self.report_path = self.root / "05_reports" / report_name
        try:
            with self.report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SilverReportError(f"Cannot read Silver quality report: {self.report_path}") from exc

        if not isinstance(report, dict):
            raise SilverReportError("Silver quality report must be a JSON object")
        required = ("output_files", "silver_sha256", "accepted_rows")
        missing = [key for key in required if key not in report]
        if missing:
            raise SilverReportError(f"Silver quality report missing keys: {', '.join(missing)}")

        output_files = report["output_files"]
        file_name = output_files.get("silver") if isinstance(output_files, dict) else None
        if (
            not isinstance(file_name, str)
            or not file_name.endswith(".jsonl")
            or Path(file_name).name != file_name
            or "\\" in file_name
        ):
            raise SilverReportError("Invalid Silver filename in quality report")
        if not isinstance(report["silver_sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", report["silver_sha256"]
        ):
            raise SilverReportError("Invalid silver_sha256 in quality report")
        if type(report["accepted_rows"]) is not int or report["accepted_rows"] < 0:
            raise SilverReportError("Invalid accepted_rows in quality report")

        self.report = report
        self.data_path = self.root / "02_silver" / file_name
        if not self.data_path.is_file():
            raise SilverFileNotFoundError(f"Silver snapshot not found: {self.data_path}")

    def verify(self) -> dict[str, Any]:
        """Compare SHA-256 and newline-delimited record count against the report."""
        digest = hashlib.sha256()
        count = 0
        try:
            with self.data_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    count += chunk.count(b"\n")
        except OSError as exc:
            raise SilverIntegrityError(f"Cannot read Silver snapshot: {self.data_path}") from exc

        actual_sha = digest.hexdigest()
        if actual_sha != self.report["silver_sha256"]:
            raise SilverIntegrityError(
                f"Silver SHA256 mismatch: expected={self.report['silver_sha256']}, actual={actual_sha}"
            )
        if count != self.report["accepted_rows"]:
            raise SilverIntegrityError(
                f"Silver record count mismatch: expected={self.report['accepted_rows']}, actual={count}"
            )
        return {"valid": True, "records": count, "sha256": actual_sha}

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Stream JSON objects from Silver without changing their contents."""
        try:
            with self.data_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SilverFormatError(
                            f"Invalid JSONL record at line {line_number}: {self.data_path}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise SilverFormatError(
                            f"Expected JSON object at line {line_number}: {self.data_path}"
                        )
                    yield record
        except FileNotFoundError as exc:
            raise SilverFileNotFoundError(f"Silver snapshot not found: {self.data_path}") from exc
        except UnicodeError as exc:
            raise SilverFormatError(f"Invalid UTF-8 in Silver snapshot: {self.data_path}") from exc
        except OSError as exc:
            raise SilverException(f"Cannot read Silver snapshot: {self.data_path}") from exc
