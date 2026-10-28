"""Read a pinned Bronze JSONL snapshot from a locally mounted data lake."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

from vietnamese_topic_classification.data.base_reader import BaseReader

from .exceptions import (
    BronzeException,
    BronzeFileNotFoundError,
    BronzeFormatError,
    BronzeIntegrityError,
    BronzeManifestError,
)


class BronzeReader(BaseReader):
    """Read and verify one manifest-addressed Bronze snapshot without modifying it.

    data_lake_root is a mounted directory, not a Google Drive API credential.
    """

    def __init__(self, data_lake_root: Path, metadata_name: str) -> None:
        if not self.is_safe_metadata_name(metadata_name):
            raise BronzeManifestError("metadata_name must be a JSON filename, not a path")
        super().__init__(data_lake_root, metadata_name)
        self.manifest_path = self.root / "00_source_metadata" / self.metadata_name

        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BronzeManifestError(f"Cannot read Bronze manifest: {self.manifest_path}") from exc

        if not isinstance(manifest, dict):
            raise BronzeManifestError("Bronze manifest must be a JSON object")

        required = ("file_name", "sha256", "sample_size", "file_size_bytes")
        missing = [key for key in required if key not in manifest]
        if missing:
            raise BronzeManifestError(f"Bronze manifest missing keys: {', '.join(missing)}")

        file_name = manifest["file_name"]
        if (
            not isinstance(file_name, str)
            or not file_name.endswith(".jsonl")
            or Path(file_name).name != file_name
            or "\\" in file_name
        ):
            raise BronzeManifestError("Invalid Bronze filename in manifest")
        if not isinstance(manifest["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", manifest["sha256"]
        ):
            raise BronzeManifestError("Invalid sha256 in Bronze manifest")
        if type(manifest["sample_size"]) is not int or manifest["sample_size"] < 1:
            raise BronzeManifestError("Invalid sample_size in Bronze manifest")
        if type(manifest["file_size_bytes"]) is not int or manifest["file_size_bytes"] < 0:
            raise BronzeManifestError("Invalid file_size_bytes in Bronze manifest")

        self.manifest = manifest
        self.data_path = self.root / "01_bronze" / file_name
        if not self.data_path.is_file():
            raise BronzeFileNotFoundError(f"Bronze snapshot not found: {self.data_path}")

    def verify(self) -> dict[str, Any]:
        """Check that file size, SHA-256 and record count match the manifest."""
        digest = hashlib.sha256()
        count = 0
        try:
            actual_size = self.data_path.stat().st_size
            if actual_size != self.manifest["file_size_bytes"]:
                raise BronzeIntegrityError(
                    f"Bronze file size mismatch: expected={self.manifest['file_size_bytes']}, "
                    f"actual={actual_size}"
                )
            with self.data_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    count += chunk.count(b"\n")
        except OSError as exc:
            raise BronzeIntegrityError(f"Cannot read Bronze snapshot: {self.data_path}") from exc

        actual_sha = digest.hexdigest()
        if actual_sha != self.manifest["sha256"]:
            raise BronzeIntegrityError(
                f"Bronze SHA256 mismatch: expected={self.manifest['sha256']}, actual={actual_sha}"
            )
        if count != self.manifest["sample_size"]:
            raise BronzeIntegrityError(
                f"Bronze record count mismatch: expected={self.manifest['sample_size']}, "
                f"actual={count}"
            )
        return {"valid": True, "records": count, "sha256": actual_sha}

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Lazily decode JSONL records; field/schema validation belongs to Silver."""
        try:
            with self.data_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeError) as exc:
                        raise BronzeFormatError(
                            f"Invalid JSONL record at line {line_number}: {self.data_path}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise BronzeFormatError(
                            f"Expected JSON object at line {line_number}: {self.data_path}"
                        )
                    yield record
        except FileNotFoundError as exc:
            raise BronzeFileNotFoundError(f"Bronze snapshot not found: {self.data_path}") from exc
        except OSError as exc:
            raise BronzeException(f"Cannot read Bronze snapshot: {self.data_path}") from exc
