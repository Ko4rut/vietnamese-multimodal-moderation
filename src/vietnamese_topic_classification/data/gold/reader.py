"""Read and verify a pinned Gold V1 train/validation/test snapshot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, cast

from vietnamese_topic_classification.config import get_domain_labels
from vietnamese_topic_classification.data.base_reader import BaseReader

from .exceptions import GoldIntegrityError, GoldManifestError
from .model import GoldManifest
from .splitter import SPLITS


def _safe_file(name: Any, extension: str) -> bool:
    return (
        isinstance(name, str)
        and name.endswith(extension)
        and Path(name).name == name
        and "\\" not in name
    )


class GoldReader(BaseReader):
    """The shared BaseReader interface; iter_records yields all three splits in order."""

    def __init__(self, data_lake_root: Path, metadata_name: str) -> None:
        if not self.is_safe_metadata_name(metadata_name):
            raise GoldManifestError("metadata_name must be a JSON filename, not a path")
        super().__init__(data_lake_root, metadata_name)
        self.manifest_path = self.root / "03_gold" / "manifests" / metadata_name
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GoldManifestError(f"Cannot read Gold manifest: {self.manifest_path}") from exc
        if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
            raise GoldManifestError("Gold manifest must be a format_version=1 JSON object")
        expected_mapping = {label: index for index, label in enumerate(get_domain_labels())}
        if manifest.get("label_to_id") != expected_mapping:
            raise GoldManifestError("Gold manifest label mapping differs from domains_v1")
        if type(manifest.get("total_rows")) is not int or manifest["total_rows"] < 1:
            raise GoldManifestError("Gold manifest total_rows must be positive")
        entries = manifest.get("files")
        if not isinstance(entries, dict) or set(entries) != set(SPLITS):
            raise GoldManifestError("Gold manifest must declare all train/validation/test files")
        paths = {}
        for split in SPLITS:
            entry = entries[split]
            if not isinstance(entry, dict) or not _safe_file(entry.get("file_name"), ".jsonl"):
                raise GoldManifestError(f"Invalid Gold file entry for {split}")
            if type(entry.get("rows")) is not int or entry["rows"] < 0:
                raise GoldManifestError(f"Invalid Gold row count for {split}")
            if not isinstance(entry.get("sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", entry["sha256"]
            ):
                raise GoldManifestError(f"Invalid Gold checksum for {split}")
            paths[split] = self.root / "03_gold" / entry["file_name"]
            if not paths[split].is_file():
                raise GoldIntegrityError(f"Gold split missing: {paths[split]}")
        if sum(entries[name]["rows"] for name in SPLITS) != manifest["total_rows"]:
            raise GoldManifestError("Gold manifest split counts do not sum to total_rows")
        if len({path.name for path in paths.values()}) != len(paths):
            raise GoldManifestError("Gold files must have distinct filenames")
        self.manifest: GoldManifest = cast(GoldManifest, manifest)
        self.data_paths = paths

    def verify(self) -> dict[str, Any]:
        counts = {}
        for split in SPLITS:
            path = self.data_paths[split]
            digest = hashlib.sha256()
            rows = 0
            try:
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                        rows += block.count(b"\n")
            except OSError as exc:
                raise GoldIntegrityError(f"Cannot read Gold split: {path}") from exc
            spec = self.manifest["files"][split]
            if digest.hexdigest() != spec["sha256"]:
                raise GoldIntegrityError(f"Gold {split} SHA256 mismatch: {path}")
            if rows != spec["rows"]:
                raise GoldIntegrityError(f"Gold {split} row count mismatch: {path}")
            counts[split] = rows
        return {"valid": True, "records": sum(counts.values()), "split_rows": counts}

    def iter_split(self, split: str) -> Iterator[dict[str, Any]]:
        if split not in SPLITS:
            raise ValueError(f"Unknown Gold split {split!r}; expected {SPLITS}")
        path = self.data_paths[split]
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        row = json.loads(line)
                    except (UnicodeError, json.JSONDecodeError) as exc:
                        raise GoldIntegrityError(
                            f"Invalid Gold JSONL at {path}:{line_number}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise GoldIntegrityError(f"Expected Gold JSON object at {path}:{line_number}")
                    if self.manifest["label_to_id"].get(row.get("domain")) != row.get("label_id"):
                        raise GoldIntegrityError(f"Gold label mismatch at {path}:{line_number}")
                    yield row
        except OSError as exc:
            raise GoldIntegrityError(f"Cannot read Gold split: {path}") from exc

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield train, validation, and test sequentially; use iter_split for a single set."""
        for split in SPLITS:
            yield from self.iter_split(split)
