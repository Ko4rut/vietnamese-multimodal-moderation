"""Read-only smoke tests for Bronze or Silver through one reader interface."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Any

from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.data.base_reader import BaseReader
from vietnamese_topic_classification.data.bronze import BronzeReader
from vietnamese_topic_classification.data.silver import SilverReader

MANIFEST_NAME = "source_manifest_b81fcce589_seed42_buf10000_n50000.json"
REPORT_NAME = "silver_quality_v1_b81fcce589_seed42_buf10000_n50000.json"


def reader_test(reader: BaseReader, preview_limit: int = 1) -> dict[str, Any]:
    """Verify and preview records without depending on the underlying Data Lake layer."""
    if preview_limit < 0:
        raise ValueError("preview_limit must be nonnegative")
    verified = reader.verify()
    layer = type(reader).__name__.removesuffix("Reader").upper()
    print(f"{layer} VALIDATION PASSED")
    print("Records:", verified["records"])
    print("SHA256:", verified["sha256"])
    for index, record in enumerate(islice(reader.iter_records(), preview_limit), start=1):
        label = "First record" if index == 1 else f"Record {index}"
        print(f"{label} ID:", record.get("id"))
        print(f"{label} content:", record.get("text"))
        print(f"{label} domain:", record.get("domain"))
    return verified


def main(
    layer: str = "bronze",
    metadata_name: str | None = None,
    data_lake_root: Path | None = None,
    preview_limit: int = 1,
) -> dict[str, Any]:
    """Choose a concrete reader; reuse the same verification and preview function."""
    readers: dict[str, type[BaseReader]] = {
        "bronze": BronzeReader,
        "silver": SilverReader,
    }
    if layer not in readers:
        raise ValueError(f"Unknown Data Lake layer: {layer}")
    root = Path(data_lake_root) if data_lake_root is not None else DataLakeConfig.from_env().root
    if metadata_name is None:
        metadata_name = MANIFEST_NAME if layer == "bronze" else REPORT_NAME
    reader = readers[layer](data_lake_root=root, metadata_name=metadata_name)
    return reader_test(reader, preview_limit=preview_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and preview a Data Lake snapshot")
    parser.add_argument("--layer", choices=("bronze", "silver"), default="bronze")
    parser.add_argument("--metadata-name", default=None, help="Manifest or quality report filename")
    parser.add_argument("--data-lake-root", type=Path, default=None, help="Overrides DATA_LAKE_ROOT")
    parser.add_argument("--limit", type=int, default=1, help="Max records to preview")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    main(
        layer=args.layer,
        metadata_name=args.metadata_name,
        data_lake_root=args.data_lake_root,
        preview_limit=args.limit,
    )
