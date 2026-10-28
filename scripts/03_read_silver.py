"""Verify and preview an existing Silver JSONL snapshot (read-only)."""

import argparse
import json
from itertools import islice
from pathlib import Path

from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.data.silver import SilverReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and preview one Silver V1 snapshot")
    parser.add_argument("--report-name", required=True, help="Silver quality report filename")
    parser.add_argument("--data-lake-root", type=Path, default=None, help="Overrides DATA_LAKE_ROOT")
    parser.add_argument("--limit", type=int, default=3, help="Max rows to preview (default: 3)")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    root = args.data_lake_root if args.data_lake_root is not None else DataLakeConfig.from_env().root
    reader = SilverReader(data_lake_root=root, metadata_name=args.report_name)
    result = reader.verify()
    print(json.dumps({"verification": result, "snapshot": reader.data_path.name}, ensure_ascii=False))
    for record in islice(reader.iter_records(), args.limit):
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
