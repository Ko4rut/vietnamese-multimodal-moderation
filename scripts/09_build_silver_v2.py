"""Build Silver V2 from a verified immutable Silver V1 snapshot."""
import argparse
import json
from pathlib import Path

from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.data.silver.cleaner_v2 import (
    SilverV2Cleaner,
    SilverV2CleanerConfig,
)
from vietnamese_topic_classification.data.silver.pipeline_v2 import SilverV2Pipeline
from vietnamese_topic_classification.data.silver.reader import SilverReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditable Silver V2 suffix boilerplate cleaner")
    parser.add_argument("--report-name", required=True, help="Silver V1 quality report filename")
    parser.add_argument("--data-lake-root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Analyze without publishing outputs")
    parser.add_argument("--min-retained-chars", type=int, default=200)
    parser.add_argument("--max-removed-fraction", type=float, default=0.35)
    args = parser.parse_args()
    root = args.data_lake_root or DataLakeConfig.from_env().root
    cleaner = SilverV2Cleaner(
        SilverV2CleanerConfig(
            min_retained_chars=args.min_retained_chars,
            max_removed_fraction=args.max_removed_fraction,
        )
    )
    result = SilverV2Pipeline(
        SilverReader(root, args.report_name), cleaner=cleaner
    ).run(dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
