"""Create an immutable Gold V1 snapshot from a verified Silver quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.data.gold import GoldConfig, GoldPipeline
from vietnamese_topic_classification.data.silver import SilverReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gold V1 train/validation/test")
    parser.add_argument("--report-name", required=True, help="Existing Silver quality report JSON name")
    parser.add_argument("--data-lake-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--max-candidates", type=int, default=2000)
    args = parser.parse_args()

    root = args.data_lake_root or DataLakeConfig.from_env().root
    reader = SilverReader(data_lake_root=root, metadata_name=args.report_name)
    config = GoldConfig(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        similarity_threshold=args.similarity_threshold,
        max_candidates_per_record=args.max_candidates,
    )
    result = GoldPipeline(reader, config=config).run()
    print(json.dumps(
        {"paths": result["paths"], "split_rows": result["report"]["split_rows"],
         "near_duplicate_audit": result["report"]["near_duplicate_audit"]},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
