"""Create Gold V2 from a complete, immutable Silver V2 snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vietnamese_topic_classification.data.gold import (
    GoldConfig,
    GoldReader,
    GoldV2Pipeline,
)
from vietnamese_topic_classification.data.silver import SilverReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gold V2 from verified Silver V2")
    parser.add_argument("--data-lake-root", type=Path, required=True)
    parser.add_argument("--report-name", required=True, help="Silver V2 report filename")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--max-candidates", type=int, default=2000)
    args = parser.parse_args()
    if Path(args.report_name).name != args.report_name:
        parser.error("--report-name must be a JSON filename, not a path")
    reader = SilverReader(args.data_lake_root, args.report_name)
    pipeline = GoldV2Pipeline(
        reader,
        config=GoldConfig(
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            similarity_threshold=args.similarity_threshold,
            max_candidates_per_record=args.max_candidates,
        ),
    )
    verified = pipeline.verify_silver_publication()
    planned = pipeline.planned_paths(verified["sha256"])
    stale = [
        str(path.with_name(path.name + ".uploading"))
        for path in planned.values()
        if path.with_name(path.name + ".uploading").exists()
    ]
    existing = [str(path) for path in planned.values() if path.exists()]
    if stale:
        raise FileExistsError(f"Stale Gold V2 uploading artifacts: {stale}")
    if existing:
        if len(existing) != len(planned):
            raise FileExistsError(f"Partial Gold V2 publication: {existing}")
        gold_reader = GoldReader(args.data_lake_root, planned["manifest"].name)
        gold_reader.verify()
        lineage = gold_reader.manifest["source"]
        if (
            lineage["silver_sha256"] != verified["sha256"]
            or lineage.get("silver_report_sha256") != pipeline._source_metadata(
                verified
            )["silver_report_sha256"]
        ):
            raise ValueError("Gold V2 lineage mismatch")
        print(json.dumps({
            "status": "reused",
            "paths": {name: str(path) for name, path in planned.items()},
        }, ensure_ascii=False, indent=2))
        return
    result = pipeline.run()
    gold_reader = GoldReader(args.data_lake_root, planned["manifest"].name)
    verification = gold_reader.verify()
    print(json.dumps({
        "status": "published", "paths": result["paths"],
        "split_rows": verification["split_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
