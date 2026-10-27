"""Build Silver from a pinned Bronze manifest on a mounted/synced Data Lake."""

import argparse
import json
from pathlib import Path

from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.data.bronze import BronzeReader
from vietnamese_topic_classification.data.silver import SilverPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VTSNLP Silver V1 from verified Bronze")
    parser.add_argument("--manifest-name", required=True, help="Bronze source manifest filename")
    parser.add_argument("--data-lake-root", type=Path, default=None, help="Overrides DATA_LAKE_ROOT")
    args = parser.parse_args()
    root = args.data_lake_root if args.data_lake_root is not None else DataLakeConfig.from_env().root
    reader = BronzeReader(drive_root=root, manifest_name=args.manifest_name)
    result = SilverPipeline(reader).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
