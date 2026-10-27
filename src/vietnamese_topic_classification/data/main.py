"""Smoke test: read and verify one existing Bronze snapshot."""

from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.data.bronze import BronzeReader

MANIFEST_NAME = "source_manifest_b81fcce589_seed42_buf10000_n50000.json"


def main() -> None:
    config = DataLakeConfig.from_env()
    reader = BronzeReader(drive_root=config.root, manifest_name=MANIFEST_NAME)

    result = reader.verify()
    print("BRONZE VALIDATION PASSED")
    print("Records:", result["records"])
    print("SHA256:", result["sha256"])

    first = next(reader.iter_records())
    print("First record ID:", first.get("id"))
    print("First record content: ", first.get("text"))
    print("First record domain:", first.get("domain"))


if __name__ == "__main__":
    main()
