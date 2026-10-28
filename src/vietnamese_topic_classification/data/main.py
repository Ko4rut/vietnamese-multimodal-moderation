"""Smoke test: read and verify one existing Bronze snapshot."""

from vietnamese_topic_classification.config.settings import DataLakeConfig
from vietnamese_topic_classification.data.bronze import BronzeReader
from vietnamese_topic_classification.data.silver import SilverReader
from vietnamese_topic_classification.data.base_reader import BaseReader
MANIFEST_NAME = "source_manifest_b81fcce589_seed42_buf10000_n50000.json"
REPORT_NAME = "silver_quality_v1_b81fcce589_seed42_buf10000_n50000.json"

config = DataLakeConfig.from_env()

def reader_test(reader: BaseReader)->None:
    pass


def bronze_reader_test():
    reader = BronzeReader(drive_root=config.root, manifest_name=MANIFEST_NAME)

    verify_rs = reader.verify()
    print("BRONZE VALIDATION PASSED")
    print("Records:", verify_rs["records"])
    print("SHA256:", verify_rs["sha256"])

    first = next(reader.iter_records())
    print("First record ID:", first.get("id"))
    print("First record content: ", first.get("text"))
    print("First record domain:", first.get("domain"))
    
def silver_reader_test()->None:
    reader = SilverReader(data_lake_root=config.root, report_name=REPORT_NAME)
    
    verify_rs = reader.verify()
    print("BRONZE VALIDATION PASSED")
    print("Records:", verify_rs["records"])
    print("SHA256:", verify_rs["sha256"])
    
    first = next(reader.iter_records())
    print("First record ID:", first.get("id"))
    print("First record content: ", first.get("text"))
    print("First record domain:", first.get("domain"))

def main() -> None:
    # bronze_reader_test()
    silver_reader_test()
    

if __name__ == "__main__":
    main()
