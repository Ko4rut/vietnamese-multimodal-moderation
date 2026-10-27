from .bronze.reader import BronzeReader
from config.settings import DataLakeConfig

root_location = DataLakeConfig.from_env()

bronze_reader = BronzeReader(root_location,"source_manifest_b81fcce589_seed42_buf10000_n50000" )

