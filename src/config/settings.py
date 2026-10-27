
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLakeConfig:
    root: Path

    @classmethod
    def from_env(cls) -> "DataLakeConfig":
        value = os.getenv("DATA_LAKE_ROOT")

        if not value:
            raise ValueError(
                "DATA_LAKE_ROOT environment variable is required"
            )

        root = Path(value).expanduser()

        if not root.is_dir():
            raise FileNotFoundError(
                f"Data Lake directory does not exist: {root}"
            )

        return cls(root=root)
