"""Runtime configuration for the locally accessible Data Lake root."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class DataLakeConfig:
    """Location of the Data Lake; the caller mounts/synchronizes Drive first."""

    root: Path

    @classmethod
    def from_env(cls) -> "DataLakeConfig":
        # Read .env at the terminal's working directory, normally the repo root.
        # A variable explicitly set in the environment takes precedence.
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

        value = os.getenv("DATA_LAKE_ROOT", "").strip()
        if not value:
            raise ValueError("DATA_LAKE_ROOT environment variable is required")

        root = Path(value).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Data Lake directory does not exist: {root}")

        return cls(root=root)
