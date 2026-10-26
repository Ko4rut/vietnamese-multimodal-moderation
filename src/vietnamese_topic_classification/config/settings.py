"""Optional runtime settings; no Drive paths or credentials are hardcoded."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path | None = None


def get_settings() -> Settings:
    raw_path = os.getenv("VTC_DATA_ROOT", "").strip()
    return Settings(data_root=Path(raw_path).expanduser() if raw_path else None)
