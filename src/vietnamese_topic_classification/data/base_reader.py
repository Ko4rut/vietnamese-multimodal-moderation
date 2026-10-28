"""Common interface and configuration for Data Lake snapshot readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


class BaseReader(ABC):
    """Share a stable constructor and read contract across Data Lake layers."""

    def __init__(self, data_lake_root: Path, metadata_name: str) -> None:
        self.root = Path(data_lake_root).expanduser()
        self.metadata_name = metadata_name

    @staticmethod
    def is_safe_metadata_name(name: str) -> bool:
        """Accept one JSON filename, never an absolute or traversing path."""
        return (
            isinstance(name, str)
            and name.endswith(".json")
            and Path(name).name == name
            and "\\" not in name
        )

    @abstractmethod
    def verify(self) -> dict[str, Any]:
        """Verify the integrity of a dataset snapshot."""
        ...

    @abstractmethod
    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Lazily yield records from a dataset snapshot."""
        ...
