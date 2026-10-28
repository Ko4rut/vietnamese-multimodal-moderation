from abc import ABC, abstractmethod
from typing import Any, Iterator


class BaseReader(ABC):
    """Common contract for Data Lake readers."""

    @abstractmethod
    def verify(self) -> dict[str, Any]:
        """Verify the integrity of a dataset snapshot."""
        ...

    @abstractmethod
    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Lazily yield records from a dataset snapshot."""
        ...
