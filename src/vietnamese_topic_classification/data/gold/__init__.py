"""Gold V1: deterministic, leakage-aware ML-ready dataset snapshots."""

from .config import GoldConfig
from .exceptions import (
    GoldError,
    GoldIntegrityError,
    GoldLeakageError,
    GoldManifestError,
    GoldPublicationError,
    GoldValidationError,
)
from .pipeline import GoldPipeline
from .reader import GoldReader
from .splitter import GoldSplitter
from .validator import GoldValidator

__all__ = [
    "GoldConfig",
    "GoldError",
    "GoldIntegrityError",
    "GoldLeakageError",
    "GoldManifestError",
    "GoldPublicationError",
    "GoldValidationError",
    "GoldPipeline",
    "GoldReader",
    "GoldSplitter",
    "GoldValidator",
]
