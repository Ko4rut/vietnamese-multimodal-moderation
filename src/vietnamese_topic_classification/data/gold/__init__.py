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
from .model import (
    GoldCandidateAudit,
    GoldManifest,
    GoldPipelineResult,
    GoldQualityReport,
    GoldRecord,
    GoldSource,
    GoldSplitAudit,
    GoldSplitFile,
)
from .pipeline import GoldPipeline
from .reader import GoldReader
from .splitter import GoldSplitter
from .validator import GoldValidator

__all__ = [
    "GoldRecord",
    "GoldManifest",
    "GoldSource",
    "GoldSplitFile",
    "GoldCandidateAudit",
    "GoldSplitAudit",
    "GoldQualityReport",
    "GoldPipelineResult",
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
