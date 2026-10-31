"""Gold V1/V2: deterministic, leakage-aware ML-ready dataset snapshots."""

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
from .v2 import GoldV2Pipeline
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
    "GoldV2Pipeline",
    "GoldReader",
    "GoldSplitter",
    "GoldValidator",
]
