"""Compatibility exports for versioned Silver modules."""

from .model import (
    SilverPublishedReport as SilverPublishedReport,
    SilverQualityReport as SilverQualityReport,
    SilverRecord as SilverRecord,
    SilverRejectedRecord as SilverRejectedRecord,
    SilverRunResult as SilverRunResult,
    ValidationResult as ValidationResult,
)
from .pipeline import (
    SILVER_VERSION as SILVER_VERSION,
    SilverPipeline as SilverPipeline,
)
from .transformer import (
    SilverTransformer as SilverTransformer,
)
from .validator import (
    SilverValidator as SilverValidator,
)
