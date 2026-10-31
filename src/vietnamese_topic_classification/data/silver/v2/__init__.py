"""Exports for versioned Silver modules."""

from .cleaner import RULESET_VERSION, SilverV2Cleaner, SilverV2CleanerConfig
from .corpus import (
    CorpusV2Stats,
    build_corpus_stats,
    is_ui_candidate,
    normalize_block,
)
from .model import (
    SilverV2Action,
    SilverV2AuditRow,
    SilverV2CleaningDecision,
    SilverV2Report,
    SilverV2RunResult,
)
from .pipeline import VERSION, SilverV2Pipeline

__all__ = [
    "RULESET_VERSION",
    "SilverV2Cleaner",
    "SilverV2CleanerConfig",
    "CorpusV2Stats",
    "build_corpus_stats",
    "is_ui_candidate",
    "normalize_block",
    "SilverV2Action",
    "SilverV2AuditRow",
    "SilverV2CleaningDecision",
    "SilverV2Report",
    "SilverV2RunResult",
    "SilverV2Pipeline",
    "VERSION",
]
