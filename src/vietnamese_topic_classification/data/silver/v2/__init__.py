"""Compatibility exports for versioned Silver modules."""

from .cleaner import (
    RULESET_VERSION as RULESET_VERSION,
    SilverV2Cleaner as SilverV2Cleaner,
    SilverV2CleanerConfig as SilverV2CleanerConfig,
)
from .corpus import (
    CorpusV2Stats as CorpusV2Stats,
    build_corpus_stats as build_corpus_stats,
    is_ui_candidate as is_ui_candidate,
    normalize_block as normalize_block,
)
from .model import (
    SilverV2Action as SilverV2Action,
    SilverV2AuditRow as SilverV2AuditRow,
    SilverV2CleaningDecision as SilverV2CleaningDecision,
    SilverV2Report as SilverV2Report,
    SilverV2RunResult as SilverV2RunResult,
)
from .pipeline import (
    SilverV2Pipeline as SilverV2Pipeline,
    VERSION as VERSION,
)
