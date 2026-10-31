"""Silver V2: corpus-aware boilerplate removal with audit."""
from .cleaner import RULESET_VERSION, SilverV2Cleaner, SilverV2CleanerConfig
from .corpus import CorpusV2Stats, build_corpus_stats, is_ui_candidate, normalize_block
from .model import SilverV2Action, SilverV2AuditRow, SilverV2CleaningDecision, SilverV2Report, SilverV2RunResult
from .pipeline import VERSION, SilverV2Pipeline

__all__ = ["SilverV2Cleaner", "SilverV2CleanerConfig", "SilverV2Pipeline", "SilverV2Action", "SilverV2AuditRow", "SilverV2CleaningDecision", "SilverV2Report", "SilverV2RunResult", "CorpusV2Stats", "build_corpus_stats", "is_ui_candidate", "normalize_block", "RULESET_VERSION", "VERSION"]
