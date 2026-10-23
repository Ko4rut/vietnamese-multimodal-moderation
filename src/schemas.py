"""Framework-independent, typed contracts for moderation evidence and outputs."""

from dataclasses import dataclass, field
from enum import Enum


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class Action(str, Enum):
    ALLOW = "ALLOW"
    RESTRICT = "RESTRICT"
    REMOVE = "REMOVE"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class Evidence:
    source: Modality
    description: str
    start_sec: float | None = None
    end_sec: float | None = None

    def __post_init__(self) -> None:
        if (self.start_sec is None) != (self.end_sec is None):
            raise ValueError("Temporal evidence requires both start_sec and end_sec.")
        if self.start_sec is not None and (
            self.start_sec < 0 or self.end_sec < self.start_sec
        ):
            raise ValueError("Temporal evidence must have a valid nonnegative interval.")


@dataclass(frozen=True)
class LabelScore:
    label_id: str
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")


@dataclass
class ModerationResult:
    content_id: str
    modality: Modality
    policy_version: str
    labels: list[LabelScore] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    recommended_action: Action | None = None
    requires_human_review: bool = True
    # None means that no decision has been made; avoid fabricated defaults.
