"""Explicit boundary for modality-specific inference implementations."""

from abc import ABC, abstractmethod

from moderation.schemas import ModerationResult


class Moderator(ABC):
    @abstractmethod
    def predict(self, content_id: str, payload: object) -> ModerationResult:
        """Run an actual model; must never produce fabricated predictions."""


class UnconfiguredModerator(Moderator):
    def predict(self, content_id: str, payload: object) -> ModerationResult:
        raise NotImplementedError("No trained model is configured for inference yet.")
