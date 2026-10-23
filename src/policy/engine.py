"""Policy engine interface; deliberately does not auto-enforce labels."""

from moderation.schemas import Action, ModerationResult


def decide(result: ModerationResult) -> Action:
    """Require human review until policy rules are validated and implemented."""
    return Action.ESCALATE
