"""Runtime settings read from the environment (never store secrets in source)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    log_level: str = "INFO"
    policy_version: str = "mcst_v1"


def get_settings() -> Settings:
    return Settings(
        log_level=os.getenv("MODERATION_LOG_LEVEL", "INFO"),
        policy_version=os.getenv("MODERATION_POLICY_VERSION", "mcst_v1"),
    )
