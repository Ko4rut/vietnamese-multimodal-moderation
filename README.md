# Vietnamese Multimodal Moderation

Research scaffold for Vietnamese community-safety moderation across **text, images, and video**.

> This PR establishes contracts, configuration, and tests only. It does **not** ship trained models or real moderation decisions.

## Quick start

Requires Python 3.10+.

```bash
python -m pip install -e ".[dev]"
pytest
python -m moderation --help
```

## Architecture

- `src/moderation/config/`: versioned labels and application settings bundled with the package.
- `src/moderation/schemas.py`: shared typed inference/evidence contracts.
- `src/moderation/models/{text,image,video,fusion}/`: future model adapters.
- `src/moderation/{data,training,inference,evaluation,policy}/`: data and application workflows.
- `apps/`: reserved for API and web demo.
- `tests/`: lightweight offline tests.

## Important principles

1. Detection labels are **not** enforcement actions.
2. Unsupported model paths must raise `NotImplementedError`, never invent predictions.
3. Human review and policy-version traceability come before automated enforcement.
4. Do not commit training corpora, credentials, large binaries, or harmful raw media.
5. The taxonomy is a **research proposal**, not official Meta/TikTok policy.

See [architecture](docs/ARCHITECTURE.md) and [implementation plan](docs/IMPLEMENTATION_PLAN.md).
