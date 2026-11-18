"""Process health and canonical label endpoints."""

from fastapi import APIRouter

from vietnamese_topic_classification.config import get_domain_labels

from ..schema.health import HealthStatus

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    """Process liveness; not a check that inference artifacts are loaded."""
    return HealthStatus(status="ok")


@router.get("/labels")
def labels() -> dict[str, object]:
    return {"labels": [{"id": i, "name": name} for i, name in enumerate(get_domain_labels())]}
