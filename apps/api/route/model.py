"""Registered models and model readiness HTTP routes."""

from fastapi import APIRouter, HTTPException, Query

from vietnamese_topic_classification.config import get_domain_labels

from ..schema.model import ModelCatalog, ModelStatus
from ..service.loader import get_loaded_model
from ..service.registry import default_model_id, list_models
from ..service.types import ModelUnavailable

router = APIRouter()


@router.get("/models", response_model=ModelCatalog)
def models() -> ModelCatalog:
    """Discover configured experiments without unpickling the model files."""
    try:
        descriptors = list_models()
        return ModelCatalog(
            models=[descriptor.as_response() for descriptor in descriptors],
            default_model_id=default_model_id(descriptors),
        )
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/status", response_model=ModelStatus)
def status(model_id: str | None = Query(default=None, max_length=128)) -> ModelStatus:
    """Readiness of the selected model, preserving offline frontend behavior."""
    try:
        model = get_loaded_model(model_id)
    except ModelUnavailable as exc:
        return ModelStatus(
            ready=False,
            message=str(exc),
            labels_count=len(get_domain_labels()),
            model_id=model_id,
            model_name=None,
            classifier=None,
            validation_macro_f1=None,
        )
    return ModelStatus(
        ready=True,
        message="Model đã sẵn sàng.",
        labels_count=len(get_domain_labels()),
        model_id=model.descriptor.model_id,
        model_name=model.experiment_name,
        classifier=model.classifier,
        validation_macro_f1=model.validation_macro_f1,
    )
