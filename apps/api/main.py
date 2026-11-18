"""HTTP endpoints for trusted, pre-trained Vietnamese topic classifiers."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from vietnamese_topic_classification.config import get_domain_labels

from .model_service import (
    ModelNotFound,
    ModelUnavailable,
    default_model_id,
    get_loaded_model,
    list_models,
    predict_text,
)
from .schemas import (
    HealthStatus,
    ModelCatalog,
    ModelStatus,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(
    title="Vietnamese Topic Classification Inference API",
    version="0.2.0",
    description=(
        "Inference for trusted existing TF-IDF + Logistic Regression / LinearSVC "
        "experiments; this service never trains or accepts model uploads."
    ),
)
# Next.js uses a same-origin /api rewrite; these origins allow direct local API debugging.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health", response_model=HealthStatus)
def health() -> HealthStatus:
    """Process health does not imply that a model has been loaded."""
    return HealthStatus(status="ok")


@app.get("/api/models", response_model=ModelCatalog)
def models() -> ModelCatalog:
    """List explicitly registered experiment IDs (no filesystem paths exposed)."""
    try:
        descriptors = list_models()
        return ModelCatalog(
            models=[descriptor.as_response() for descriptor in descriptors],
            default_model_id=default_model_id(descriptors),
        )
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/status", response_model=ModelStatus)
def status(model_id: str | None = Query(default=None, max_length=128)) -> ModelStatus:
    """Readiness for the selected model; unreachable/missing models do not crash the API."""
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


@app.get("/api/labels")
def labels() -> dict[str, object]:
    return {"labels": [{"id": i, "name": name} for i, name in enumerate(get_domain_labels())]}


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    text = request.text.strip()
    if len(text) < 3:
        raise HTTPException(status_code=422, detail="Vui lòng nhập ít nhất 3 ký tự.")
    try:
        model = get_loaded_model(request.model_id)
    except ModelNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        return PredictResponse.model_validate(predict_text(model, text))
    except (ValueError, TypeError, AttributeError, IndexError) as exc:
        raise HTTPException(
            status_code=500, detail="Model không thể dự đoán văn bản này."
        ) from exc
