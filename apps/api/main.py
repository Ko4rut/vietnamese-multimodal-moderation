"""HTTP adapter for text-only, single-label Vietnamese topic inference."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vietnamese_topic_classification.config import get_domain_labels

from .model_service import ModelUnavailable, get_loaded_model, predict_text

app = FastAPI(
    title="Vietnamese Topic Classification Demo API",
    version="0.1.0",
    description="Text-only inference with a trusted, pre-trained scikit-learn experiment.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10000)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, object]:
    """A missing artifact must not prevent the API or the web app from starting."""
    try:
        model = get_loaded_model()
    except ModelUnavailable as exc:
        return {
            "ready": False,
            "message": str(exc),
            "labels_count": len(get_domain_labels()),
            "model_name": None,
            "classifier": None,
            "validation_macro_f1": None,
        }
    return {
        "ready": True,
        "message": "Model đã sẵn sàng.",
        "labels_count": len(get_domain_labels()),
        "model_name": model.experiment_name,
        "classifier": model.classifier,
        "validation_macro_f1": model.validation_macro_f1,
    }


@app.get("/api/labels")
def labels() -> dict[str, object]:
    return {
        "labels": [
            {"id": idx, "name": name} for idx, name in enumerate(get_domain_labels())
        ]
    }


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict[str, object]:
    text = request.text.strip()
    if len(text) < 3:
        raise HTTPException(status_code=422, detail="Vui lòng nhập ít nhất 3 ký tự.")
    try:
        model = get_loaded_model()
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        return predict_text(model, text)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=500, detail="Model không thể dự đoán văn bản này.") from exc
