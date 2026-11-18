"""Validate prediction requests and translate inference errors into HTTP status codes."""

from fastapi import APIRouter, HTTPException

from ..schema.prediction import PredictRequest, PredictResponse
from ..service.inference import predict_text
from ..service.loader import get_loaded_model
from ..service.types import ModelNotFound, ModelUnavailable

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
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
