"""Validate prediction requests and translate inference errors into HTTP status codes."""

from fastapi import APIRouter, HTTPException

from ..schema.prediction import (
    CompareRequest,
    CompareResponse,
    ComparisonResult,
    PredictRequest,
    PredictResponse,
)
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


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    """Run the same text through 2-4 registered models, preserving input order.

    A broken or missing model produces a per-model error; other models continue.
    Existing /predict semantics remain unchanged.
    """
    text = request.text.strip()
    if len(text) < 3:
        raise HTTPException(status_code=422, detail="Vui lòng nhập ít nhất 3 ký tự.")

    results: list[ComparisonResult] = []
    for model_id in request.model_ids:
        try:
            model = get_loaded_model(model_id)
            prediction = PredictResponse.model_validate(predict_text(model, text))
        except ModelNotFound:
            results.append(
                ComparisonResult(
                    model_id=model_id,
                    status="error",
                    error="Model không tồn tại trong registry.",
                )
            )
        except ModelUnavailable:
            results.append(
                ComparisonResult(
                    model_id=model_id,
                    status="error",
                    error="Model chưa sẵn sàng hoặc artifact không tương thích.",
                )
            )
        except (ValueError, TypeError, AttributeError, IndexError):
            results.append(
                ComparisonResult(
                    model_id=model_id,
                    status="error",
                    error="Model không thể dự đoán văn bản này.",
                )
            )
        else:
            results.append(
                ComparisonResult(model_id=model_id, status="ok", prediction=prediction)
            )
    return CompareResponse(results=results)
