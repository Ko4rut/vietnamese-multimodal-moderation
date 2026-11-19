"""Pydantic contracts for POST /api/predict."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ScoreKind = Literal["probability", "decision_margin"]


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10000)
    model_id: str | None = Field(default=None, min_length=1, max_length=128)


class RankedPrediction(BaseModel):
    label_id: int
    label: str
    score: float


class PredictResponse(BaseModel):
    model_id: str
    model_name: str
    label_id: int
    label: str
    score_kind: ScoreKind
    top_predictions: list[RankedPrediction]


class CompareRequest(BaseModel):
    """Compare a bounded set of registered experiments on exactly one text."""

    text: str = Field(min_length=3, max_length=10000)
    model_ids: list[str] = Field(min_length=2, max_length=4)

    @field_validator("model_ids")
    @classmethod
    def validate_model_ids(cls, values: list[str]) -> list[str]:
        if any(not 1 <= len(item) <= 128 or item != item.strip() for item in values):
            raise ValueError("model_ids phải là các ID không rỗng, tối đa 128 ký tự.")
        if len(values) != len(set(values)):
            raise ValueError("model_ids không được trùng nhau.")
        return values


class ComparisonResult(BaseModel):
    model_id: str
    status: Literal["ok", "error"]
    prediction: PredictResponse | None = None
    error: str | None = None


class CompareResponse(BaseModel):
    results: list[ComparisonResult]
