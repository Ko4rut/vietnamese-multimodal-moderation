"""Pydantic contracts for POST /api/predict."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
