"""Typed external API contracts; internal model paths are intentionally excluded."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClassifierName = Literal["logistic_regression", "linear_svc"]
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


class ModelSummary(BaseModel):
    id: str
    model_name: str
    classifier: ClassifierName
    validation_macro_f1: float | None
    sklearn_version: str | None


class ModelCatalog(BaseModel):
    models: list[ModelSummary]
    default_model_id: str | None


class ModelStatus(BaseModel):
    ready: bool
    message: str
    labels_count: int
    model_id: str | None
    model_name: str | None
    classifier: ClassifierName | None
    validation_macro_f1: float | None


class HealthStatus(BaseModel):
    status: Literal["ok"]
