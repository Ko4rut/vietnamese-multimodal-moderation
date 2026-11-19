"""Pydantic contracts for model discovery and status."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ClassifierName = Literal["logistic_regression", "linear_svc"]

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
