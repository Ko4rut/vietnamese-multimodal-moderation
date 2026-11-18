"""Typed model descriptors and domain errors, independent of FastAPI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sklearn.pipeline import Pipeline

ClassifierName = Literal["logistic_regression", "linear_svc"]
MODEL_FILES = ("model.joblib", "metadata.json", "config.json")


class ModelUnavailable(RuntimeError):
    """A configured model cannot be served (missing or incompatible artifact)."""


class ModelNotFound(ModelUnavailable):
    """The requested model ID is not registered."""


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    model_id: str
    directory: Path
    classifier: ClassifierName
    validation_macro_f1: float | None
    sklearn_version: str | None

    def as_response(self) -> dict[str, object]:
        return {
            "id": self.model_id,
            "model_name": self.model_id,
            "classifier": self.classifier,
            "validation_macro_f1": self.validation_macro_f1,
            "sklearn_version": self.sklearn_version,
        }


@dataclass(frozen=True, slots=True)
class LoadedModel:
    descriptor: ModelDescriptor
    pipeline: Pipeline
    classes: tuple[int, ...]

    @property
    def experiment_name(self) -> str:
        return self.descriptor.model_id

    @property
    def classifier(self) -> ClassifierName:
        return self.descriptor.classifier

    @property
    def validation_macro_f1(self) -> float | None:
        return self.descriptor.validation_macro_f1
