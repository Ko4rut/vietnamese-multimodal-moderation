"""Compute authentic ranked predictions with classifier-specific score semantics."""

from __future__ import annotations

import math

from vietnamese_topic_classification.config import get_domain_labels

from .types import LoadedModel


def predict_text(model: LoadedModel, text: str, top_k: int = 5) -> dict[str, object]:
    """Return genuine model rankings; a LinearSVC margin is not a probability."""
    if not (1 <= top_k):
        raise ValueError("top_k is out of range")
    top_k = min(top_k, len(model.classes))
    labels = get_domain_labels()
    predicted_id = int(model.pipeline.predict([text])[0])
    if predicted_id not in model.classes:
        raise ValueError("Prediction outside fitted classes")

    if model.classifier == "logistic_regression":
        values = [float(x) for x in model.pipeline.predict_proba([text])[0]]
        score_kind = "probability"
    else:
        raw = model.pipeline.decision_function([text])[0]
        if len(model.classes) == 2:
            signed_margin = float(raw)
            values = [-signed_margin, signed_margin]
        else:
            values = [float(x) for x in raw]
        score_kind = "decision_margin"

    if len(values) != len(model.classes) or not all(math.isfinite(x) for x in values):
        raise ValueError("Invalid inference scores")
    ranked = sorted(zip(model.classes, values), key=lambda pair: pair[1], reverse=True)
    return {
        "model_id": model.descriptor.model_id,
        "model_name": model.experiment_name,
        "label_id": predicted_id,
        "label": labels[predicted_id],
        "score_kind": score_kind,
        "top_predictions": [
            {"label_id": label_id, "label": labels[label_id], "score": score}
            for label_id, score in ranked[:top_k]
        ],
    }
