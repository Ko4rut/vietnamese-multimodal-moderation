"""Offline contract tests for the real inference HTTP adapter."""

import json

import joblib
import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from vietnamese_topic_classification.training import ClassifierConfig, ExperimentConfig
from vietnamese_topic_classification.training.factory import build_pipeline

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_local_models(tmp_path, monkeypatch):
    """Offline tests should not load real joblib files from a developer's models/."""
    monkeypatch.delenv("VTC_MODELS_ROOT", raising=False)
    monkeypatch.delenv("VTC_DEFAULT_MODEL", raising=False)
    monkeypatch.setattr(
        "apps.api.service.registry._default_models_root", lambda: tmp_path / "models"
    )



def test_missing_model_is_explicit(monkeypatch):
    monkeypatch.delenv("VTC_MODEL_DIR", raising=False)
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["labels_count"] == 25
    assert client.post("/api/predict", json={"text": "Bóng đá Việt Nam"}).status_code == 503


def test_input_validation(monkeypatch):
    monkeypatch.delenv("VTC_MODEL_DIR", raising=False)
    assert client.post("/api/predict", json={"text": "   "}).status_code == 422
    assert client.post("/api/predict", json={"text": "x" * 10001}).status_code == 422


@pytest.mark.parametrize(
    ("classifier", "expected_kind"),
    [("logistic_regression", "probability"), ("linear_svc", "decision_margin")],
)
def test_real_model_inference(tmp_path, monkeypatch, classifier, expected_kind):
    config = ExperimentConfig(
        name="test_demo", gold_manifest_name="gold.json",
        classifier=ClassifierConfig(kind=classifier),
    )
    pipeline = build_pipeline(config)
    pipeline.set_params(tfidf__min_df=1)
    pipeline.fit(
        [
            "sách văn học mới", "tiểu thuyết văn học", "thơ văn học",
            "đội tuyển bóng đá", "trận đấu bóng đá", "thể thao bóng đá",
            "xe ô tô mới", "động cơ ô tô", "lái xe ô tô",
        ],
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    artifact = tmp_path / classifier
    artifact.mkdir()
    joblib.dump(pipeline, artifact / "model.joblib")
    (artifact / "metadata.json").write_text(
        json.dumps({"experiment_name": "test_demo", "validation_metrics": {"macro_f1": 0.6}})
    )
    (artifact / "config.json").write_text(
        json.dumps({"classifier": {"kind": classifier}})
    )
    monkeypatch.setenv("VTC_MODEL_DIR", str(artifact))
    status = client.get("/api/status").json()
    assert status["ready"] is True
    assert status["model_name"] == "test_demo"
    assert status["validation_macro_f1"] == 0.6
    result = client.post("/api/predict", json={"text": "bóng đá và thể thao"})
    assert result.status_code == 200
    data = result.json()
    assert data["score_kind"] == expected_kind
    assert data["label_id"] in (0, 1, 2)
    assert len(data["top_predictions"]) == 3
    assert all("label" in item and "score" in item for item in data["top_predictions"])
    assert data["label"] == ["Arts_and_Entertainment", "Autos_and_Vehicles", "Beauty_and_Fitness"][data["label_id"]]
    if classifier == "logistic_regression":
        assert abs(sum(x["score"] for x in data["top_predictions"]) - 1.0) < 1e-6
