"""Real HTTP inference tests using small, freshly trained scikit-learn pipelines."""

from __future__ import annotations

import json

import joblib
import pytest
import sklearn
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.service.loader import _load_pipeline
from vietnamese_topic_classification.training import ClassifierConfig, ExperimentConfig
from vietnamese_topic_classification.training.factory import build_pipeline

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("VTC_MODEL_DIR", raising=False)
    monkeypatch.delenv("VTC_MODELS_ROOT", raising=False)
    monkeypatch.delenv("VTC_DEFAULT_MODEL", raising=False)
    # Do not touch a developer's real model files during the test suite.
    monkeypatch.setattr(
        "apps.api.service.registry._default_models_root", lambda: tmp_path / "models"
    )
    _load_pipeline.cache_clear()
    yield
    _load_pipeline.cache_clear()


def make_artifact(directory, classifier, name):
    """Mimic the exact three files published by training/artifacts.py."""
    directory.mkdir(parents=True)
    config = ExperimentConfig(
        name=name,
        gold_manifest_name="gold.json",
        classifier=ClassifierConfig(kind=classifier),
    )
    pipeline = build_pipeline(config)
    pipeline.set_params(tfidf__min_df=1)
    texts = [
        "bóng đá thể thao cầu thủ", "đội bóng đá hôm nay", "trận đấu thể thao",
        "ngân hàng tài chính giao dịch", "chứng khoán thị trường tài chính",
        "lãi suất ngân hàng",
        "khoa học máy tính phần mềm", "máy tính công nghệ điện tử",
        "phần mềm và khoa học máy tính",
    ]
    labels = [23, 23, 23, 6, 6, 6, 5, 5, 5]
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, directory / "model.joblib")
    (directory / "config.json").write_text(
        json.dumps({
            "name": name,
            "features": {"kind": "tfidf"},
            "classifier": {"kind": classifier},
        }),
        encoding="utf-8",
    )
    (directory / "metadata.json").write_text(
        json.dumps({
            "experiment_name": name,
            "sklearn_version": sklearn.__version__,
            "validation_metrics": {"macro_f1": 0.71},
        }),
        encoding="utf-8",
    )
    return directory


@pytest.mark.parametrize("classifier,expected_kind", [
    ("logistic_regression", "probability"),
    ("linear_svc", "decision_margin"),
])
def test_existing_baseline_can_serve_real_http_predictions(
    tmp_path, monkeypatch, classifier, expected_kind
):
    directory = make_artifact(tmp_path / classifier, classifier, "experiment_v1")
    monkeypatch.setenv("VTC_MODEL_DIR", str(directory))

    assert client.get("/api/health").json() == {"status": "ok"}
    catalog = client.get("/api/models")
    assert catalog.status_code == 200
    assert [model["id"] for model in catalog.json()["models"]] == ["experiment_v1"]
    assert catalog.json()["default_model_id"] == "experiment_v1"

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["ready"] is True
    assert status.json()["classifier"] == classifier
    assert status.json()["validation_macro_f1"] == pytest.approx(0.71)

    response = client.post("/api/predict", json={"text": "đội bóng đá thể thao"})
    assert response.status_code == 200
    result = response.json()
    assert result["model_id"] == "experiment_v1"
    assert result["score_kind"] == expected_kind
    assert result["label"] in ["Sports", "Finance", "Computers_and_Electronics"]
    assert len(result["top_predictions"]) == 3
    scores = [item["score"] for item in result["top_predictions"]]
    assert scores == sorted(scores, reverse=True)
    assert len(set(item["label_id"] for item in result["top_predictions"])) == 3
    if expected_kind == "probability":
        assert all(0 <= value <= 1 for value in scores)
    else:
        assert all(isinstance(value, float) for value in scores)


def test_auto_discovery_from_project_models_directory(tmp_path):
    root = tmp_path / "models"
    assert client.get("/api/models").json() == {"models": [], "default_model_id": None}

    name = "tfidf_linear_svc_v2_9ff85ff3a7"
    make_artifact(root / name, "linear_svc", name)

    # The next GET sees the model; no environment variable or API restart needed.
    catalog = client.get("/api/models")
    assert catalog.status_code == 200
    assert [item["id"] for item in catalog.json()["models"]] == [name]
    assert catalog.json()["default_model_id"] == name
    assert client.get("/api/status").json()["ready"] is True
    result = client.post("/api/predict", json={"text": "đội tuyển bóng đá Việt Nam"})
    assert result.status_code == 200
    assert result.json()["model_id"] == name


def test_auto_discovery_skips_invalid_metadata_without_hiding_good_model(tmp_path):
    root = tmp_path / "models"
    make_artifact(root / "good", "logistic_regression", "good_model")
    bad = make_artifact(root / "bad", "linear_svc", "bad_model")
    (bad / "metadata.json").write_text("{invalid JSON", encoding="utf-8")

    catalog = client.get("/api/models")
    assert catalog.status_code == 200
    assert [item["id"] for item in catalog.json()["models"]] == ["good_model"]
    assert client.get("/api/status").json()["ready"] is True


def test_explicit_root_overrides_local_default(tmp_path, monkeypatch):
    make_artifact(tmp_path / "models" / "local", "linear_svc", "local_model")
    external = tmp_path / "external"
    make_artifact(external / "remote", "logistic_regression", "remote_model")
    monkeypatch.setenv("VTC_MODELS_ROOT", str(external))

    catalog = client.get("/api/models").json()
    assert [item["id"] for item in catalog["models"]] == ["remote_model"]


def test_catalog_selects_between_v1_v2_experiments(tmp_path, monkeypatch):
    root = tmp_path / "experiments"
    make_artifact(root / "tfidf_baselines_v1" / "old", "logistic_regression", "logreg_v1")
    make_artifact(root / "tfidf_baselines_v2" / "new", "linear_svc", "svc_v2")
    monkeypatch.setenv("VTC_MODELS_ROOT", str(root))
    monkeypatch.setenv("VTC_DEFAULT_MODEL", "svc_v2")

    data = client.get("/api/models").json()
    assert [model["id"] for model in data["models"]] == ["logreg_v1", "svc_v2"]
    assert data["default_model_id"] == "svc_v2"
    assert client.get("/api/status").json()["classifier"] == "linear_svc"
    assert client.get("/api/status", params={"model_id": "logreg_v1"}).json()["ready"]
    selected = client.post(
        "/api/predict", json={"text": "ngân hàng và lãi suất", "model_id": "logreg_v1"}
    )
    assert selected.status_code == 200
    assert selected.json()["model_id"] == "logreg_v1"
    assert selected.json()["score_kind"] == "probability"
    assert client.post(
        "/api/predict", json={"text": "ngân hàng và lãi suất", "model_id": "svc_v2"}
    ).json()["score_kind"] == "decision_margin"


def test_missing_model_and_invalid_request_do_not_fabricate_inference():
    assert client.get("/api/models").json() == {"models": [], "default_model_id": None}
    assert client.get("/api/status").json()["ready"] is False
    result = client.post("/api/predict", json={"text": "Bóng đá Việt Nam"})
    assert result.status_code == 503
    assert client.post("/api/predict", json={"text": "ab"}).status_code == 422
    assert client.post("/api/predict", json={"text": "   "}).status_code == 422
    assert client.post("/api/predict", json={"text": "a" * 10001}).status_code == 422
    assert len(client.get("/api/labels").json()["labels"]) == 25


def test_unknown_id_is_not_a_filesystem_path(tmp_path, monkeypatch):
    root = tmp_path / "experiments"
    make_artifact(root / "model", "logistic_regression", "safe_id")
    monkeypatch.setenv("VTC_MODELS_ROOT", str(root))
    response = client.post(
        "/api/predict", json={"text": "Văn bản hợp lệ", "model_id": "../../secrets"}
    )
    assert response.status_code == 404
    assert client.get("/api/status", params={"model_id": "unknown"}).json()["ready"] is False


def test_incompatible_sklearn_version_is_rejected_before_loading(tmp_path, monkeypatch):
    root = make_artifact(tmp_path / "model", "logistic_regression", "version_test")
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["sklearn_version"] = "0.15.0"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("VTC_MODEL_DIR", str(root))

    status = client.get("/api/status")
    assert status.json()["ready"] is False
    assert "0.15.0" in status.json()["message"]
    response = client.post("/api/predict", json={"text": "Bóng đá Việt Nam"})
    assert response.status_code == 503


def test_model_config_mismatch_is_rejected(tmp_path, monkeypatch):
    root = make_artifact(tmp_path / "model", "logistic_regression", "identity_test")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["name"] = "other"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("VTC_MODEL_DIR", str(root))
    assert client.get("/api/status").json()["ready"] is False
    assert client.get("/api/models").status_code == 503


def test_explicit_default_must_exist(tmp_path, monkeypatch):
    root = make_artifact(tmp_path / "model", "linear_svc", "only_model")
    monkeypatch.setenv("VTC_MODEL_DIR", str(root))
    monkeypatch.setenv("VTC_DEFAULT_MODEL", "not_registered")
    assert client.get("/api/models").status_code == 503
    assert client.get("/api/status").json()["ready"] is False


def test_corrupted_artifact_returns_service_unavailable(tmp_path, monkeypatch):
    root = make_artifact(tmp_path / "model", "logistic_regression", "broken_model")
    (root / "model.joblib").write_bytes(b"this is not a fitted sklearn pipeline")
    monkeypatch.setenv("VTC_MODEL_DIR", str(root))
    assert client.get("/api/status").json()["ready"] is False
    response = client.post("/api/predict", json={"text": "Thị trường tài chính"})
    assert response.status_code == 503
    assert "Traceback" not in response.text


def test_modular_app_and_openapi_route_contract():
    """Ensure the router composition preserves all Next.js-facing endpoints."""
    from apps.api.main import create_app

    fresh_app = create_app()
    operations = fresh_app.openapi()["paths"]
    assert {"/api/health", "/api/models", "/api/status", "/api/labels", "/api/predict"} <= set(operations)
    assert "post" in operations["/api/predict"]
    assert "get" in operations["/api/models"]
