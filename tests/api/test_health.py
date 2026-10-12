import pytest
from fastapi.testclient import TestClient

from hcdr.api.app import create_app

pytestmark = pytest.mark.api


@pytest.fixture
def client(bundle_path):
    with TestClient(create_app(bundle_path)) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as c:
        yield c


def test_health_ready(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_health_degraded_without_artifact(client_without_model):
    r = client_without_model.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "degraded", "model_loaded": False}


def test_model_info_is_safe_metadata(client):
    r = client.get("/model-info")
    body = r.json()
    assert r.status_code == 200 and body["models"] == ["hgb"] and body["n_features"] == 78 and body["n_train_rows"] == 600
    text = r.text.lower()
    assert "joblib" not in text and "/tmp" not in text  # no filesystem paths


def test_model_info_unavailable(client_without_model):
    r = client_without_model.get("/model-info")
    assert r.status_code == 503 and r.json() == {"detail": "Model is not available. Train and install the model artifact first."}


def test_corrupt_artifact_starts_degraded_without_leaking_details(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"garbage")
    with TestClient(create_app(bad)) as c:
        assert c.get("/health").json()["model_loaded"] is False
        r = c.post("/predict", json={})
        assert r.status_code in (422, 503) and "garbage" not in r.text and str(tmp_path) not in r.text


def test_openapi_and_docs_are_served(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    assert {"/health", "/model-info", "/predict", "/predict/batch"} <= set(spec.json()["paths"])
