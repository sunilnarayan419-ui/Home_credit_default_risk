import numpy as np
import pandas as pd
import pytest
from conftest import records
from fastapi.testclient import TestClient

from hcdr.api.app import create_app
from hcdr.api.schemas import MAX_BATCH_SIZE
from hcdr.inference import predict_default_probability

pytestmark = pytest.mark.api


@pytest.fixture
def client(bundle_path):
    with TestClient(create_app(bundle_path)) as c:
        yield c


def test_predict_success(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    body = r.json()
    assert set(body) == {"default_probability"} and 0.0 <= body["default_probability"] <= 1.0


def test_predict_matches_library_scoring(client, bundle):
    rows = records(4)
    expected = predict_default_probability(bundle, pd.DataFrame(rows))
    got = [client.post("/predict", json=r).json()["default_probability"] for r in rows]
    assert np.allclose(got, expected)


def test_predict_accepts_missing_optional_fields(client, payload):
    for k in ("OWN_CAR_AGE", "EXT_SOURCE_1", "OCCUPATION_TYPE", "AMT_REQ_CREDIT_BUREAU_YEAR"):
        payload.pop(k, None)
    assert client.post("/predict", json=payload).status_code == 200


def test_batch_success_preserves_order(client, bundle):
    rows = records(5, seed=11)
    r = client.post("/predict/batch", json={"records": rows})
    assert r.status_code == 200
    body = r.json()
    expected = predict_default_probability(bundle, pd.DataFrame(rows))
    assert body["count"] == 5 and np.allclose([p["default_probability"] for p in body["predictions"]], expected)


@pytest.mark.parametrize("field, value", [
    ("AMT_INCOME_TOTAL", 0), ("AMT_INCOME_TOTAL", -5), ("AMT_CREDIT", "a lot"), ("DAYS_BIRTH", 100),
    ("EXT_SOURCE_2", 1.5), ("FLAG_MOBIL", 2), ("REGION_RATING_CLIENT", 4), ("CNT_CHILDREN", -1),
    ("CODE_GENDER", ""), ("AMT_ANNUITY", "nan"),
])
def test_predict_rejects_invalid_values(client, payload, field, value):
    payload[field] = value
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert all(set(e) == {"loc", "msg", "type"} for e in r.json()["detail"])  # no echoed input values


def test_validation_errors_do_not_echo_submitted_values(client, payload):
    payload["AMT_INCOME_TOTAL"] = "SENSITIVE-12345"
    assert "SENSITIVE-12345" not in client.post("/predict", json=payload).text


def test_predict_rejects_missing_required_and_unknown_fields(client, payload):
    missing = {k: v for k, v in payload.items() if k != "AMT_CREDIT"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**payload, "TARGET": 1}).status_code == 422  # label is never an input
    assert client.post("/predict", json={**payload, "SK_ID_CURR": 1}).status_code == 422


def test_predict_rejects_malformed_json_and_wrong_types(client):
    assert client.post("/predict", content=b"{not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/predict", json=[1, 2, 3]).status_code == 422
    assert client.post("/predict").status_code == 422


def test_batch_limits(client, payload):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422
    assert client.post("/predict/batch", json={"records": [payload] * (MAX_BATCH_SIZE + 1)}).status_code == 422
    assert client.post("/predict/batch", json={"records": [{**payload, "AMT_CREDIT": -1}]}).status_code == 422


def test_wrong_method_is_405(client):
    assert client.get("/predict").status_code == 405


def test_predict_without_model_is_503(tmp_path, payload):
    with TestClient(create_app(tmp_path / "missing.joblib")) as c:
        for path, body in (("/predict", payload), ("/predict/batch", {"records": [payload]})):
            r = c.post(path, json=body)
            assert r.status_code == 503 and "missing.joblib" not in r.text and str(tmp_path) not in r.text


def test_unexpected_error_returns_generic_500(bundle_path, payload, monkeypatch):
    import hcdr.api.app as app_module

    def boom(*a, **k):
        raise RuntimeError("secret internal detail at /srv/model")

    monkeypatch.setattr(app_module, "predict_default_probability", boom)
    with TestClient(create_app(bundle_path), raise_server_exceptions=False) as c:
        r = c.post("/predict", json=payload)
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}
