"""Integration: CSV files -> features -> models -> artifact/submission, on tiny synthetic data."""
import joblib
import numpy as np
import pandas as pd
import pytest
from conftest import make_frame

from hcdr.inference import load_bundle, predict_default_probability
from hcdr.pipeline import train_and_predict, train_and_save
from hcdr.submission import validate_submission

pytestmark = pytest.mark.integration


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    make_frame(700, seed=1).to_csv(d / "train.csv", index=False)
    test = make_frame(100, seed=2, with_target=False, start_id=200000)
    test.to_csv(d / "test.csv", index=False)
    pd.DataFrame({"SK_ID_CURR": test.SK_ID_CURR, "TARGET": 0.5}).to_csv(d / "sample_submission.csv", index=False)
    return d


def test_train_and_predict_writes_valid_submission(data_dir, tmp_path):
    out = train_and_predict(data_dir, tmp_path / "sub.csv", force_fallback=True, verbose=False)
    sample, test = pd.read_csv(data_dir / "sample_submission.csv"), pd.read_csv(data_dir / "test.csv")
    assert validate_submission(out, sample, test)["rows"] == 100


def test_train_and_save_then_serve_matches_submission(data_dir, tmp_path):
    art = train_and_save(data_dir, tmp_path / "m.joblib", force_fallback=True, verbose=False)
    sub = pd.read_csv(train_and_predict(data_dir, tmp_path / "sub.csv", force_fallback=True, verbose=False))
    test = pd.read_csv(data_dir / "test.csv")
    served = predict_default_probability(load_bundle(art), test)
    assert np.allclose(served, sub.TARGET.to_numpy())  # same models, same rows -> same scores


def test_train_and_save_never_stores_training_rows(data_dir, tmp_path):
    art = train_and_save(data_dir, tmp_path / "m.joblib", force_fallback=True, verbose=False)
    payload = joblib.load(art)
    assert not any(isinstance(v, pd.DataFrame) for v in payload.values())
    assert "SK_ID_CURR" not in payload["raw_columns"] and "TARGET" not in payload["raw_columns"]
