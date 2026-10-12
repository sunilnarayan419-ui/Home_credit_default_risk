import numpy as np
import pandas as pd
import pytest
from conftest import make_frame

from hcdr.features import prepare
from hcdr.inference import (
    ArtifactError,
    InferenceInputError,
    fit_bundle,
    load_bundle,
    predict_default_probability,
    save_bundle,
)
from hcdr.models import fit_predict


def _rows(n=8, seed=7):
    return make_frame(n, seed=seed, with_target=False, start_id=300000)


def test_bundle_contract(bundle):
    assert bundle.models.keys() == {"hgb"}
    assert "TARGET" not in bundle.raw_columns and "SK_ID_CURR" not in bundle.raw_columns
    assert len(bundle.raw_columns) == 32 and len(bundle.feature_columns) == 78
    assert bundle.metadata["n_train_rows"] == 600 and bundle.metadata["models"] == ["hgb"]


def test_predictions_are_valid_probabilities(bundle):
    p = predict_default_probability(bundle, _rows())
    assert p.shape == (8,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_serving_path_matches_batch_pipeline(bundle):
    """The API/bundle path must give exactly what the competition pipeline gives for the same model and rows."""
    train, rows = make_frame(600, seed=1), _rows()
    X, y, Xt = prepare(train, rows)
    expected, _ = fit_predict("hgb", X, y, Xt, n_iter=20)
    assert np.allclose(predict_default_probability(bundle, rows), expected)


def test_column_order_and_id_do_not_matter(bundle):
    rows = _rows()
    a = predict_default_probability(bundle, rows)
    b = predict_default_probability(bundle, rows[rows.columns[::-1]].drop(columns="SK_ID_CURR"))
    assert np.allclose(a, b)


def test_row_scores_do_not_depend_on_batch_composition(bundle):
    rows = _rows()
    assert np.allclose(predict_default_probability(bundle, rows)[:1], predict_default_probability(bundle, rows.head(1)))


def test_unseen_category_level_is_scored_as_missing(bundle):
    rows = _rows(2).copy()
    rows.loc[rows.index[0], "OCCUPATION_TYPE"] = "A level never seen in training"
    assert np.isfinite(predict_default_probability(bundle, rows)).all()


def test_all_optional_values_missing(bundle):
    rows = _rows(2).copy()
    rows[["OWN_CAR_AGE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "OCCUPATION_TYPE"]] = None
    rows[[c for c in rows.columns if c.startswith("AMT_REQ_")]] = None
    assert np.isfinite(predict_default_probability(bundle, rows)).all()


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d.drop(columns="AMT_CREDIT"), "missing"),
    (lambda d: d.assign(NOT_A_FEATURE=1), "unknown"),
    (lambda d: d.assign(TARGET=1), "unknown"),  # the label must never be accepted as an input
])
def test_rejects_columns_outside_the_contract(bundle, mutate, match):
    with pytest.raises(InferenceInputError, match=match):
        predict_default_probability(bundle, mutate(_rows()))


def test_rejects_empty_and_non_numeric(bundle):
    with pytest.raises(InferenceInputError):
        predict_default_probability(bundle, _rows().head(0))
    bad = _rows(2).astype({"AMT_CREDIT": object})
    bad.loc[bad.index[0], "AMT_CREDIT"] = "lots"
    with pytest.raises(InferenceInputError, match="non-numeric"):
        predict_default_probability(bundle, bad)


def test_save_load_roundtrip(bundle, tmp_path):
    path = save_bundle(bundle, tmp_path / "nested" / "m.joblib")
    loaded = load_bundle(path)
    rows = _rows()
    assert loaded.feature_columns == bundle.feature_columns and loaded.metadata == bundle.metadata
    assert np.array_equal(predict_default_probability(loaded, rows), predict_default_probability(bundle, rows))


def test_load_missing_corrupt_and_foreign_artifacts(tmp_path):
    with pytest.raises(ArtifactError, match="not found"):
        load_bundle(tmp_path / "nope.joblib")
    (tmp_path / "corrupt.joblib").write_bytes(b"not a pickle")
    with pytest.raises(ArtifactError, match="could not be read"):
        load_bundle(tmp_path / "corrupt.joblib")
    import joblib

    joblib.dump({"hello": "world"}, tmp_path / "foreign.joblib")
    with pytest.raises(ArtifactError, match="structure"):
        load_bundle(tmp_path / "foreign.joblib")


def test_load_rejects_other_format_version(bundle, tmp_path):
    import joblib

    path = save_bundle(bundle, tmp_path / "m.joblib")
    payload = joblib.load(path)
    payload["format_version"] = 999
    joblib.dump(payload, path)
    with pytest.raises(ArtifactError, match="version"):
        load_bundle(path)


def test_fit_bundle_does_not_touch_training_frame():
    train = make_frame(300, seed=3)
    before = train.copy()
    fit_bundle(train, models=["hgb"], n_iter={"hgb": 5})
    pd.testing.assert_frame_equal(train, before)
