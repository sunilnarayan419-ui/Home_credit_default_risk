"""The production boosters (LightGBM / XGBoost / CatBoost) on tiny synthetic data; skipped if a library is absent."""
import numpy as np
import pytest
from conftest import make_frame

from hcdr.features import prepare
from hcdr.inference import fit_bundle, predict_default_probability
from hcdr.models import available_models, fit_model, fit_predict, predict_proba

LIBS = {"lgb": "lightgbm", "xgb": "xgboost", "cat": "catboost"}


@pytest.fixture(scope="module")
def data():
    return prepare(make_frame(600, seed=1), make_frame(60, seed=2, with_target=False, start_id=200000))


@pytest.mark.parametrize("name", list(LIBS))
def test_final_fit_gives_valid_probabilities(name, data):
    pytest.importorskip(LIBS[name])
    X, y, Xt = data
    model = fit_model(name, X, y, n_iter=15)
    p = predict_proba(name, model, Xt)
    assert p.shape == (len(Xt),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    p2, trees = fit_predict(name, X, y, Xt, n_iter=15)  # same code path, same result
    assert trees == 15 and np.allclose(p, p2)


@pytest.mark.parametrize("name", list(LIBS))
def test_cv_path_returns_best_iteration(name, data):
    pytest.importorskip(LIBS[name])
    X, y, _ = data
    cut = 450
    p, it = fit_predict(name, X.iloc[:cut], y[:cut], X.iloc[cut:], y[cut:], n_iter=40)
    assert len(p) == len(X) - cut and 1 <= it <= 40 and ((p >= 0) & (p <= 1)).all()


def test_unknown_model_is_rejected(data):
    X, y, Xt = data
    with pytest.raises(ValueError, match="unknown"):
        fit_model("nope", X, y)
    with pytest.raises(ValueError, match="unknown"):
        fit_predict("nope", X, y, Xt, y)


def test_available_models_fallback_and_order():
    assert available_models(force_fallback=True) == ["hgb"]
    assert available_models() == ([m for m in LIBS if _has(LIBS[m])] or ["hgb"])


def _has(lib):
    try:
        __import__(lib)
        return True
    except ImportError:
        return False


def test_full_blend_bundle_serves_the_mean_of_its_members():
    for lib in LIBS.values():
        pytest.importorskip(lib)
    train = make_frame(600, seed=1)
    bundle = fit_bundle(train, models=list(LIBS), n_iter={m: 15 for m in LIBS})
    rows = make_frame(10, seed=3, with_target=False, start_id=300000)
    X, _, Xt = prepare(train, rows)
    members = [predict_proba(n, m, Xt) for n, m in bundle.models.items()]
    assert np.allclose(predict_default_probability(bundle, rows), np.mean(members, axis=0))
