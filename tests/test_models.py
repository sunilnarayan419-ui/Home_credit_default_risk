import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from hcdr.features import prepare
from hcdr.models import available_models, fit_predict
from hcdr.validation import cross_validate, summarize


def test_fallback_when_forced():
    assert available_models(force_fallback=True) == ["hgb"]


@pytest.mark.parametrize("name", ["hgb", "lgb"])
def test_fit_predict_returns_probabilities(name, train_df, test_df):
    if name != "hgb" and name not in available_models():
        pytest.skip(f"{name} not installed")
    X, y, Xt = prepare(train_df, test_df)
    p, it = fit_predict(name, X, y, Xt, n_iter=30)
    assert p.shape == (len(Xt),) and np.all((p >= 0) & (p <= 1)) and np.isfinite(p).all() and it == 30


def test_fit_predict_is_deterministic(train_df, test_df):
    X, y, Xt = prepare(train_df, test_df)
    a, _ = fit_predict("hgb", X, y, Xt, n_iter=20)
    b, _ = fit_predict("hgb", X, y, Xt, n_iter=20)
    assert np.array_equal(a, b)


def test_cross_validate_stratified_and_sane(train_df, test_df):
    X, y, _ = prepare(train_df, test_df)
    res = cross_validate(X, y, ["hgb"], n_splits=3, verbose=False)
    assert len(res["fold_auc"]["hgb"]) == 3 and roc_auc_score(y, res["oof"]["hgb"]) > 0.5
    assert list(summarize(res).columns)[:2] == ["model", "cv_auc_mean"]
