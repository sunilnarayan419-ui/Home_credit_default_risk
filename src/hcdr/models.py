"""Model factory: LightGBM / XGBoost / CatBoost with a scikit-learn HistGradientBoosting fallback."""
from __future__ import annotations

import importlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import CAT_COLS, CAT_PARAMS, FINAL_ITERS, HGB_PARAMS, LGB_PARAMS, XGB_PARAMS
from .features import to_string_cat

_LIBS = {"lgb": "lightgbm", "xgb": "xgboost", "cat": "catboost"}


def available_models(force_fallback: bool = False) -> list[str]:
    """Models whose library is installed, in blend order; ['hgb'] if none (or if forced)."""
    if force_fallback:
        return ["hgb"]
    found = []
    for name, lib in _LIBS.items():
        try:
            importlib.import_module(lib)
            found.append(name)
        except ImportError:
            pass
    return found or ["hgb"]


def fit_predict(name: str, Xtr, ytr, Xva, yva=None, n_iter: int | None = None):
    """Fit one model and predict probabilities for Xva.

    If ``yva`` is given, early stopping on it is used (CV only) and the best iteration is returned;
    otherwise ``n_iter`` trees (default ``FINAL_ITERS``) are fitted with no validation data.
    Returns (probabilities, number_of_trees).
    """
    if yva is None and n_iter is None:
        n_iter = FINAL_ITERS[name]
    if name == "lgb":
        lightgbm = importlib.import_module("lightgbm")
        p = dict(LGB_PARAMS)
        p["n_estimators"] = n_iter or p["n_estimators"]
        m = lightgbm.LGBMClassifier(**p)
        if yva is not None:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                  callbacks=[lightgbm.early_stopping(100, verbose=False)])
            it = m.best_iteration_
        else:
            m.fit(Xtr, ytr)
            it = p["n_estimators"]
    elif name == "xgb":
        xgboost = importlib.import_module("xgboost")
        p = dict(XGB_PARAMS)
        p["n_estimators"] = n_iter or p["n_estimators"]
        if yva is not None:
            m = xgboost.XGBClassifier(**p, early_stopping_rounds=100).fit(
                Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            it = int(m.best_iteration)
        else:
            m = xgboost.XGBClassifier(**p).fit(Xtr, ytr)
            it = p["n_estimators"]
    elif name == "cat":
        catboost = importlib.import_module("catboost")
        p = dict(CAT_PARAMS, cat_features=CAT_COLS)
        p["iterations"] = n_iter or p["iterations"]
        Xtr, Xva = to_string_cat(Xtr), to_string_cat(Xva)
        if yva is not None:
            m = catboost.CatBoostClassifier(**p, od_type="Iter", od_wait=80).fit(
                Xtr, ytr, eval_set=(Xva, yva))
            it = int(m.get_best_iteration())
        else:
            m = catboost.CatBoostClassifier(**p).fit(Xtr, ytr)
            it = p["iterations"]
    elif name == "hgb":
        p = dict(HGB_PARAMS)
        p["max_iter"] = n_iter or p["max_iter"]
        m = HistGradientBoostingClassifier(**p).fit(Xtr, ytr)
        it = p["max_iter"]
    else:
        raise ValueError(f"unknown model '{name}'")
    return np.asarray(m.predict_proba(Xva)[:, 1]), it
