"""Stratified K-fold cross-validation with out-of-fold predictions and blend scoring."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import N_SPLITS, SEED
from .models import fit_predict


def cross_validate(X, y, models, n_splits: int = N_SPLITS, seed: int = SEED, verbose: bool = True) -> dict:
    """Return per-model OOF predictions, fold AUCs, best iterations and the equal-weight blend's fold AUCs."""
    splits = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y))
    oof = {m: np.zeros(len(y)) for m in models}
    fold_auc = {m: [] for m in models}
    iters = {m: [] for m in models}
    for k, (a, b) in enumerate(splits):
        for m in models:
            t0 = time.time()
            p, it = fit_predict(m, X.iloc[a], y[a], X.iloc[b], y[b])
            oof[m][b] = p
            fold_auc[m].append(roc_auc_score(y[b], p))
            iters[m].append(it)
            if verbose:
                print(f"fold {k} {m}: AUC {fold_auc[m][-1]:.5f}  trees {it}  ({time.time() - t0:.0f}s)", flush=True)
    blend = [roc_auc_score(y[b], np.mean([oof[m][b] for m in models], axis=0)) for _, b in splits]
    return {"oof": oof, "fold_auc": fold_auc, "iters": iters, "blend_fold_auc": blend}


def summarize(res: dict) -> pd.DataFrame:
    """Tidy table: model, mean AUC, std AUC, mean best iteration."""
    rows = [(m, np.mean(a), np.std(a), int(np.mean(res["iters"][m]))) for m, a in res["fold_auc"].items()]
    if len(res["fold_auc"]) > 1:
        rows.append(("BLEND (equal-weight mean)", np.mean(res["blend_fold_auc"]), np.std(res["blend_fold_auc"]), None))
    return pd.DataFrame(rows, columns=["model", "cv_auc_mean", "cv_auc_std", "mean_best_iter"]).round(5)
