#!/usr/bin/env python
"""Stratified K-fold CV (ROC-AUC) for one or more models, the equal-weight blend, and feature-group ablations.

Examples
    python scripts/run_cv.py                                  # LGB+XGB+Cat blend, all features (recorded: 0.7622)
    python scripts/run_cv.py --models lgb --groups ratios     # ablation: ratio features only
    python scripts/run_cv.py --models lgb --groups            # cleaned raw features, no engineering
"""
import argparse

import numpy as np

from hcdr.config import FEATURE_GROUPS, N_SPLITS, SEED
from hcdr.data import load_data
from hcdr.features import prepare
from hcdr.models import available_models
from hcdr.validation import cross_validate, summarize

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, help="subset of: lgb xgb cat hgb (default: all installed)")
    ap.add_argument("--groups", nargs="*", default=list(FEATURE_GROUPS), choices=FEATURE_GROUPS)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--save-oof", default=None, help="optional .npz path for out-of-fold predictions")
    a = ap.parse_args()
    train, test, _ = load_data(a.data_dir)
    X, y, _ = prepare(train, test, tuple(a.groups))
    models = a.models or available_models()
    print(f"features: {X.shape[1]} (groups: {a.groups or 'none'}) | models: {models} | folds: {a.n_splits} | seed: {SEED}")
    res = cross_validate(X, y, models, n_splits=a.n_splits)
    print(summarize(res).to_string(index=False))
    print("fold AUCs:", {m: np.round(v, 5).tolist() for m, v in res["fold_auc"].items()})
    if a.save_oof:
        np.savez(a.save_oof, **res["oof"])
