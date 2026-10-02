#!/usr/bin/env python
"""Seeded random search over LightGBM hyper-parameters (5-fold CV, all engineered features)."""
import argparse
import time

import numpy as np
import pandas as pd

from hcdr import config
from hcdr.data import load_data
from hcdr.features import prepare
from hcdr.validation import cross_validate

SPACE = dict(num_leaves=[7, 15, 31, 63], min_child_samples=[50, 100, 200, 400], colsample_bytree=[0.3, 0.5, 0.7],
             reg_lambda=[1, 10, 50, 100], reg_alpha=[0, 1, 10], subsample=[0.7, 0.8, 1.0],
             min_split_gain=[0, 0.0, 0.01], max_bin=[63, 255])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=14)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="experiments/tuning_results.csv")
    a = ap.parse_args()
    train, test, _ = load_data(a.data_dir)
    X, y, _ = prepare(train, test)
    rng = np.random.RandomState(config.SEED)
    base = dict(config.LGB_PARAMS)
    rows = []
    for i in range(a.n_trials):
        p = {k: v[rng.randint(len(v))] for k, v in SPACE.items()}
        config.LGB_PARAMS.update(p)           # models.fit_predict reads these
        t0 = time.time()
        res = cross_validate(X, y, ["lgb"], verbose=False)
        f = res["fold_auc"]["lgb"]
        rows.append(dict(trial=i, cv_auc_mean=round(np.mean(f), 5), cv_auc_std=round(np.std(f), 5), **p))
        print(f"trial {i}: {rows[-1]['cv_auc_mean']:.5f} +/- {rows[-1]['cv_auc_std']:.5f} {p} ({time.time() - t0:.0f}s)", flush=True)
        config.LGB_PARAMS.clear()
        config.LGB_PARAMS.update(base)
    pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).to_csv(a.out, index=False)
    print("saved", a.out)
