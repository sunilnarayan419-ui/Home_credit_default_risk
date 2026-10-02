#!/usr/bin/env python
"""Data audit: shapes, ID integrity, missingness, sentinels, and (optionally) adversarial validation."""
import argparse

import numpy as np
import pandas as pd

from hcdr.config import ID, SEED, TARGET
from hcdr.data import load_data

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--adversarial", action="store_true", help="train-vs-test classifier (needs lightgbm)")
    a = ap.parse_args()
    train, test, sample = load_data(a.data_dir)
    print("train", train.shape, "| test", test.shape, "| sample_submission", sample.shape)
    print("columns only in train:", set(train.columns) - set(test.columns), "| only in test:", set(test.columns) - set(train.columns))
    print("duplicate IDs:", train[ID].duplicated().sum(), test[ID].duplicated().sum(), "| overlap:", len(set(train[ID]) & set(test[ID])))
    print("duplicate rows (excl. ID):", train.drop(columns=ID).duplicated().sum())
    print("test ID order == sample_submission:", test[ID].equals(sample[ID]))
    print(f"target mean {train[TARGET].mean():.4f} ({train[TARGET].sum()} positives)")
    miss = pd.DataFrame({"train_%": train.isnull().mean() * 100, "test_%": test.isnull().mean().reindex(train.columns) * 100}).round(1)
    print(miss[miss["train_%"] > 0].sort_values("train_%", ascending=False).to_string())
    sent = lambda d: (d.DAYS_EMPLOYED == 365243).mean()  # noqa: E731
    print(f"DAYS_EMPLOYED==365243: train {sent(train):.3f} test {sent(test):.3f}")
    print("XNA:", {c: int((train[c] == 'XNA').sum()) for c in train.select_dtypes(exclude='number') if (train[c] == 'XNA').any()})
    if a.adversarial:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        f = [c for c in test.columns if c != ID]
        A = pd.concat([train[f], test[f]], ignore_index=True)
        for c in A.select_dtypes(exclude="number"):
            A[c] = A[c].astype("category")
        lab = np.r_[np.zeros(len(train)), np.ones(len(test))]
        oof = np.zeros(len(A))
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(A, lab):
            m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, random_state=SEED, n_jobs=1, verbose=-1).fit(A.iloc[tr], lab[tr])
            oof[va] = m.predict_proba(A.iloc[va])[:, 1]
        print("adversarial AUC:", round(roc_auc_score(lab, oof), 4), "(0.5 = same distribution)")
