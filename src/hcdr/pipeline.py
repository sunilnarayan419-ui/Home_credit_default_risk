"""End-to-end: load -> features -> fit on ALL training rows -> equal-weight blend -> submission.csv."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import OUTPUT_DIR, SEED
from .data import load_data
from .features import prepare
from .models import available_models, fit_predict
from .submission import validate_submission, write_submission


def train_and_predict(data_dir=None, out_path=None, force_fallback: bool = False, verbose: bool = True) -> Path:
    np.random.seed(SEED)
    train, test, sample = load_data(data_dir)
    X_train, y, X_test = prepare(train, test)
    models = available_models(force_fallback)
    if verbose:
        print(f"features: {X_train.shape[1]} | models: {models}", flush=True)
    preds = {}
    for m in models:
        t0 = time.time()
        preds[m], _ = fit_predict(m, X_train, y, X_test)
        if verbose:
            print(f"{m}: fitted on {len(X_train)} rows, predicted {len(X_test)} rows ({time.time() - t0:.0f}s)", flush=True)
    pred = np.mean(list(preds.values()), axis=0)  # equal-weight probability average
    path = write_submission(sample, test, pred, out_path or OUTPUT_DIR / "submission.csv")
    info = validate_submission(path, sample, test)
    if verbose:
        print("submission OK:", info)
    return Path(path)
