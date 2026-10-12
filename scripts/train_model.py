#!/usr/bin/env python
"""Fit the final models on all training rows and save the artifact the FastAPI service loads (output/model_bundle.joblib)."""
import argparse

from hcdr.pipeline import train_and_save

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help="folder with train.csv/test.csv/sample_submission.csv (default: data/)")
    ap.add_argument("--out", default=None, help="artifact path (default: output/model_bundle.joblib or $HCDR_MODEL_PATH)")
    ap.add_argument("--force-fallback", action="store_true", help="use scikit-learn HistGradientBoosting only")
    a = ap.parse_args()
    train_and_save(a.data_dir, a.out, a.force_fallback)
