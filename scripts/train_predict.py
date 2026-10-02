#!/usr/bin/env python
"""Fit the final blend on all training rows and write output/submission.csv (then verify it)."""
import argparse

from hcdr.pipeline import train_and_predict

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help="folder with train.csv/test.csv/sample_submission.csv (default: data/)")
    ap.add_argument("--out", default=None, help="output CSV path (default: output/submission.csv)")
    ap.add_argument("--force-fallback", action="store_true", help="use scikit-learn HistGradientBoosting only")
    a = ap.parse_args()
    train_and_predict(a.data_dir, a.out, a.force_fallback)
