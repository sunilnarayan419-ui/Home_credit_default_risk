#!/usr/bin/env python
"""Check a submission CSV against every competition requirement."""
import argparse

from hcdr.config import OUTPUT_DIR
from hcdr.data import load_data
from hcdr.submission import validate_submission

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default=str(OUTPUT_DIR / "submission.csv"))
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    _, test, sample = load_data(a.data_dir)
    print(validate_submission(a.path, sample, test))
    print("All submission checks passed.")
