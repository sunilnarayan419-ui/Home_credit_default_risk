"""Submission writing and verification."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID, TARGET


def write_submission(sample: pd.DataFrame, test: pd.DataFrame, pred, path: str | Path) -> Path:
    """Fill ``sample_submission`` with predictions (same row order as test) and save without an index."""
    if not test[ID].equals(sample[ID]):
        raise ValueError("test IDs are not in sample_submission order")
    if len(pred) != len(sample):
        raise ValueError("one prediction per test row is required")
    sub = sample.copy()
    sub[TARGET] = pred
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def validate_submission(path: str | Path, sample: pd.DataFrame, test: pd.DataFrame | None = None) -> dict:
    """Reload the CSV and assert every competition requirement. Returns a summary dict."""
    sub = pd.read_csv(path)
    assert list(sub.columns) == [ID, TARGET], f"columns must be exactly [{ID}, {TARGET}], got {list(sub.columns)}"
    assert sub.shape == sample.shape, f"shape {sub.shape} != sample_submission {sample.shape}"
    assert sub[ID].equals(sample[ID]), "IDs differ from sample_submission (values or order)"
    if test is not None:
        assert sub[ID].equals(test[ID]), "IDs differ from test order"
    assert sub[TARGET].notna().all(), "NaN predictions"
    assert np.isfinite(sub[TARGET]).all(), "non-finite predictions"
    assert sub[TARGET].between(0, 1).all(), "predictions must be probabilities in [0, 1]"
    assert sub[TARGET].nunique() > 2, "predictions look like hard 0/1 labels; AUC needs continuous scores"
    return {"rows": len(sub), "columns": list(sub.columns), "nan": int(sub[TARGET].isna().sum()),
            "min": float(sub[TARGET].min()), "max": float(sub[TARGET].max()),
            "unique": int(sub[TARGET].nunique()), "ids_match_sample": True}
