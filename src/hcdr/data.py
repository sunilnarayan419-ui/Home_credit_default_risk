"""Data loading (competition files are NOT stored in the repository)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR


def _find(data_dir: Path, name: str) -> Path:
    for p in (data_dir / name, data_dir / "input" / name):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{name} not found in {data_dir} (or {data_dir / 'input'}). See data/README.md."
    )


def load_data(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, sample_submission)."""
    d = Path(data_dir) if data_dir else DATA_DIR
    return (pd.read_csv(_find(d, "train.csv")), pd.read_csv(_find(d, "test.csv")),
            pd.read_csv(_find(d, "sample_submission.csv")))
