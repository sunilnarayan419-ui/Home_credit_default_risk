import numpy as np
import pandas as pd
import pytest

from hcdr.submission import validate_submission, write_submission


@pytest.fixture
def sample(test_df):
    return pd.DataFrame({"SK_ID_CURR": test_df.SK_ID_CURR, "TARGET": 0.5})


def test_roundtrip_ok(tmp_path, sample, test_df):
    pred = np.random.RandomState(0).rand(len(sample))
    path = write_submission(sample, test_df, pred, tmp_path / "out" / "submission.csv")
    info = validate_submission(path, sample, test_df)
    assert info["rows"] == len(sample) and info["nan"] == 0 and info["columns"] == ["SK_ID_CURR", "TARGET"]


def test_rejects_bad_submissions(tmp_path, sample, test_df):
    good = np.random.RandomState(0).rand(len(sample))
    for name, mutate in {
        "nan": lambda s: s.assign(TARGET=np.where(s.index == 0, np.nan, s.TARGET)),
        "hard_labels": lambda s: s.assign(TARGET=(s.TARGET > 0.5).astype(int)),
        "out_of_range": lambda s: s.assign(TARGET=s.TARGET * 5),
        "extra_col": lambda s: s.assign(extra=1),
        "wrong_order": lambda s: s.iloc[::-1].reset_index(drop=True),
        "short": lambda s: s.iloc[:-1],
    }.items():
        s = sample.assign(TARGET=good)
        mutate(s).to_csv(tmp_path / f"{name}.csv", index=False)
        with pytest.raises(AssertionError):
            validate_submission(tmp_path / f"{name}.csv", sample, test_df)


def test_write_rejects_misordered_ids(sample, test_df, tmp_path):
    with pytest.raises(ValueError):
        write_submission(sample.iloc[::-1].reset_index(drop=True), test_df, np.zeros(len(sample)), tmp_path / "x.csv")
