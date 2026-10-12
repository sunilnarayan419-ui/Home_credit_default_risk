import numpy as np
import pandas as pd
import pytest
from conftest import make_frame

from hcdr import config
from hcdr.data import load_data
from hcdr.features import add_features, apply_categories, clean, fit_categories


def _write_csvs(folder, with_input_subdir=False):
    d = folder / "input" if with_input_subdir else folder
    d.mkdir(parents=True, exist_ok=True)
    tr, te = make_frame(30, seed=1), make_frame(10, seed=2, with_target=False, start_id=200000)
    tr.to_csv(d / "train.csv", index=False)
    te.to_csv(d / "test.csv", index=False)
    te[["SK_ID_CURR"]].assign(TARGET=0.5).to_csv(d / "sample_submission.csv", index=False)
    return tr, te


@pytest.mark.parametrize("subdir", [False, True])
def test_load_data_finds_files(tmp_path, subdir):
    tr, te = _write_csvs(tmp_path, subdir)
    train, test, sample = load_data(tmp_path)
    assert (len(train), len(test), len(sample)) == (len(tr), len(te), len(te))


def test_load_data_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.csv"):
        load_data(tmp_path)


def test_clean_does_not_mutate_input_and_is_idempotent(train_df):
    before = train_df.copy()
    once = clean(train_df)
    pd.testing.assert_frame_equal(train_df, before)
    twice = clean(once.drop(columns="DAYS_EMPLOYED_ANOM"))
    pd.testing.assert_frame_equal(once.drop(columns="DAYS_EMPLOYED_ANOM"), twice.drop(columns="DAYS_EMPLOYED_ANOM"))


def test_zero_denominators_become_nan_not_inf(train_df):
    d = train_df.copy()
    d.loc[d.index[:3], ["CNT_FAM_MEMBERS", "AMT_ANNUITY"]] = 0
    f = add_features(clean(d))
    assert not np.isinf(f.select_dtypes("number").to_numpy(dtype=float)).any()
    assert f.loc[d.index[:3], "CREDIT_TERM"].isna().all()  # AMT_CREDIT / 0 annuity


def test_all_missing_external_sources(train_df):
    d = train_df.copy()
    d[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]] = np.nan
    f = add_features(clean(d))
    assert (f.EXT_NNULL == 3).all() and f.EXT_MEAN.isna().all() and f.EXT_PROD.isna().all()


def test_apply_categories_unseen_and_missing_levels(train_df):
    levels = fit_categories(clean(train_df))
    assert set(levels) == set(config.CAT_COLS)
    X = clean(train_df).head(4).copy()
    X.loc[X.index[0], "NAME_INCOME_TYPE"] = "never seen"
    X.loc[X.index[1], "NAME_INCOME_TYPE"] = np.nan
    out = apply_categories(X, levels)
    assert out.NAME_INCOME_TYPE.iloc[:2].isna().all() and out.NAME_INCOME_TYPE.iloc[2:].notna().all()
    assert list(out.NAME_INCOME_TYPE.cat.categories) == levels["NAME_INCOME_TYPE"]


def test_config_is_internally_consistent():
    assert set(config.FINAL_ITERS) == {"lgb", "xgb", "cat", "hgb"}
    assert config.TARGET not in config.CAT_COLS and len(set(config.CAT_COLS)) == len(config.CAT_COLS)
