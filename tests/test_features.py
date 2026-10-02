import numpy as np
import pytest

from hcdr.config import CAT_COLS, FEATURE_GROUPS
from hcdr.features import add_features, build_matrix, clean, prepare, to_categorical


def test_sentinels_are_cleaned(train_df):
    c = clean(train_df)
    assert (train_df.DAYS_EMPLOYED == 365243).any()
    assert not (c.DAYS_EMPLOYED == 365243).any()
    assert c.DAYS_EMPLOYED_ANOM.sum() == (train_df.DAYS_EMPLOYED == 365243).sum()
    assert not (c.ORGANIZATION_TYPE == "XNA").any() and not (c.CODE_GENDER == "XNA").any()


def test_matrix_has_no_target_or_id_and_no_inf(train_df, test_df):
    Xtr, Xte = build_matrix(train_df), build_matrix(test_df)
    assert "TARGET" not in Xtr.columns and "SK_ID_CURR" not in Xtr.columns
    assert list(Xtr.columns) == list(Xte.columns)
    num = Xtr.select_dtypes("number")
    assert np.isfinite(num.to_numpy(dtype=float)[~np.isnan(num.to_numpy(dtype=float))]).all()


def test_age_and_ratio_semantics(train_df):
    f = add_features(clean(train_df))
    assert (f.AGE_YEARS > 0).all()
    ok = f.AMT_INCOME_TOTAL > 0
    assert np.allclose(f.CREDIT_INCOME[ok], (f.AMT_CREDIT / f.AMT_INCOME_TOTAL)[ok])


def test_feature_groups_are_additive_and_validated(train_df):
    base = clean(train_df).shape[1]
    sizes = {g: add_features(clean(train_df), (g,)).shape[1] - base for g in FEATURE_GROUPS}
    assert all(s > 0 for s in sizes.values())
    total = add_features(clean(train_df)).shape[1] - base
    assert total == sum(sizes.values()) + 1  # +1: EXT x CREDIT_TERM interaction needs the ratios group
    with pytest.raises(ValueError):
        add_features(train_df, ("nonsense",))


def test_categories_come_from_train_only(train_df, test_df):
    test_df = test_df.copy()
    test_df.loc[test_df.index[:5], "OCCUPATION_TYPE"] = "A level only seen in test"
    Xtr, Xte = build_matrix(train_df), build_matrix(test_df)
    Ctr, Cte = to_categorical(Xtr, Xte)
    assert "A level only seen in test" not in Ctr.OCCUPATION_TYPE.cat.categories
    assert Cte.OCCUPATION_TYPE.iloc[:5].isna().all()
    assert all(str(Ctr[c].dtype) == "category" for c in CAT_COLS)


def test_prepare_shapes(train_df, test_df):
    X, y, Xt = prepare(train_df, test_df)
    assert len(X) == len(y) == len(train_df) and len(Xt) == len(test_df) and X.shape[1] == Xt.shape[1]
