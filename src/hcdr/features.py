"""Cleaning and feature engineering.

Leakage note: nothing here uses TARGET and nothing learns statistics from data. Every transform is
row-wise (ratios, unit conversions, sentinel replacement). The only fitted object is the list of
category levels, taken from the TRAINING frame only (see :func:`to_categorical`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CAT_COLS, FEATURE_GROUPS, ID, TARGET

DAYS_EMPLOYED_SENTINEL = 365243


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinels: DAYS_EMPLOYED==365243 -> NaN (+flag); 'XNA' -> NaN."""
    d = df.copy()
    d["DAYS_EMPLOYED_ANOM"] = (d.DAYS_EMPLOYED == DAYS_EMPLOYED_SENTINEL).astype("int8")
    d.loc[d.DAYS_EMPLOYED == DAYS_EMPLOYED_SENTINEL, "DAYS_EMPLOYED"] = np.nan
    d["ORGANIZATION_TYPE"] = d["ORGANIZATION_TYPE"].replace("XNA", np.nan)
    d["CODE_GENDER"] = d["CODE_GENDER"].replace("XNA", np.nan)
    return d


def _ratios(d: pd.DataFrame) -> None:
    inc = d.AMT_INCOME_TOTAL
    d["CREDIT_INCOME"] = d.AMT_CREDIT / inc
    d["ANNUITY_INCOME"] = d.AMT_ANNUITY / inc
    d["GOODS_INCOME"] = d.AMT_GOODS_PRICE / inc
    d["CREDIT_GOODS"] = d.AMT_CREDIT / d.AMT_GOODS_PRICE
    d["ANNUITY_CREDIT"] = d.AMT_ANNUITY / d.AMT_CREDIT
    d["CREDIT_TERM"] = d.AMT_CREDIT / d.AMT_ANNUITY
    d["CREDIT_MINUS_GOODS"] = d.AMT_CREDIT - d.AMT_GOODS_PRICE
    d["INCOME_PER_FAM"] = inc / d.CNT_FAM_MEMBERS
    d["CREDIT_PER_FAM"] = d.AMT_CREDIT / d.CNT_FAM_MEMBERS
    d["ANNUITY_PER_FAM"] = d.AMT_ANNUITY / d.CNT_FAM_MEMBERS
    d["INCOME_PER_CHILD"] = inc / (d.CNT_CHILDREN + 1)


def _time(d: pd.DataFrame) -> None:  # DAYS_* are negative = days before application
    inc = d.AMT_INCOME_TOTAL
    d["AGE_YEARS"] = -d.DAYS_BIRTH / 365.25
    d["EMPLOYED_YEARS"] = -d.DAYS_EMPLOYED / 365.25
    d["EMPLOYED_TO_AGE"] = d.DAYS_EMPLOYED / d.DAYS_BIRTH
    d["REG_TO_AGE"] = d.DAYS_REGISTRATION / d.DAYS_BIRTH
    d["ID_TO_AGE"] = d.DAYS_ID_PUBLISH / d.DAYS_BIRTH
    d["PHONE_TO_AGE"] = d.DAYS_LAST_PHONE_CHANGE / d.DAYS_BIRTH
    d["CREDIT_TO_AGE"] = d.AMT_CREDIT / d.AGE_YEARS
    d["INCOME_TO_EMPLOYED"] = inc / (d.EMPLOYED_YEARS + 1)
    d["CAR_TO_EMPLOYED"] = d.OWN_CAR_AGE / (d.EMPLOYED_YEARS + 1)
    d["CAR_TO_AGE"] = d.OWN_CAR_AGE / d.AGE_YEARS
    d["ANNUITY_TO_AGE"] = d.AMT_ANNUITY / d.AGE_YEARS


def _ext(d: pd.DataFrame, with_ratios: bool) -> None:
    e = d[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    d["EXT_MEAN"] = e.mean(axis=1)
    d["EXT_MIN"] = e.min(axis=1)
    d["EXT_MAX"] = e.max(axis=1)
    d["EXT_STD"] = e.std(axis=1)
    d["EXT_NNULL"] = e.isna().sum(axis=1)
    d["EXT_PROD"] = e.prod(axis=1, skipna=True).where(e.notna().any(axis=1))
    d["EXT_1_2"] = d.EXT_SOURCE_1 - d.EXT_SOURCE_2
    d["EXT_1_3"] = d.EXT_SOURCE_1 - d.EXT_SOURCE_3
    d["EXT_2_3"] = d.EXT_SOURCE_2 - d.EXT_SOURCE_3
    d["EXT_2x3"] = d.EXT_SOURCE_2 * d.EXT_SOURCE_3
    d["EXT_1x2"] = d.EXT_SOURCE_1 * d.EXT_SOURCE_2
    d["EXT_1x3"] = d.EXT_SOURCE_1 * d.EXT_SOURCE_3
    if with_ratios:
        d["EXT_MEAN_x_CREDIT_TERM"] = d.EXT_MEAN * d.CREDIT_TERM
    d["EXT_MEAN_x_AGE"] = d.EXT_MEAN * (-d.DAYS_BIRTH / 365.25)


def _bureau(d: pd.DataFrame) -> None:
    b = d[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON",
           "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]]
    d["BUREAU_TOTAL"] = b.sum(axis=1, min_count=1)
    d["BUREAU_NULL"] = b.isna().any(axis=1).astype("int8")
    d["BUREAU_RECENT"] = b[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON"]].sum(axis=1, min_count=1)
    d["BUREAU_YEAR_SHARE_RECENT"] = d.BUREAU_RECENT / (d.BUREAU_TOTAL + 1)


def _misc(d: pd.DataFrame) -> None:
    d["OWN_CAR_AGE_OUTLIER"] = (d.OWN_CAR_AGE >= 60).astype("int8")
    d["HAS_CAR"] = d.OWN_CAR_AGE.notna().astype("int8")
    d["FLAG_SUM"] = d.FLAG_EMP_PHONE + d.FLAG_WORK_PHONE
    d["REGION_POP_x_RATING"] = d.REGION_POPULATION_RELATIVE * d.REGION_RATING_CLIENT
    d["CHILD_RATIO"] = d.CNT_CHILDREN / d.CNT_FAM_MEMBERS


def add_features(df: pd.DataFrame, groups: tuple[str, ...] = FEATURE_GROUPS) -> pd.DataFrame:
    """Add engineered feature groups (default: all). Column order is fixed: ratios, time, ext, bureau, misc."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {unknown}")
    d = df.copy()
    if "ratios" in groups:
        _ratios(d)
    if "time" in groups:
        _time(d)
    if "ext" in groups:
        _ext(d, with_ratios="ratios" in groups)
    if "bureau" in groups:
        _bureau(d)
    if "misc" in groups:
        _misc(d)
    return d.replace([np.inf, -np.inf], np.nan)


def build_matrix(df: pd.DataFrame, groups: tuple[str, ...] = FEATURE_GROUPS) -> pd.DataFrame:
    """clean -> add_features -> drop ID/TARGET (TARGET never reaches the model inputs)."""
    out = add_features(clean(df), groups)
    return out.drop(columns=[c for c in (TARGET, ID) if c in out.columns])


def to_categorical(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """category dtype with levels from TRAIN only (no labels involved); unseen test levels become NaN."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in CAT_COLS:
        levels = sorted(Xtr[c].dropna().unique())
        Xtr[c] = pd.Series(pd.Categorical(Xtr[c], categories=levels), index=Xtr.index)
        seen = Xte[c].astype(object).where(Xte[c].isin(levels))  # unseen levels -> NaN
        Xte[c] = pd.Series(pd.Categorical(seen, categories=levels), index=Xte.index)
    return Xtr, Xte


def to_string_cat(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants plain strings for categorical columns (NaN -> 'NA')."""
    X = X.copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
    return X


def prepare(train: pd.DataFrame, test: pd.DataFrame, groups: tuple[str, ...] = FEATURE_GROUPS):
    """Return (X_train, y, X_test) ready for the models."""
    Xtr, Xte = build_matrix(train, groups), build_matrix(test, groups)
    assert list(Xtr.columns) == list(Xte.columns) and TARGET not in Xte.columns
    Xtr, Xte = to_categorical(Xtr, Xte)
    return Xtr, train[TARGET].to_numpy(), Xte
