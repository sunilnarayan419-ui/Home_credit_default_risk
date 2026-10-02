import numpy as np
import pandas as pd
import pytest

CATS = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"], "CODE_GENDER": ["F", "M", "XNA"],
    "NAME_INCOME_TYPE": ["Working", "Pensioner", "State servant"],
    "NAME_EDUCATION_TYPE": ["Higher education", "Lower secondary"],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married"], "NAME_HOUSING_TYPE": ["House / apartment", "With parents"],
    "OCCUPATION_TYPE": ["Drivers", "Core staff", "Other"], "ORGANIZATION_TYPE": ["Business Entity Type 3", "School", "XNA"],
}


def make_frame(n=400, seed=0, with_target=True, start_id=100000):
    """Synthetic frame with the real competition schema (no real data is used in tests)."""
    r = np.random.RandomState(seed)
    d = pd.DataFrame({"SK_ID_CURR": np.arange(start_id, start_id + n)})
    for c, lv in CATS.items():
        d[c] = r.choice(lv, n)
    d["CNT_CHILDREN"] = r.randint(0, 4, n)
    d["AMT_INCOME_TOTAL"] = r.choice([90000.0, 135000.0, 270000.0], n)
    d["AMT_CREDIT"] = r.uniform(50000, 900000, n)
    d["AMT_ANNUITY"] = r.uniform(5000, 60000, n)
    d["AMT_GOODS_PRICE"] = d.AMT_CREDIT * r.uniform(0.8, 1.0, n)
    d["REGION_POPULATION_RELATIVE"] = r.uniform(0.001, 0.07, n)
    d["DAYS_BIRTH"] = -r.randint(7500, 25000, n)
    d["DAYS_EMPLOYED"] = np.where(r.rand(n) < 0.18, 365243, -r.randint(30, 9000, n))
    d["DAYS_REGISTRATION"] = -r.randint(0, 20000, n).astype(float)
    d["DAYS_ID_PUBLISH"] = -r.randint(0, 6000, n)
    d["OWN_CAR_AGE"] = np.where(r.rand(n) < 0.6, np.nan, r.randint(0, 70, n).astype(float))
    d["FLAG_MOBIL"] = 1
    d["FLAG_EMP_PHONE"] = r.randint(0, 2, n)
    d["FLAG_WORK_PHONE"] = r.randint(0, 2, n)
    d["CNT_FAM_MEMBERS"] = r.randint(1, 6, n).astype(float)
    d["REGION_RATING_CLIENT"] = r.randint(1, 4, n)
    for k in (1, 2, 3):
        d[f"EXT_SOURCE_{k}"] = np.where(r.rand(n) < 0.3, np.nan, r.rand(n))
    d["DAYS_LAST_PHONE_CHANGE"] = -r.randint(0, 4000, n).astype(float)
    nb = r.rand(n) < 0.13
    for s, hi in [("HOUR", 2), ("MON", 5), ("QRT", 4), ("YEAR", 8)]:
        d[f"AMT_REQ_CREDIT_BUREAU_{s}"] = np.where(nb, np.nan, r.randint(0, hi, n).astype(float))
    if with_target:
        d["TARGET"] = (r.rand(n) < 0.1 + 0.3 * (d.EXT_SOURCE_2.fillna(0.5) < 0.3)).astype(int)
    return d


@pytest.fixture
def train_df():
    return make_frame(600, seed=1)


@pytest.fixture
def test_df():
    return make_frame(200, seed=2, with_target=False, start_id=200000)
