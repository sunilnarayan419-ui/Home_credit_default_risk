# Data and features

## Dataset

The competition provides `train.csv`, `test.csv` and `sample_submission.csv`. The data is subject to the
competition's terms and is not redistributed here; this project does not document its licence beyond that.

Findings from the project's own data audit (`make audit`):

* 171,202 training rows and 61,500 test rows, 8.07 % positives (`TARGET = 1`, payment difficulties).
* No duplicate rows or IDs; train and test distributions are alike (adversarial-validation AUC 0.512).
* `DAYS_EMPLOYED = 365243` is a sentinel (18 % of rows); `XNA` marks missing categories.
* `OWN_CAR_AGE` is 66 % missing and contains implausible values of 60 or more.

## Raw input columns

The models use 32 raw columns (everything except `SK_ID_CURR` and `TARGET`):

| Kind | Columns |
|---|---|
| Categorical (8) | `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE` |
| Amounts | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` |
| Time (`DAYS_*`, negative = days before the application) | `DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE` |
| External scores | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| Credit-bureau requests | `AMT_REQ_CREDIT_BUREAU_HOUR`, `_MON`, `_QRT`, `_YEAR` |
| Other | `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `REGION_POPULATION_RELATIVE`, `REGION_RATING_CLIENT`, `OWN_CAR_AGE`, `FLAG_MOBIL`, `FLAG_EMP_PHONE`, `FLAG_WORK_PHONE` |

The full competition data dictionary is distributed with the competition, not with this repository.

## Cleaning

* `DAYS_EMPLOYED == 365243` becomes missing and a `DAYS_EMPLOYED_ANOM` flag records it.
* `XNA` in `ORGANIZATION_TYPE` and `CODE_GENDER` becomes missing.
* Missing values are left as `NaN`: the tree models handle them natively and missingness can be informative.

## Feature engineering

The anomaly flag and five feature groups add 46 columns to the 32 raw ones, giving 78 model inputs:

| Group | Examples |
|---|---|
| `ratios` | `CREDIT_INCOME`, `ANNUITY_INCOME`, `CREDIT_TERM`, per-family-member amounts |
| `time` | `AGE_YEARS`, `EMPLOYED_YEARS`, ratios of `DAYS_*` to age |
| `ext` | mean, min, max, std, product and differences of the three external scores |
| `bureau` | total and recent bureau-request counts, missingness flag |
| `misc` | car-age outlier flag, `HAS_CAR`, `FLAG_SUM`, `CHILD_RATIO` |

Division by zero is converted to missing, never infinity. Ablations in the
[experiment log](https://github.com/sunilnarayan419-ui/Home_credit_default_risk/blob/main/experiments/EXPERIMENTS.md)
showed the `ratios` group carries the gain; the other groups are roughly neutral and were kept.

## Leakage controls

`TARGET` and `SK_ID_CURR` are removed before modelling, nothing learns statistics from data, and category
levels come from the training frame only. The API additionally rejects `TARGET` and `SK_ID_CURR` as inputs.
