# Home Credit Default Risk

This project predicts the probability that a loan applicant has payment difficulties (`TARGET = 1`) for the
in-class **Home Credit Default Risk** competition (GCI Global 2026). The competition metric is **ROC-AUC**, so
the output is a continuous score that ranks applicants rather than a hard yes/no label.

The final model is an equal-weight blend of **LightGBM, XGBoost and CatBoost** trained on cleaned raw
columns plus engineered features. The repository reports a 5-fold stratified cross-validation ROC-AUC of
**0.762** for it (see [Model and evaluation](model-and-evaluation.md)).

## What is in the repository

| Part | Purpose | Start here |
|---|---|---|
| Batch pipeline | Reproduce the competition `submission.csv` | [Getting started](getting-started.md) |
| Inference API | Score single applications or small batches with a saved model | [API reference](api-reference.md) |
| Test suite | Synthetic-data tests, coverage threshold | [Testing](testing.md) |
| CI and local checks | Ruff, pytest, strict docs build | [Development and CI](development-guide.md) |

!!! warning "Not a credit-decision system"
    Scores come from a competition dataset and were validated only with cross-validated ROC-AUC. They are not
    shown to be calibrated, fair across groups or fit for operational lending. Read
    [Limitations and responsible use](limitations.md) before using any output outside the competition.

## Data availability

The competition files are **not** in the repository and must not be redistributed. See
[Getting started](getting-started.md#data-setup).
