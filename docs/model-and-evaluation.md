# Model and evaluation

## Final model

An equal-weight mean of the predicted probabilities of three gradient-boosting models, each a small,
strongly regularised tree ensemble with native categorical handling. Hyper-parameters live in
`src/hcdr/config.py`; the final fit uses fixed tree counts (`FINAL_ITERS`, the mean CV best iteration × 1.1).
Class weighting was tested and hurt (CV AUC 0.717), so it is not used.

## Validation method

Stratified 5-fold cross-validation (`shuffle=True`, `random_state=42`), reporting ROC-AUC as mean ± standard
deviation across folds. Early stopping on the validation fold makes every score slightly optimistic by about the
same amount.

## Results reported by the project

These figures come from the project's own experiment log; they have not been independently re-run for this
documentation.

| Stage | CV ROC-AUC |
|---|---:|
| Tutorial baseline (5 features, Random Forest) | 0.6702 |
| LightGBM, 32 raw features | 0.7500 |
| + financial ratios | 0.7587 |
| + tuned small trees | 0.7614 |
| **Blend of LightGBM + XGBoost + CatBoost** | **0.7622** |

The full table is in the
[experiment log](https://github.com/sunilnarayan419-ui/Home_credit_default_risk/blob/main/experiments/EXPERIMENTS.md).

## Reading the metrics

* **Class imbalance.** Only about 8 % of applicants are positives, so accuracy is misleading; a model that
  always predicts 0 is about 92 % "accurate".
* **ROC-AUC** is the probability that a random positive is scored above a random negative. It measures ranking
  only and says nothing about whether a score of 0.2 means a 20 % default rate.
* **PR-AUC and precision/recall** depend strongly on the positive rate and the chosen threshold, which matter in
  lending. They are not computed by this project; add them before choosing an operating threshold.
* **Calibration.** Nothing here shows the blend is calibrated. Do not read scores as literal probabilities.
* A high ROC-AUC on one competition dataset is not evidence of business readiness.

## Reproducing a run

```bash
python scripts/train_predict.py --data-dir path/to/folder
python scripts/validate_submission.py --data-dir path/to/folder
```

Seed 42 and single-threaded boosting make repeated runs on the same library versions repeatable; the project
reports byte-identical submissions from two clean runs. Different library versions can change results.
