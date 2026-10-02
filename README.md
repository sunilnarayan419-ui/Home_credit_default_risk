# Home Credit Default Risk: GCI Global 2026

[![CI](https://github.com/<your-username>/home-credit-default-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/home-credit-default-risk/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Reproducible solution for the in-class **Home Credit Default Risk** competition: predict the probability that a
customer has payment difficulties (`TARGET = 1`). The metric is **ROC-AUC**, so the output is a continuous probability.

| | 5-fold stratified CV ROC-AUC |
|---|---:|
| Tutorial baseline (5 features, Random Forest) | 0.670 |
| LightGBM on all raw features | 0.750 |
| + financial-ratio features + tuned small trees | 0.761 |
| **Final: equal-weight blend of LightGBM + XGBoost + CatBoost** | **0.762** |

CV uses early stopping on the validation fold, so scores are very slightly optimistic (equally for every row).
Full experiment log and the reasoning behind each decision: [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md).

## Competition rules and how this repo respects them

| Rule | How |
|---|---|
| No external data | Only `train.csv`, `test.csv`, `sample_submission.csv`; every feature is derived from them |
| No hand-labelling | Every test prediction comes from trained models (`predict_proba`), no manual rules |
| Reproducibility | Seed `42` everywhere, single-threaded boosting (`N_JOBS=1`); two clean runs produced byte-identical submissions |
| No target leakage | No target encoding or target-based selection. All features are row-wise (nothing is fitted on labels); category levels come from the training frame only. Unit tests check it |
| Submission integrity | Columns exactly `SK_ID_CURR,TARGET`, same IDs and order as `sample_submission.csv`, no index, no NaN, probabilities in [0, 1]; verified automatically after every run |

## Repository layout

```
.
├── src/hcdr/                  installable package
│   ├── config.py              seeds, paths, tuned hyper-parameters
│   ├── data.py                loading
│   ├── features.py            cleaning + feature engineering (5 switchable groups)
│   ├── models.py              LightGBM / XGBoost / CatBoost (+ scikit-learn fallback)
│   ├── validation.py          stratified CV, OOF predictions, blend scoring
│   ├── submission.py          writing + validating submission.csv
│   └── pipeline.py            end-to-end fit -> blend -> submission
├── scripts/                   command-line entry points
│   ├── audit.py               data audit (+ optional adversarial validation)
│   ├── run_cv.py              CV for any models / feature groups (ablations)
│   ├── tune.py                seeded random search
│   ├── train_predict.py       final fit -> output/submission.csv
│   └── validate_submission.py
├── notebooks/tutorial.ipynb   competition notebook (protected 6.2 cells untouched), for Omnicampus / Honors
├── experiments/               EXPERIMENTS.md + experiment_table.csv (35 recorded experiments)
├── tests/                     pytest suite on synthetic data (no competition data needed)
├── data/                      put the competition CSVs here (git-ignored)
├── output/                    submission.csv is written here (git-ignored)
├── .github/workflows/ci.yml   lint + tests on Python 3.10-3.12, plus a no-boosting-libs fallback job
├── Makefile  pyproject.toml  requirements*.txt  .pre-commit-config.yaml  LICENSE
```

## Quick start

```bash
git clone https://github.com/<your-username>/home-credit-default-risk.git
cd home-credit-default-risk
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
make install-dev                                        # or: pip install -r requirements-dev.txt && pip install -e . --no-deps
# copy train.csv, test.csv, sample_submission.csv into data/   (see data/README.md)

make test          # unit tests (synthetic data)
make audit         # data audit
make submission    # ~3 min on one core -> output/submission.csv (verified automatically)
make verify        # re-check an existing submission
```

Optional (long-running) experiment commands:

```bash
make cv                                              # 5-fold CV of the final blend (~20-30 min, 1 core)
python scripts/run_cv.py --models lgb --groups ratios   # feature-group ablation
python scripts/run_cv.py --models lgb --groups          # no engineered features
make tune                                            # random search over LightGBM params
```

No `make`? Every target is one plain `python scripts/...` command, see the `Makefile`.

## Method

1. **Audit** : 171,202 train / 61,500 test rows, 8.07% positives, no duplicate rows/IDs, identical train/test distribution
   (adversarial AUC 0.512). `DAYS_EMPLOYED = 365243` is a sentinel (18% of rows), `XNA` marks missing categories,
   `OWN_CAR_AGE` is 66% missing with implausible values >= 60.
2. **Cleaning** : sentinels -> NaN plus an anomaly flag. Trees receive NaNs natively because missingness is informative.
3. **Features** : 46 engineered features in 5 groups (ratios, time, external sources, bureau requests, misc). Ablations showed
   the financial ratios carry the gain; the others are neutral and kept. Categoricals use native categorical handling
   (no arbitrary integer codes).
4. **Validation** : stratified 5-fold CV, ROC-AUC mean +/- std and fold scores.
5. **Models** : strongly regularised small-tree boosters. Class weighting was tested and *hurt* (0.717), so it is not used.
6. **Ensemble** : equal-weight probability mean of LightGBM, XGBoost and CatBoost (+0.0007 AUC, better on 4/5 folds).
   Optimised weights and a Logistic-Regression member gave no gain, so they are not used.
7. **Final fit** : each model refit on all training rows with a fixed tree count (mean CV best iteration x 1.1).

If a boosting library is not installed the pipeline falls back to scikit-learn `HistGradientBoostingClassifier`
(works, but weaker: 0.7514 vs 0.7580 AUC on one fold; untuned).

## Reproducing the submission

```bash
pip install -r requirements.txt && pip install -e . --no-deps
python scripts/train_predict.py --data-dir path/to/folder-with-the-csvs
python scripts/validate_submission.py --data-dir path/to/folder-with-the-csvs
```

For the Omnicampus Honors/Outstanding requirement use `notebooks/tutorial.ipynb`: place it next to the competition's
`input/` folder (or edit the Colab `%cd` path in cell 1.2), run all cells, then submit `output/submission.csv` and the
`tutorial.zip` that its last cell creates.

## Development

```bash
pre-commit install      # optional: ruff + hygiene hooks on every commit
make lint test
```

CI never sees competition data; it runs the unit tests on synthetic frames that follow the real schema.

## License

MIT, see [`LICENSE`](LICENSE). The competition dataset itself remains subject to the competition's terms and is not redistributed here.
