# Architecture

```text
src/hcdr/
├── config.py        seeds, paths, categorical columns, tuned hyper-parameters, FINAL_ITERS
├── data.py          load train/test/sample_submission (data/ or data/input/)
├── features.py      clean -> add_features -> build_matrix; category handling
├── models.py        model factory: fit_model, predict_proba, fit_predict (CV with early stopping)
├── validation.py    stratified K-fold CV and summaries
├── pipeline.py      train_and_predict (submission) and train_and_save (model artifact)
├── submission.py    write and validate submission.csv
├── inference.py     ModelBundle: fit, save, load and score raw application rows
└── api/             FastAPI app (app.py) and Pydantic schemas (schemas.py)
scripts/             thin command-line entry points (train_predict, run_cv, train_model, ...)
tests/               pytest suite on synthetic frames (no competition data)
experiments/         experiment log and results table
notebooks/           competition tutorial notebook
```

## Two workflows, one code path

* **Batch (competition):** `scripts/train_predict.py` reads the CSVs, builds features, fits each model with a
  fixed tree count on all training rows and writes `output/submission.csv`.
* **Serving:** `scripts/train_model.py` runs the same fit, but saves the fitted models together with the
  category levels and the exact column order. The API loads that file once at startup.

Scoring in the API calls the same `build_matrix`, `apply_categories` and `predict_proba` functions as the batch
pipeline. A test (`test_serving_path_matches_batch_pipeline`) asserts that both paths return identical scores
for the same model and rows.

## Design decisions

* **No fitted preprocessing statistics.** Every transform is row-wise. The only fitted object is the list of
  category levels, learned from the training frame only (no labels).
* **Single-threaded boosting** (`N_JOBS = 1`) and seed 42 for run-to-run repeatability.
* **Graceful degradation.** If LightGBM, XGBoost or CatBoost are not installed the pipeline falls back to
  scikit-learn `HistGradientBoostingClassifier`.
* **API without a model still starts.** `/health` reports `model_loaded: false` and prediction routes return 503.
