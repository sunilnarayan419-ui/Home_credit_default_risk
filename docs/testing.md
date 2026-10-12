# Testing

```bash
pytest                                  # whole suite
pytest -m "not api and not integration" # unit tests only
pytest --cov                            # with coverage (threshold from pyproject.toml)
```

All tests use **synthetic** frames that follow the real 32-column schema (`tests/conftest.py`). They need no
competition data, no network and no real customer information, and train throw-away models of a few trees.

| File | Covers |
|---|---|
| `test_features.py`, `test_preprocessing.py` | cleaning, feature groups, missing values, zero denominators, category handling, data loading |
| `test_models.py`, `test_boosters.py` | model factory; LightGBM, XGBoost and CatBoost fit/predict and CV paths (skipped if a library is absent) |
| `test_inference.py` | bundle fit/save/load, column-contract errors, **serving equals batch pipeline** |
| `test_submission.py` | submission writing and validation |
| `test_pipeline.py` (`integration`) | CSVs → submission and CSVs → artifact → scores |
| `tests/api/` (`api`) | health, prediction, validation errors, missing model, no echo of inputs, schema/feature contract |

## Coverage

`pytest --cov` measures the `hcdr` package with branch coverage and fails below the threshold in
`pyproject.toml` (`fail_under = 90`; about 96 % was measured when the threshold was set). Coverage shows which code
ran, not that the model is correct: a passing run says nothing about predictive quality or fairness.

## Markers

* `integration`: multi-component workflows on tiny data.
* `api`: FastAPI `TestClient` tests.
