.PHONY: install install-dev lint test coverage check docs docs-serve train-model api audit cv ablation tune submission verify notebook clean all

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps

install-dev:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps

lint:
	ruff check src scripts tests

test:
	pytest

coverage:      ## pytest with the coverage threshold from pyproject.toml
	pytest --cov --cov-report=term-missing

check:         ## same essential checks as CI: ruff + pytest --cov + mkdocs build --strict
	$(PY) scripts/check_project.py

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

train-model:   ## fit on all training rows and save output/model_bundle.joblib for the API
	$(PY) scripts/train_model.py

api:           ## needs output/model_bundle.joblib (make train-model)
	uvicorn hcdr.api.app:app

audit:
	$(PY) scripts/audit.py

cv:            ## 5-fold CV of the final LGB+XGB+CatBoost blend (about 20-30 min on 1 core)
	$(PY) scripts/run_cv.py

ablation:      ## one feature group at a time, LightGBM only
	$(PY) scripts/run_cv.py --models lgb --groups
	@for g in ratios ext time bureau misc; do echo "== $$g"; $(PY) scripts/run_cv.py --models lgb --groups $$g; done

tune:          ## seeded random search (long)
	$(PY) scripts/tune.py

submission:    ## fit on all training rows -> output/submission.csv (about 3 min)
	$(PY) scripts/train_predict.py

verify:
	$(PY) scripts/validate_submission.py

notebook:      ## execute the competition notebook top to bottom (run from a folder that has input/)
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace tutorial.ipynb

all: install-dev lint test submission verify

clean:
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov site build dist src/*.egg-info catboost_info
