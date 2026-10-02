.PHONY: install install-dev lint test audit cv ablation tune submission verify notebook clean all

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps

install-dev:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps

lint:
	ruff check src scripts tests

test:
	pytest

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
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info catboost_info
