# Getting started

## Install

Python 3.10 or newer is tested in CI (3.10, 3.11, 3.12).

```bash
git clone https://github.com/sunilnarayan419-ui/Home_credit_default_risk.git
cd Home_credit_default_risk
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
make install-dev       # or: pip install -r requirements-dev.txt && pip install -e . --no-deps
```

| Requirements file | Contents |
|---|---|
| `requirements.txt` | Runtime: numpy, pandas, scikit-learn, LightGBM, XGBoost, CatBoost |
| `requirements-api.txt` | Runtime plus FastAPI, uvicorn, pydantic (needed only for the API) |
| `requirements-docs.txt` | MkDocs Material |
| `requirements-dev.txt` | Everything above plus pytest, pytest-cov, httpx, ruff, notebook tooling |

## Data setup

Copy the three competition files into `data/` (they are git-ignored):

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

An `input/` sub-folder (as in the competition ZIP) also works. Use another location with
`HCDR_DATA_DIR=/path/to/folder` or the `--data-dir` flag of every script.

## Run the existing pipeline

```bash
make audit         # data audit
make submission    # fit on all training rows -> output/submission.csv, verified automatically
make verify        # re-check an existing submission
make cv            # 5-fold CV of the blend (long)
```

## Train the model artifact and start the API

```bash
python scripts/train_model.py                      # -> output/model_bundle.joblib (git-ignored)
uvicorn hcdr.api.app:app --reload                  # http://127.0.0.1:8000
```

Interactive documentation is then served at `http://127.0.0.1:8000/docs` and the schema at
`/openapi.json`. Details: [API reference](api-reference.md).

## Run the checks and the docs

```bash
python scripts/check_project.py     # ruff + pytest with coverage + mkdocs build --strict
mkdocs serve                        # live preview at http://127.0.0.1:8000 (add -a 127.0.0.1:8001 if the API is running)
mkdocs build --strict
```
