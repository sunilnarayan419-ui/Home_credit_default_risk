# Home Credit Default Risk — GCI Global 2026

[![CI](https://github.com/sunilnarayan419-ui/Home_credit_default_risk/actions/workflows/ci.yml/badge.svg)](https://github.com/sunilnarayan419-ui/Home_credit_default_risk/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-informational)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage%20threshold-90%25-success)](https://coverage.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-MkDocs%20Material-526cfe)](https://squidfunk.github.io/mkdocs-material/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A reproducible machine-learning project for the **Home Credit Default Risk** in-class competition. It estimates a model score for payment difficulties (`TARGET = 1`) from applicant-level features. The repository combines data preparation, feature engineering, gradient-boosting models, cross-validation, model-artifact persistence, a validated FastAPI inference service, automated testing, documentation, and continuous integration.

> [!WARNING]
> **Educational/competition use only.** This repository is not a credit-decision system and must not be used to approve, reject, price, or otherwise make decisions about real applicants. A strong ROC-AUC score does not establish calibration, fairness, regulatory compliance, explainability, or suitability for lending.

## Contents

- [Results](#results)
- [Capabilities](#capabilities)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Quick start — Windows and VS Code](#quick-start--windows-and-vs-code)
- [Testing and code quality](#testing-and-code-quality)
- [GitHub Actions CI](#github-actions-ci)
- [Documentation with MkDocs Material](#documentation-with-mkdocs-material)
- [Training and submission generation](#training-and-submission-generation)
- [FastAPI inference service](#fastapi-inference-service)
- [Modeling workflow](#modeling-workflow)
- [Configuration and reproducibility](#configuration-and-reproducibility)
- [Security and responsible use](#security-and-responsible-use)
- [Known limitations](#known-limitations)
- [License](#license)

## Results

Recorded five-fold stratified cross-validation results:

| Experiment | Mean ROC-AUC |
|---|---:|
| Tutorial baseline: five features + Random Forest | 0.670 |
| LightGBM on raw features | 0.750 |
| LightGBM with financial-ratio features and tuned small trees | 0.761 |
| **Equal-weight LightGBM + XGBoost + CatBoost blend** | **0.762** |

These figures are historical results on the competition dataset, not a guarantee of performance on unseen populations. Cross-validation uses early stopping on validation folds; consult the experiment notes and model-evaluation documentation before interpreting small differences.

## Capabilities

- **Data validation and preprocessing:** dataset-specific sentinel handling, missing-value processing, categorical handling, and feature preparation.
- **Feature engineering:** financial ratios and additional feature groups used by the model pipeline.
- **Modeling:** LightGBM, XGBoost, and CatBoost with a scikit-learn `HistGradientBoostingClassifier` fallback when the optional boosting libraries are unavailable.
- **Evaluation:** stratified cross-validation and ROC-AUC-based model comparison.
- **Inference artifacts:** serializes fitted models, learned categorical levels, feature-column contracts, and metadata into a reusable model bundle.
- **REST API:** FastAPI endpoints for health, model metadata, single-record predictions, and batch predictions.
- **Input contracts:** Pydantic schemas validate request shape and fields before inference.
- **Automated testing:** pytest unit/integration/API tests using synthetic data; coverage is configured to fail below 90%.
- **Code quality:** Ruff linting for the source, scripts, and tests.
- **Documentation:** MkDocs Material site with strict-build validation.
- **Continuous integration:** GitHub Actions checks multiple Python versions, runs tests and coverage, builds the documentation, and includes a fallback-dependency job.

## Technology stack

| Area | Tools |
|---|---|
| Language | Python |
| Data and feature work | NumPy, pandas |
| Machine learning | scikit-learn, LightGBM, XGBoost, CatBoost |
| API and schemas | FastAPI, Uvicorn, Pydantic |
| Tests and coverage | pytest, pytest-cov, HTTPX / FastAPI TestClient |
| Linting | Ruff |
| Documentation | MkDocs, MkDocs Material |
| CI | GitHub Actions |
| Packaging/configuration | `pyproject.toml`, requirements files, setuptools |
| Model persistence | joblib (pickle-based serialization) |

## Repository structure

```text
Home_credit_default_risk/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions: lint, tests/coverage, docs, fallback job
├── docs/                          # MkDocs source documentation
│   ├── index.md
│   ├── getting-started.md
│   ├── architecture.md
│   ├── data-and-features.md
│   ├── model-and-evaluation.md
│   ├── api-reference.md
│   ├── testing.md
│   ├── development-guide.md
│   ├── limitations.md
│   └── troubleshooting.md
├── scripts/
│   ├── check_project.py           # Local quality gate: Ruff + pytest coverage + strict docs build
│   └── train_model.py             # Train and save the inference model bundle
├── src/
│   └── hcdr/
│       ├── __init__.py
│       ├── config.py              # Seeds, paths, feature/model configuration
│       ├── data.py                # Competition data loading
│       ├── features.py            # Cleaning and feature engineering
│       ├── models.py              # Model selection, fitting, and prediction helpers
│       ├── validation.py          # Cross-validation and evaluation helpers
│       ├── submission.py          # Submission creation and validation
│       ├── pipeline.py            # End-to-end training and prediction workflows
│       ├── inference.py           # Bundle persistence and inference
│       └── api/
│           ├── __init__.py
│           ├── app.py             # FastAPI routes and application lifecycle
│           └── schemas.py         # Pydantic request/response contracts
├── tests/
│   ├── conftest.py               # Shared synthetic fixtures
│   ├── test_preprocessing.py     # Cleaning and feature tests
│   ├── test_boosters.py          # Model/booster tests
│   ├── test_inference.py         # Bundle and prediction tests
│   ├── test_pipeline.py          # End-to-end pipeline tests
│   └── api/
│       ├── test_api_contract.py
│       ├── test_health.py
│       └── test_prediction_endpoint.py
├── .gitignore                    # Excludes local data, artifacts, envs, and caches
├── Makefile                      # Convenience targets for common developer tasks
├── mkdocs.yml                    # MkDocs Material theme, plugins, and navigation
├── pyproject.toml                # Package metadata and pytest/Ruff/coverage configuration
├── requirements-api.txt          # API/runtime dependency group
├── requirements-dev.txt          # Development and test dependencies
├── requirements-docs.txt         # Documentation dependencies
├── LICENSE
└── README.md
```

> Keep this tree synchronized with the actual files in your checkout. If your local repository contains additional scripts, notebooks, experiment logs, or requirements files, add them here only after confirming their names and purpose. Competition data and generated model artifacts should remain untracked.

## Quick start — Windows and VS Code

### 1. Clone and open the repository

```powershell
git clone https://github.com/sunilnarayan419-ui/Home_credit_default_risk.git
cd Home_credit_default_risk
code .
```

Open **Terminal → New Terminal** in VS Code. Run the following commands from the repository root.

### 2. Create and activate a virtual environment

Python 3.10–3.12 is covered by the configured CI matrix.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `py -3.12` is unavailable, use an installed compatible interpreter with `python -m venv .venv`. If PowerShell blocks activation, switch the VS Code terminal profile to Command Prompt and run:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install development dependencies and the package

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

The development requirements bring in the API, documentation, testing, and linting dependencies. Installing the package in editable mode makes the `hcdr` package importable from the environment.

### 4. Run the tests

```powershell
python -m pytest
```

Run with coverage and the configured 90% minimum:

```powershell
python -m pytest --cov --cov-report=term-missing
```

A recent local run passed **80 tests with 96.02% coverage**. This is a snapshot of that run, not a permanent guarantee; the result can change when code, tests, or dependencies change.

## Testing and code quality

### One-command quality gate

```powershell
python scripts/check_project.py
```

The checker runs the repository's main local quality checks and stops at the first failure:

1. **Ruff:** `ruff check src scripts tests`
2. **pytest + coverage:** `pytest --cov --cov-report=term-missing` (coverage must be at least 90%)
3. **MkDocs:** `mkdocs build --strict`

Run each check separately when debugging:

```powershell
python -m ruff check src scripts tests
python -m pytest --cov --cov-report=term-missing
python -m mkdocs build --strict
```

### Useful test commands

```powershell
# Entire test suite
python -m pytest

# Coverage details, including lines not executed
python -m pytest --cov --cov-report=term-missing

# API tests only
python -m pytest tests/api/

# Tests marked as neither API nor integration
python -m pytest -m "not api and not integration"
```

Tests use synthetic fixtures and should not require competition CSVs. Coverage indicates which code was executed; it does not prove that the model is correct or free from leakage.

## GitHub Actions CI

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

The workflow is configured to run on:

- pushes to `main`;
- pull requests targeting `main`;
- manual execution through **Actions → CI → Run workflow** (`workflow_dispatch`).

### CI jobs

| Job | Environment | What it checks |
|---|---|---|
| `test` | Ubuntu; Python 3.10, 3.11, and 3.12 | Installs development dependencies, runs Ruff, executes pytest with coverage, and uploads `coverage.xml` as an artifact on Python 3.12 when available. |
| `docs` | Ubuntu; Python 3.12 | Installs documentation dependencies and runs `mkdocs build --strict`. |
| `fallback` | Ubuntu; Python 3.11 | Intended to test the scikit-learn fallback path without installing LightGBM, XGBoost, or CatBoost. |

The workflow uses pip caching, a read-only repository permission, a job timeout, and concurrency cancellation for superseded runs. CI tests use synthetic data; competition data should not be committed or required as a CI secret.

**Before relying on CI:** verify that the test paths named in `.github/workflows/ci.yml` still exist in your current checkout. When test files are renamed, update the fallback job in the same commit; otherwise, the main test job can pass while the fallback job fails because of stale paths.

## Documentation with MkDocs Material

The documentation site is configured in [`mkdocs.yml`](mkdocs.yml) and authored in [`docs/`](docs/). It uses the Material theme, built-in search, section navigation, code-copy controls, light/dark palettes, and Markdown extensions for tables, admonitions, tabs, and code blocks.

Build the site and fail on strict-mode issues:

```powershell
python -m mkdocs build --strict
```

Serve a live local preview:

```powershell
python -m mkdocs serve
```

Open the local address printed in the terminal. The generated `site/` directory is a build artifact and is ignored by Git. **CI builds the documentation; it does not automatically publish it to GitHub Pages.** A public documentation URL is not configured by this repository's workflow.

## Training and submission generation

The competition dataset is not included in this repository. Obtain it from the authorized course/competition source and comply with its terms. Keep the files local and untracked.

Expected data files:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

The repository's `.gitignore` excludes data and generated output contents. Check each script's help output for the exact supported options:

```powershell
python scripts/train_model.py --help
```

The training workflow uses the competition data to fit the final models and save a model bundle for inference. In the full competition workflow, submission generation and validation should also be run using the submission-related scripts available in your checkout. Do not assume a script exists solely because it is mentioned in an older README or Makefile; verify it with `Get-ChildItem scripts`.

## FastAPI inference service

### 1. Train and save the model artifact

With the competition CSVs in the configured data directory:

```powershell
python scripts/train_model.py
```

The default artifact path is `output/model_bundle.joblib`. The artifact contains fitted model objects, categorical levels, raw and engineered feature-column contracts, and metadata.

### 2. Start the API

```powershell
python -m uvicorn hcdr.api.app:app --reload
```

Open the local service:

- Interactive Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI schema: <http://127.0.0.1:8000/openapi.json>
- Health endpoint: <http://127.0.0.1:8000/health>

### Endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Reports service health and whether a model bundle is loaded. |
| `GET` | `/model-info` | Returns safe metadata about the loaded model. |
| `POST` | `/predict` | Scores one application record. |
| `POST` | `/predict/batch` | Scores a batch of up to 1,000 records. |

Request bodies must match the Pydantic contracts in `src/hcdr/api/schemas.py`. Invalid inputs are rejected by schema validation. If a usable model bundle is not loaded, prediction endpoints return HTTP `503`. The returned score is between 0 and 1 but is **not proven to be a calibrated real-world probability**.

### API configuration and security

- The model artifact path can be configured through the `HCDR_MODEL_PATH` environment variable if supported by the API configuration in your checkout.
- `joblib` uses pickle-based serialization. **Never load a model artifact from an untrusted source.** Deserialization can execute code.
- The demonstration API has no authentication by default. Do not expose it publicly without authentication, authorization, request limits, transport security, monitoring, and deployment hardening.

## Modeling workflow

At a high level, the pipeline is:

1. **Load and audit** the supplied competition tables.
2. **Clean and prepare** features, including dataset-specific sentinels and missing values.
3. **Engineer features** such as financial ratios and other configured feature groups.
4. **Train and validate** candidate estimators using stratified cross-validation and ROC-AUC.
5. **Blend** model outputs using the recorded equal-weight ensemble configuration.
6. **Refit and persist** the model bundle for repeatable inference.
7. **Serve predictions** through the API using the same feature preparation and column contract.

See [`docs/architecture.md`](docs/architecture.md), [`docs/data-and-features.md`](docs/data-and-features.md), and [`docs/model-and-evaluation.md`](docs/model-and-evaluation.md) for more detail.

## Configuration and reproducibility

- Model configuration, feature groups, paths, and seeds are centralized in `src/hcdr/config.py`.
- Test, Ruff, and coverage settings are declared in `pyproject.toml`.
- Coverage uses branch measurement and is configured with `fail_under = 90`.
- CI runs tests across Python 3.10–3.12.
- Exact reproducibility can depend on Python and third-party library versions, hardware, and native-library behavior. Record the environment used for important experiments.
- Avoid committing local virtual environments, competition data, `.env` files, generated model bundles, coverage reports, or build directories.

## Security and responsible use

This project uses applicant attributes to estimate payment-difficulty risk in a competition setting. Credit-related datasets may include sensitive attributes or proxies. A high ranking metric alone is not evidence of fair, lawful, explainable, calibrated, or appropriate decisions.

Before any real-world use, a system would require independent external validation, calibration analysis, subgroup fairness evaluation, leakage review, explainability, privacy and security assessment, regulatory/legal review, drift monitoring, and accountable human governance. This repository is not a substitute for those controls.

## Known limitations

- Results are based on one in-class competition dataset; external generalization has not been established.
- Probability calibration, subgroup fairness, operational drift, and regulatory suitability are not established by the recorded ROC-AUC results.
- The API is a local demonstration service and has no authentication by default.
- Model artifacts are trusted-input-only because of pickle-based serialization.
- Test coverage measures executed code, not correctness, predictive validity, fairness, or safety.

See [`docs/limitations.md`](docs/limitations.md) and [`docs/troubleshooting.md`](docs/troubleshooting.md) for further context.

## License

MIT. See [`LICENSE`](LICENSE). The competition dataset is not redistributed by this repository and remains subject to the applicable competition/course terms.
