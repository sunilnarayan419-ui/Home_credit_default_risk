# Development and CI

## One command

```bash
python scripts/check_project.py
```

Runs, from the repository root and stopping at the first failure:

1. `ruff check src scripts tests`
2. `pytest --cov` (with the coverage threshold)
3. `mkdocs build --strict`

`make check` is an alias. Other useful targets: `make lint`, `make test`, `make docs`, `make docs-serve`,
`make train-model`, `make api`.

## Documentation

```bash
mkdocs serve            # live preview
mkdocs build --strict   # fails on broken links or configuration errors
```

The site is not deployed automatically.

## GitHub Actions

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests and manual dispatch with read-only
permissions and no secrets:

| Job | What it does |
|---|---|
| `test` | Python 3.10, 3.11, 3.12: install, `ruff check`, `pytest --cov` (threshold enforced); uploads `coverage.xml` from 3.12 |
| `docs` | `mkdocs build --strict` |
| `fallback` | Core tests with only scikit-learn, proving the pipeline works without the boosting libraries |

CI never downloads or needs the competition data.

## Conventions

* Ruff (`E, F, I, W, B`, line length 140) is enforced; no repository-wide reformatting is applied.
* New behaviour comes with synthetic-data tests; never commit real customer data or model artifacts.
* `pre-commit install` adds ruff and hygiene hooks locally.
