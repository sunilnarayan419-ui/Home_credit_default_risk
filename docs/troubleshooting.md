# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `FileNotFoundError: ... train.csv` | Data not in `data/` (or `data/input/`). Set `HCDR_DATA_DIR` or pass `--data-dir`. |
| `/health` shows `"status": "degraded"` | No usable model artifact. Run `python scripts/train_model.py`, or set `HCDR_MODEL_PATH`. The server log says whether the file was missing or invalid. |
| `/predict` returns 503 | Same as above. |
| `/predict` returns 422 | Body does not match the schema. The `detail` list gives the field `loc` and message; unknown fields (including `TARGET`) are rejected. |
| Artifact rejected after upgrading the code | The artifact format or raw columns changed; retrain with `scripts/train_model.py`. |
| Different scores from another machine | Library versions differ; compare `/model-info` `library_versions`. |
| Tests for LightGBM/XGBoost/CatBoost skipped | The library is not installed; install `requirements.txt`. |
| `mkdocs build --strict` fails | Fix the reported broken link or nav entry; strict mode turns warnings into errors. |
| `ModuleNotFoundError: hcdr` | Run `pip install -e . --no-deps` (or use `pytest`, which adds `src` to the path). |
