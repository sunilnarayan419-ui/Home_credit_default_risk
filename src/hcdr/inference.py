"""Fitted-model bundle: training, persistence and scoring of new raw application rows.

A bundle holds everything inference needs that the batch pipeline normally rebuilds on every run: the fitted
models, the category levels learned from the training frame, and the exact raw/feature column order.
Scoring reuses the *same* ``build_matrix`` / ``apply_categories`` / ``predict_proba`` code as the batch pipeline,
so serving and the competition submission cannot drift apart.

Security: bundles are joblib (pickle) files. Only load artifacts you produced yourself.
"""
from __future__ import annotations

import datetime as dt
import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .config import CAT_COLS, FEATURE_GROUPS, ID, TARGET
from .features import apply_categories, build_matrix, fit_categories
from .models import available_models, fit_model, predict_proba

FORMAT_VERSION = 1
_REQUIRED_KEYS = ("format_version", "models", "categories", "raw_columns", "feature_columns", "groups", "metadata")


class ArtifactError(Exception):
    """The model artifact is missing, unreadable, or incompatible with this version of the code."""


class InferenceInputError(ValueError):
    """The rows passed for scoring do not match the model's raw-input contract."""


@dataclass
class ModelBundle:
    models: dict[str, Any]
    categories: dict[str, list]
    raw_columns: list[str]  # raw input columns, in training order (no ID / TARGET)
    feature_columns: list[str]  # model input columns after feature engineering, in training order
    groups: tuple[str, ...] = FEATURE_GROUPS
    metadata: dict = field(default_factory=dict)


def _library_versions(names: list[str]) -> dict[str, str]:
    libs = {"hgb": "scikit-learn", "lgb": "lightgbm", "xgb": "xgboost", "cat": "catboost"}
    wanted = {"pandas", "numpy", "scikit-learn"} | {libs[n] for n in names}
    return {lib: importlib.metadata.version(lib) for lib in sorted(wanted)}


def fit_bundle(train: pd.DataFrame, models: list[str] | None = None, n_iter: dict[str, int] | None = None,
               force_fallback: bool = False) -> ModelBundle:
    """Fit every model on ALL rows of ``train`` (fixed tree counts, as in the final submission fit)."""
    names = models or available_models(force_fallback)
    X = build_matrix(train)
    categories = fit_categories(X)
    X = apply_categories(X, categories)
    y = train[TARGET].to_numpy()
    fitted = {m: fit_model(m, X, y, (n_iter or {}).get(m)) for m in names}
    metadata = {
        "hcdr_version": __version__,
        "models": names,
        "blend": "equal-weight mean of predicted probabilities",
        "n_train_rows": int(len(train)),
        "n_raw_columns": int(len([c for c in train.columns if c not in (ID, TARGET)])),
        "n_features": int(X.shape[1]),
        "feature_groups": list(FEATURE_GROUPS),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "library_versions": _library_versions(names),
    }
    raw = [c for c in train.columns if c not in (ID, TARGET)]
    return ModelBundle(fitted, categories, raw, list(X.columns), tuple(FEATURE_GROUPS), metadata)


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION, "models": bundle.models, "categories": bundle.categories,
        "raw_columns": bundle.raw_columns, "feature_columns": bundle.feature_columns,
        "groups": tuple(bundle.groups), "metadata": bundle.metadata,
    }
    joblib.dump(payload, path)
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    """Load and structurally validate a bundle. Raises :class:`ArtifactError` for any problem."""
    path = Path(path)
    if not path.is_file():
        raise ArtifactError("model artifact not found")
    try:
        payload = joblib.load(path)
    except Exception as exc:  # joblib/pickle raise many unrelated types for a corrupt or foreign file
        raise ArtifactError("model artifact could not be read") from exc
    if not isinstance(payload, dict) or any(k not in payload for k in _REQUIRED_KEYS):
        raise ArtifactError("model artifact has an unexpected structure")
    if payload["format_version"] != FORMAT_VERSION:
        raise ArtifactError(f"unsupported artifact format version {payload['format_version']!r}")
    if not payload["models"] or set(payload["categories"]) != set(CAT_COLS):
        raise ArtifactError("model artifact content is inconsistent")
    return ModelBundle(payload["models"], payload["categories"], list(payload["raw_columns"]),
                       list(payload["feature_columns"]), tuple(payload["groups"]), dict(payload["metadata"]))


def predict_default_probability(bundle: ModelBundle, rows: pd.DataFrame) -> np.ndarray:
    """Equal-weight blend probability of TARGET = 1 for each raw application row (same order as ``rows``)."""
    if rows.empty:
        raise InferenceInputError("no rows to score")
    cols = set(rows.columns) - {ID}
    unknown, missing = sorted(cols - set(bundle.raw_columns)), sorted(set(bundle.raw_columns) - cols)
    if unknown or missing:
        raise InferenceInputError(f"input columns do not match the model contract (unknown: {unknown}, missing: {missing})")
    raw = rows[bundle.raw_columns].copy()  # training column order, ID dropped
    try:
        for c in raw.columns.difference(CAT_COLS):
            raw[c] = pd.to_numeric(raw[c])
    except (ValueError, TypeError) as exc:
        raise InferenceInputError("non-numeric value in a numeric column") from exc
    X = apply_categories(build_matrix(raw, bundle.groups), bundle.categories)
    if list(X.columns) != bundle.feature_columns:
        raise InferenceInputError("engineered feature columns differ from those the model was trained on")
    return np.mean([predict_proba(name, m, X) for name, m in bundle.models.items()], axis=0)

