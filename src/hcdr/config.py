"""Central configuration: seeds, paths, categorical columns and the tuned model hyper-parameters."""
from __future__ import annotations

import os
from pathlib import Path

SEED = 42
N_SPLITS = 5
N_JOBS = 1  # single-threaded boosting -> repeatable run-to-run

TARGET = "TARGET"
ID = "SK_ID_CURR"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("HCDR_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("HCDR_OUTPUT_DIR", ROOT / "output"))

CAT_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
]
FEATURE_GROUPS = ("ratios", "time", "ext", "bureau", "misc")

# --- hyper-parameters selected by cross-validation (see experiments/EXPERIMENTS.md) ---
LGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.05, num_leaves=7, min_child_samples=100, subsample=1.0,
    subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0, reg_alpha=10.0, min_split_gain=0.0,
    max_bin=255, random_state=SEED, n_jobs=N_JOBS, verbose=-1,
)
XGB_PARAMS = dict(
    max_depth=3, learning_rate=0.05, n_estimators=3000, subsample=0.9, colsample_bytree=0.3,
    min_child_weight=20, reg_lambda=10, reg_alpha=1, tree_method="hist", enable_categorical=True,
    max_cat_to_onehot=8, n_jobs=N_JOBS, random_state=SEED, eval_metric="auc",
)
CAT_PARAMS = dict(
    iterations=1500, learning_rate=0.1, depth=5, l2_leaf_reg=10, border_count=64, random_seed=SEED,
    thread_count=N_JOBS, eval_metric="AUC", verbose=0,
)
HGB_PARAMS = dict(  # fallback only: untuned
    learning_rate=0.05, max_iter=600, max_leaf_nodes=8, min_samples_leaf=100, l2_regularization=10.0,
    max_features=0.3, categorical_features="from_dtype", random_state=SEED,
)
# Trees for the final full-data fit = mean CV best iteration x 1.1 (LGB 1129, XGB 1051, CatBoost 583)
FINAL_ITERS = {"lgb": 1240, "xgb": 1155, "cat": 640, "hgb": 600}
