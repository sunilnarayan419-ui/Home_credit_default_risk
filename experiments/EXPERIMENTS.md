# Experiment log

All scores are ROC-AUC from **5-fold stratified CV** (`shuffle=True, random_state=42`); `std` is the standard deviation across folds. Tree models use early stopping on the validation fold to pick the number of trees, which makes every CV score very slightly optimistic by about the same amount. Raw machine-readable log: [`experiment_table.csv`](experiment_table.csv).

## Summary: why the final model is better

| Stage | What changed | CV AUC |
|---|---|---:|
| Baseline | tutorial: 5 features, Random Forest depth 10 | 0.6702 |
| Model | LightGBM on all 32 raw features (native NaN + categorical handling) | 0.7500 |
| Cleaning | sentinel -> NaN (`DAYS_EMPLOYED`, `XNA`) | 0.7499 (no gain alone) |
| Features | + financial ratios (only group that clearly helps, +0.009 on every fold) | 0.7587 |
| Tuning | small heavily regularised trees (`num_leaves=7`, `colsample_bytree=0.3`, L1/L2=10) | 0.7614 |
| Class weights | `scale_pos_weight=3` | 0.7170 (**hurt**, not used) |
| Ensemble | equal-weight mean of LightGBM + XGBoost + CatBoost | **0.7622** |

The ensemble gain (+0.0007 over LightGBM alone) is small but positive on 4 of 5 folds. Optimised blend weights gave no further gain (nested check 0.7622) and adding Logistic Regression hurt (0.7600), so equal weights over the three boosters are used. Probability averaging and rank averaging scored the same (0.76215 vs 0.76216).

Train and test come from the same distribution (adversarial-validation AUC 0.512), so CV is a reasonable proxy for the leaderboard.

## Every experiment

| Experiment | Features | Model | Hyper-parameters | CV AUC | std |
|---|---|---|---|---:|---:|
| E0_tutorial_LR | 5 tutorial feats | LR | tutorial defaults | 0.66177 | 0.00236 |
| E0_tutorial_RF | 5 tutorial feats | RF(depth10) | tutorial defaults | 0.67018 | 0.00221 |
| E1_lgb_raw31 | 32 raw cols | LightGBM | lr.05 leaves31 mcs100 ss.8 cs.6; ES100 | 0.75004 | 0.00357 |
| E2_lgb_clean | 32 raw + DAYS_EMPLOYED_ANOM flag | LightGBM | lr.05 leaves31 mcs100 ss.8 cs.6; ES100 | 0.74994 | 0.00375 |
| E3_lgb_clean_fe | all engineered (ratios,time,ext,bureau,misc) | LightGBM | lr.05 leaves31 mcs100 ss.8 cs.6; ES100 | 0.75825 | 0.00237 |
| E3_add_ext_only | clean + ext group | LightGBM | base params | 0.74947 | 0.00435 |
| E3_add_ratios_only | clean + ratios group | LightGBM | base params | 0.75867 | 0.00306 |
| E3_add_time_only | clean + time group | LightGBM | base params | 0.74969 | 0.00365 |
| E3_add_bureau_only | clean + bureau group | LightGBM | base params | 0.74947 | 0.00410 |
| E3_add_misc_only | clean + misc group | LightGBM | base params | 0.75039 | 0.00311 |
| T00_lgb_rs | full FE | LightGBM | {'num_leaves': 31, 'min_child_samples': 400, 'colsample_bytree': 0.3, 'reg_lambda': 50, 'r | 0.75821 | 0.00377 |
| T01_lgb_rs | full FE | LightGBM | {'num_leaves': 15, 'min_child_samples': 200, 'colsample_bytree': 0.7, 'reg_lambda': 50, 'r | 0.75835 | 0.00308 |
| T02_lgb_rs | full FE | LightGBM | {'num_leaves': 7, 'min_child_samples': 100, 'colsample_bytree': 0.5, 'reg_lambda': 10, 're | 0.75948 | 0.00264 |
| T03_lgb_rs | full FE | LightGBM | {'num_leaves': 15, 'min_child_samples': 100, 'colsample_bytree': 0.3, 'reg_lambda': 100, ' | 0.75880 | 0.00326 |
| T04_lgb_rs | full FE | LightGBM | {'num_leaves': 31, 'min_child_samples': 100, 'colsample_bytree': 0.7, 'reg_lambda': 10, 'r | 0.75700 | 0.00297 |
| T05_lgb_rs | full FE | LightGBM | {'num_leaves': 63, 'min_child_samples': 200, 'colsample_bytree': 0.3, 'reg_lambda': 50, 'r | 0.75888 | 0.00334 |
| T06_lgb_rs | full FE | LightGBM | {'num_leaves': 7, 'min_child_samples': 200, 'colsample_bytree': 0.5, 'reg_lambda': 100, 'r | 0.75979 | 0.00324 |
| T07_lgb_rs | full FE | LightGBM | {'num_leaves': 7, 'min_child_samples': 100, 'colsample_bytree': 0.3, 'reg_lambda': 10, 're | 0.76143 | 0.00294 |
| T08_lgb_rs | full FE | LightGBM | {'num_leaves': 31, 'min_child_samples': 200, 'colsample_bytree': 0.5, 'reg_lambda': 1, 're | 0.75873 | 0.00301 |
| T09_lgb_rs | full FE | LightGBM | {'num_leaves': 15, 'min_child_samples': 100, 'colsample_bytree': 0.5, 'reg_lambda': 1, 're | 0.75956 | 0.00320 |
| T10_lgb_rs | full FE | LightGBM | {'num_leaves': 15, 'min_child_samples': 100, 'colsample_bytree': 0.5, 'reg_lambda': 100, ' | 0.75976 | 0.00271 |
| F1_T07_ratios | ratios only | LightGBM | T07 | 0.76108 | 0.00350 |
| F2_T07_ratios_ext | ratios+ext | LightGBM | T07 | 0.76091 | 0.00338 |
| F3_T07_ratios_ext_time | ratios+ext+time | LightGBM | T07 | 0.76104 | 0.00338 |
| W1_T07_all_spw | all FE | LightGBM | T07 + scale_pos_weight=3 | 0.71695 | 0.00395 |
| R1_leaves5_cs2 | all FE | LightGBM | leaves5 cs.2 | 0.76118 | 0.00327 |
| R2_leaves7_lr03 | all FE | LightGBM | T07 lr.03 | 0.76097 | 0.00341 |
| R3_leaves10_cs3_mcs150 | all FE | LightGBM | leaves10 mcs150 | 0.76119 | 0.00327 |
| lgb | all FE (78) | LightGBM | {'num_leaves': 7, 'min_child_samples': 100, 'colsample_bytree': 0.3, 'reg_lambda': 10, 're | 0.76143 | 0.00294 |
| lr | all FE (78) | LogReg(C=.02) | C=0.02 quantile-normal + OHE | 0.74040 | 0.00415 |
| xgb | all FE (78) | XGBoost | depth3 lr.05 cs.3 mcw20 l2=10 | 0.76049 | 0.00337 |
| cat | all FE (78) | CatBoost | depth5 lr.1 l2=10 | 0.75873 | 0.00370 |
| B1_blend_prob_avg | all FE | LGB+XGB+Cat equal-weight probability average | equal weights | 0.76215 | 0.00316 |
| B2_blend_rank_avg | all FE | LGB+XGB+Cat equal-weight rank average | equal weights | 0.76216 | 0.00320 |
| B3_blend_all4_rank | all FE | LGB+XGB+Cat+LR rank avg | equal weights | 0.75996 | 0.00343 |

Notes: experiments `T00`-`T10` are the random search (11 of 14 planned trials finished). `E3_add_*_only` add a single feature group to the cleaned raw features. `F*` repeat feature-set comparisons under the tuned LightGBM parameters. `lgb`/`xgb`/`cat`/`lr` are the final-model CV runs whose out-of-fold predictions fed the blend experiments `B1`-`B3`.
