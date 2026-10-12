# Limitations and responsible use

Credit scoring is a sensitive domain. This section lists what the project does **not** establish.

## Model and data limits

* **Single dataset.** Results are cross-validated on one in-class competition dataset. External validity,
  distribution shift over time or across markets, and reject inference are not examined.
* **Ranking, not probabilities.** Performance is measured by ROC-AUC. Calibration, PR-AUC and thresholded
  precision/recall are not evaluated.
* **Slightly optimistic CV** from early stopping on the validation fold.
* **External scores.** `EXT_SOURCE_*` are opaque third-party scores whose provenance is outside the project;
  operational use would require understanding and monitoring them.
* **Leakage and class imbalance.** Leakage was controlled by design (see
  [Data and features](data-and-features.md#leakage-controls)) but not independently audited; about 8 % positives
  means thresholds need explicit cost analysis.

## Fairness, explainability and regulation

The inputs include demographic and socio-economic attributes (for example `CODE_GENDER`, family status, age
through `DAYS_BIRTH`). The project has **not** measured subgroup performance, disparate impact or explanations.
Lending decisions are subject to law and regulation (anti-discrimination, adverse-action explanations, data
protection) that vary by jurisdiction; nothing here claims compliance. Any operational use needs fairness testing,
explainability work, legal review and independent model validation.

## Security and privacy

* Never commit competition data, customer records, credentials or `.env` files; `data/*`, `output/*` and
  environment files are git-ignored.
* The model artifact is a pickle file: load only files you created.
* The API logs no payloads and returns no submitted values, paths or stack traces, but has no authentication or
  rate limiting.

## Reproducibility

Seed 42 and single-threaded boosting give repeatable results for fixed library versions. Artifacts record the
library versions used (`/model-info`); load them with matching versions.
