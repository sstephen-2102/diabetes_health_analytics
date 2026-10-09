# Member 3: Modeling handoff

This change implements the modeling contracts (spec section 25, Member 3):
- logistic regression with the BMI × Age interaction;
- three classifiers with duplicate-leakage control;
- class-imbalance experiments and threshold analysis;
- calibration and interpretation;
- a duplicate-handling sensitivity experiment;
- the Prediction Explorer and the six modeling pages of the Streamlit app.

Results and report guidance are in `docs/MODELING_REPORT.md`. This file covers
the code: what each function returns, how to run it, the design decisions and
what was tested.

## Implemented contracts

### Splitting and leakage (`src/modeling/classification.py`, `validation.py`)

| Function | Return and behavior |
|---|---|
| `split_data(data, target_column, test_size=0.20, random_state=42, stratify=True)` | Dict with `X_train`, `X_test`, `y_train`, `y_test` and `metadata`. Rows are grouped by a hash of the feature vector, so identical profiles stay on one side (`StratifiedGroupKFold` when stratified). Raises `DataValidationError` if any profile crosses the split. |
| `check_duplicate_leakage(X_train, X_test)` | Dict with `leakage_detected`, `overlapping_unique_rows`, row counts and `status` ("PASS"/"FAIL"). |

### Classifiers (`src/modeling/classification.py`)

| Function | Return and behavior |
|---|---|
| `train_logistic_classifier(X_train, y_train, class_weight=None, random_state=42)` | Fitted `Pipeline` (StandardScaler, LogisticRegression with max_iter 1000). `class_weight` is None, "balanced" or a dict. |
| `train_random_forest(X_train, y_train, class_weight=None, random_state=42, **model_parameters)` | Fitted `Pipeline` with RandomForestClassifier. Also accepts "balanced_subsample"; extra keyword arguments go to the classifier. |
| `train_gradient_boosting(X_train, y_train, random_state=42, **model_parameters)` | Fitted `Pipeline` with GradientBoostingClassifier (no class-weight option in scikit-learn). |
| `generate_predictions(model, X, threshold=0.50)` | Dict with `probabilities`, `predictions` (probability ≥ threshold) and `threshold`. |
| `evaluate_classifier(y_true, y_pred, y_probability)` | Dict with accuracy, precision, recall, specificity, F1, ROC AUC, PR AUC, Brier score, `confusion_matrix` counts and `n`. |
| `compare_models(evaluation_results)` | DataFrame indexed by model, one column per metric and count. No winner is picked. |

### Imbalance, thresholds, calibration, interpretation

| Function | Return and behavior |
|---|---|
| `inspect_class_distribution(y)` | Dict with counts, percentages, total, imbalance ratio, majority and minority class. |
| `balance_training_data(X_train, y_train, method="undersample", random_state=42)` | Dict with balanced `X_train`/`y_train`, method and before/after distributions. Training data only; "oversample" is also supported. |
| `train_class_weighted_model(model_type, X_train, y_train, random_state=42)` | Fitted pipeline with `class_weight="balanced"` for "logistic_regression" or "random_forest". |
| `evaluate_thresholds(y_true, probabilities, thresholds)` | DataFrame, one row per threshold: precision, recall, specificity, F1 and confusion counts. |
| `generate_threshold_curve(threshold_results)` | Plotting table from `evaluate_thresholds`: threshold, precision, recall, specificity and F1, sorted by threshold. |
| `calculate_calibration_metrics(y_true, probabilities, n_bins=10)` | Dict with Brier score, expected calibration error, `calibration_table`, `n_bins` and `n`. |
| `generate_calibration_data(y_true, probabilities, n_bins=10)` | Equal-width bins over [0, 1]: mean predicted, observed frequency and count; empty bins omitted. |
| `calculate_feature_importance(model, X, y, method="permutation", scoring="roc_auc", n_repeats=10, random_state=42)` | DataFrame of feature, importance mean and standard deviation, largest first; settings stored in `attrs`. |

### Regression (`src/modeling/regression.py`)

| Function | Return and behavior |
|---|---|
| `fit_logistic_regression(data, target_column, feature_columns, interaction_terms=None)` | statsmodels Logit. Dict with a `results` table (coefficient, SE, p, OR, 95% CI), n, log-likelihood, AIC, BIC, pseudo-R², `converged`, covariance and the design (features, interaction terms, dataset variant). |
| `compare_regression_models(baseline_model, interaction_model)` | Dict with both models' fit statistics, ΔAIC, ΔBIC and the likelihood-ratio test (statistic, df, p). No automatic ranking. |
| `extract_odds_ratios(fitted_model)` | DataFrame of feature, odds ratio, CI and p-value for every model term, sorted by odds ratio (largest first). |

### Prediction Explorer (`src/analysis/profiles.py`)

| Function | Return and behavior |
|---|---|
| `build_risk_profile(feature_values, metadata, defaults=None)` | Dict with model-ready `values`, the `defaulted_features` list and a labeled `display` table. Every `prediction_input` feature must be supplied; the rest come from `defaults` (training mode or median). |
| `predict_risk_profile(model, risk_profile, threshold=0.50)` | Dict with probability, classification, spec wording for both, and the "not a medical diagnosis" disclaimer. |

### Exports (`src/analysis/`)

Each export validates the CSV, runs the dataset reference check by default, and
writes tables, figures and a manifest recording the dataset SHA-256.

| Function | Writes |
|---|---|
| `export_regression(csv_path, output_directory="outputs", check_reference=True)` | Baseline and interaction coefficients, odds ratios, model comparison, BMI odds by age, two figures, `regression_manifest.json`. |
| `export_modeling(csv_path, output_directory="outputs", check_reference=True, importance_sample=10_000)` | Model comparison, imbalance experiments, and per model: thresholds, calibration, importance, ROC and PR curves. Also five figures, `models/*.pkl`, `models/metadata.json`, `models/metrics.json` and `modeling_manifest.json`. |
| `export_duplicate_sensitivity(csv_path, output_directory="outputs", check_reference=True)` | `tables/duplicate_sensitivity.csv` and its manifest (two dataset variants × two split types, all three models). |

### App (`app/`)

- **Service loaders** (`services.py`):
  - `load_model_artifacts`, `load_modeling_outputs`, `load_regression_outputs` and `load_duplicate_sensitivity`;
  - each returns `{status, results, warnings, dataset_variant, target_definition, metadata}`;
  - when an export is missing, each raises `DataLoadError` with the exact command to run.
- **Other services:** `get_threshold_comparison` returns one row per model at a grid threshold; `get_prediction_result` backs the Prediction Explorer.
- **Caching:** `artifacts.py` caches models with `st.cache_resource` and tables with `st.cache_data`.
- **Pages:** `machine_learning`, `imbalance_threshold`, `calibration`, `interpretation`, `regression_lab` and `prediction`, registered in the `PAGES` dict in `app/app.py`.
- **Components** (`components/`): Plotly charts, metric tables and context captions.

## Usage

```python
from src.common.config import FEATURE_COLUMNS
from src.data.loader import load_dataset
from src.data.preprocessing import prepare_analysis_data
from src.modeling.classification import (evaluate_classifier, generate_predictions,
                                         split_data, train_logistic_classifier)

raw = load_dataset("data/raw/cdc_diabetes.csv")
data = prepare_analysis_data(raw, "Diabetes_binary", FEATURE_COLUMNS,
                             dataset_variant="unique_profile_v1")
split = split_data(data, "Diabetes_binary")
model = train_logistic_classifier(split["X_train"], split["y_train"])
predicted = generate_predictions(model, split["X_test"], threshold=0.5)
metrics = evaluate_classifier(split["y_test"], predicted["predictions"], predicted["probabilities"])
```

## Reproduce

```text
python -m pip install -r requirements.txt
python -m src.analysis.export_regression --csv data/raw/cdc_diabetes.csv
python -m src.analysis.export_modeling --csv data/raw/cdc_diabetes.csv
python -m src.analysis.export_duplicate_sensitivity --csv data/raw/cdc_diabetes.csv
python -m pytest tests/ -q
streamlit run streamlit_app.py
```

- **Run from the repo root.** Running `app/app.py` directly breaks the page imports.
- **Rerun `export_modeling` after model code changes.** The app reads the saved
  `.pkl` files, so it shows the old models until then.
- **Restart Streamlit after editing a page.** A running server does not always
  reload page modules.

## Design decisions

1. **Dataset variants.**
   - The regression uses `full_clean_v1`, consistent with the descriptive statistics.
   - ML uses `unique_profile_v1` (spec section 7).
   - The duplicate-sensitivity experiment compares both.
2. **Grouped split.**
   - Complete-row deduplication still leaves 1,566 feature vectors with both labels.
   - Grouping by feature vector keeps each one on a single side of the split.
3. **No threshold is selected.** There is no validation set, so the full grid
   (0.10–0.90) is reported on the test set as a trade-off.
4. **Imbalance changes training data only.** The test set keeps its natural 15.3%
   prevalence. Class weighting and undersampling are compared on logistic
   regression only.
5. **Centered interaction.** BMI and Age are mean-centered before forming BMI × Age,
   so the main effects describe an average respondent.
6. **Prediction Explorer defaults.**
   - Only the 8 `prediction_input` features are asked for.
   - The other 13 use the training mode or median, stored in `models/metadata.json` as `default_profile`.
   - The page lists which inputs were defaulted.
7. **Scaling inside pipelines.** `StandardScaler` is fitted on training data only.
   Saved `.pkl` files contain the whole pipeline.
8. **Seed.** Seed 42 everywhere (`PROJECT_CONFIG["random_state"]`).

## Validation performed

- **Full suite:** 438 passed.
  - About 380 tests are in the modeling, export, profile, service and app test files.
  - Those include `test_models`, `test_modeling_split`, `test_regression`,
    `test_export_*`, `test_profiles` and `test_app_modeling`.
- **What the tests cover:**
  - input validation and error messages;
  - leakage control;
  - an untouched test set in the imbalance experiments;
  - threshold-grid coverage, and calibration counts summing to the test size;
  - reproducibility of every export, and saved models reloading and predicting;
  - service contracts, including the error for missing exports;
  - page rendering, and the absence of diagnostic wording on every page;
  - Prediction Explorer inputs, threshold behavior and the disclaimer.
- **Real data:**
  - leakage check PASS with 0 shared profiles;
  - both regression models converged;
  - the primary duplicate-sensitivity scenario reproduces `model_comparison.csv` exactly.
- **CI:** GitHub Actions on Python 3.12 passes; development used Python 3.14.
- **Visual check:** all six modeling pages were checked in a browser on the real outputs.

## Scope boundaries and notes for the team

- **Member 2:**
  - the Probability Explorer and Statistical Testing Lab pages are listed in `app/app.py`, so implementing their `render()` makes them appear;
  - the statistics references in `docs/references.md` are marked "planned". Keep only the methods you implement.
- **Member 1:**
  - `load_dataset` silently shifts columns when every row has one extra field, because pandas reads the first column as an index. Rows with mixed field counts are caught.
  - Education code 1 is "No schooling / kindergarten only" in `src/common/config.py` but "Never attended school / kindergarten only" in spec section 6.6.
- **Not done:** hyperparameter tuning, cross-validation, a validation set, confidence intervals on ML metrics, survey weights and external validation. All are listed as limitations in `docs/MODELING_REPORT.md`.
- **Excluded from Git:** raw data, `.venv/`, caches and the model files `outputs/models/*.pkl` (the random forest alone is about 86 MB). The exported tables, figures, `metadata.json` and `metrics.json` are committed, so every modeling page except the Prediction Explorer works from a fresh clone. Run `export_modeling` to recreate the `.pkl` files.

## Review

- Branch: `feature/member3-modeling`.
- Pull request: [#2](https://github.com/sstephen-2102/diabetes_health_analytics/pull/2).

AI assistance was used for planning, code review, tests and documentation, and to
write parts of the code. Every change was reviewed before committing; the author
should understand the contribution before presenting it as project work.
