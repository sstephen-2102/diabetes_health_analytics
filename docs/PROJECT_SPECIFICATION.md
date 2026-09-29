# Diabetes Health Analytics & Prediction Lab
## Complete Project Specification

**Project:** Understanding and Predicting Diabetes Risk: A Statistical and Machine Learning Study of Health, Lifestyle, and Socioeconomic Factors  
**Application:** Diabetes Health Analytics & Prediction Lab  
**Team:** 3 members  
**Duration:** 20 days  
**Framework:** Python + Streamlit  
**Academic context:** University of San Diego — M.S. in Applied Artificial Intelligence  
**Specification version:** 1.1  
**Dataset verification status:** Complete — verified against the supplied CSV on 2026-09-29

---

## 1. Purpose

This document is the authoritative specification for the project. It defines the research objectives, questions, hypotheses, dataset requirements, statistical and machine-learning methodology, software architecture, UI requirements, function contracts, validation, testing, team ownership, report requirements, limitations, and definition of done.

The companion document `IMPLEMENTATION_PLAN.md` defines the 20-day execution schedule.

If implementation decisions conflict with this document, update the specification before silently changing the design.

---

## 2. Executive Summary

The project investigates relationships between demographic, health, lifestyle, and socioeconomic indicators and the dataset outcome `Diabetes_binary`, then evaluates whether those variables can predict that outcome using interpretable statistical and machine-learning methods.

The project combines:

- exploratory data analysis;
- descriptive statistics;
- marginal, joint, conditional, and posterior probability;
- Bayes theorem;
- confidence intervals;
- hypothesis testing;
- effect sizes;
- association analysis;
- logistic regression;
- interaction analysis;
- Logistic Regression, Random Forest, and Gradient Boosting;
- class-imbalance experiments;
- threshold analysis;
- calibration;
- model interpretation;
- an interactive Streamlit research laboratory.

This is an educational/research project. It is **not a clinical diagnostic system**, does not provide medical advice, and does not establish causal relationships.

### Critical terminology rule

The target `Diabetes_binary` must **not** be described throughout the project simply as “confirmed diabetes.” In this BRFSS-derived dataset, the positive class represents **prediabetes or diabetes**, while the negative class represents neither condition.

Therefore:

- use **“`Diabetes_binary` outcome”**, **“positive target class”**, or **“prediabetes-or-diabetes outcome”** when precision is required;
- use “diabetes-related outcome” when a shorter phrase is necessary;
- do not write “the model diagnoses diabetes”;
- do not interpret a positive prediction as confirmation of a clinical diagnosis.

---

## 3. Research Objective

### Primary objective

Investigate relationships between demographic, health, behavioral, and socioeconomic indicators and the `Diabetes_binary` outcome, and evaluate whether those indicators can predict that outcome using interpretable statistical and machine-learning methods.

### Secondary objectives

1. Understand dataset structure and quality.
2. Quantify the distribution of the target outcome in the analytical sample.
3. Demonstrate marginal, joint, conditional, and posterior probabilities.
4. Quantify uncertainty with confidence intervals.
5. Test associations and group differences.
6. Report practical effect sizes alongside p-values.
7. Estimate adjusted associations using logistic regression.
8. Investigate the BMI × Age interaction.
9. Compare three classification models.
10. Investigate class imbalance.
11. Examine threshold trade-offs.
12. Assess probability calibration.
13. Provide transparent model interpretation.
14. Build a reusable Streamlit research laboratory.
15. Demonstrate how data-quality decisions, especially duplicate handling, can affect predictive evaluation.

---

## 4. Research Questions

- **RQ1:** What demographic, health, lifestyle, and socioeconomic characteristics describe the analytical sample?
- **RQ2:** How is `Diabetes_binary` distributed?
- **RQ3:** How does the probability of the positive `Diabetes_binary` class change under selected conditions such as high blood pressure, high cholesterol, physical inactivity, and demographic categories?
- **RQ4:** Can Bayes theorem be demonstrated and independently verified from the observed data?
- **RQ5:** What confidence intervals describe selected proportions, means, and group differences?
- **RQ6:** Which variables show statistically detectable associations or differences with the target outcome?
- **RQ7:** How large are the observed effects, and how should they be interpreted alongside p-values?
- **RQ8:** Which variables remain associated with the target after simultaneous adjustment?
- **RQ9:** Does the BMI–target relationship vary by age?
- **RQ10:** How do Logistic Regression, Random Forest, and Gradient Boosting compare across multiple metrics?
- **RQ11:** How does class imbalance affect model behavior?
- **RQ12:** How do thresholds affect precision, recall, specificity, and F1, and are predicted probabilities reasonably calibrated?
- **RQ13:** How does exact-duplicate handling affect train/test integrity and model evaluation?

---

## 5. Hypotheses

| ID | Alternative/research hypothesis | Null hypothesis |
|---|---|---|
| H1 | The `Diabetes_binary` outcome is associated with HighBP. | The `Diabetes_binary` outcome and HighBP are independent. |
| H2 | The `Diabetes_binary` outcome is associated with HighChol. | The `Diabetes_binary` outcome and HighChol are independent. |
| H3 | Mean BMI differs between target groups. | Mean BMI is equal between target groups. |
| H4 | The `Diabetes_binary` outcome is associated with physical activity. | The `Diabetes_binary` outcome and physical activity are independent. |
| H5 | The `Diabetes_binary` outcome is associated with smoking. | The `Diabetes_binary` outcome and smoking are independent. |
| H6 | The target outcome differs across age categories. | The target outcome is independent of age category. |
| H7 | The target outcome differs across income categories. | The target outcome is independent of income category. |
| H8 | Selected predictors remain associated with the target after adjustment. | Selected predictors have no adjusted association. |
| H9 | The BMI–target association varies with age. | There is no BMI × Age interaction. |
| H10 | Class-imbalance strategy changes predictive behavior. | Relevant predictive behavior is unchanged. |
| H11 | Exact-duplicate handling changes some model evaluation results. | Exact-duplicate handling does not materially change model evaluation results. |

H10 and H11 are primarily experimental/modeling questions rather than conventional population hypothesis tests.

---

## 6. Dataset Specification

### 6.1 Source

Primary dataset: Kaggle **Diabetes Health Indicators Dataset** associated with BRFSS 2015.

Kaggle identifier:

`alexteboul/diabetes-health-indicators-dataset`

Primary file:

`diabetes_binary_health_indicators_BRFSS2015.csv`

The supplied CSV has now been inspected and the verification gate is complete.

### 6.2 Verified dataset snapshot

The supplied CSV contains:

| Property | Verified value |
|---|---:|
| Rows | 253,680 |
| Columns | 22 |
| Missing cells | 0 |
| Exact duplicate rows | 24,206 |
| Exact duplicate rate | 9.5419% |
| Rows after `drop_duplicates()` | 229,474 |
| Target value 0 | 218,334 (86.07%) |
| Target value 1 | 35,346 (13.93%) |

The exact duplicate count is based on complete-row equality across all 22 columns.

### 6.3 Verified columns

```text
Diabetes_binary
HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income
```

All 22 supplied columns are numeric in the CSV.

### 6.4 Target definition

`Diabetes_binary` is binary:

- `0` = neither prediabetes nor diabetes;
- `1` = prediabetes or diabetes.

This distinction is mandatory in the report, UI, README, model documentation, and presentation.

### 6.5 Variable dictionary

| Variable | Meaning / treatment |
|---|---|
| `Diabetes_binary` | Binary target: 0 = neither prediabetes nor diabetes; 1 = prediabetes or diabetes |
| `HighBP` | Binary high-blood-pressure indicator |
| `HighChol` | Binary high-cholesterol indicator |
| `CholCheck` | Binary indicator of cholesterol check within the relevant survey period |
| `BMI` | Numeric body-mass-index value; observed range 12–98 |
| `Smoker` | Binary smoking indicator using the source survey definition |
| `Stroke` | Binary stroke-history indicator |
| `HeartDiseaseorAttack` | Binary heart-disease/heart-attack indicator |
| `PhysActivity` | Binary physical-activity indicator |
| `Fruits` | Binary fruit-consumption indicator |
| `Veggies` | Binary vegetable-consumption indicator |
| `HvyAlcoholConsump` | Binary heavy-alcohol-consumption indicator |
| `AnyHealthcare` | Binary healthcare-coverage/access indicator |
| `NoDocbcCost` | Binary indicator of inability to see a doctor because of cost |
| `GenHlth` | Ordinal general-health rating, 1–5 |
| `MentHlth` | Count of poor mental-health days in the previous 30 days, 0–30 |
| `PhysHlth` | Count of poor physical-health days in the previous 30 days, 0–30 |
| `DiffWalk` | Binary difficulty-walking indicator |
| `Sex` | Binary sex coding: 0 = female, 1 = male |
| `Age` | Ordinal age category, 1–13 |
| `Education` | Ordinal education category, 1–6 |
| `Income` | Ordinal income category, 1–8 |

### 6.6 Verified ordinal mappings

**Age**

| Code | Category |
|---:|---|
| 1 | 18–24 |
| 2 | 25–29 |
| 3 | 30–34 |
| 4 | 35–39 |
| 5 | 40–44 |
| 6 | 45–49 |
| 7 | 50–54 |
| 8 | 55–59 |
| 9 | 60–64 |
| 10 | 65–69 |
| 11 | 70–74 |
| 12 | 75–79 |
| 13 | 80+ |

**General health (`GenHlth`)**

| Code | Category |
|---:|---|
| 1 | Excellent |
| 2 | Very good |
| 3 | Good |
| 4 | Fair |
| 5 | Poor |

**Education**

| Code | Category |
|---:|---|
| 1 | Never attended school / kindergarten only |
| 2 | Grades 1–8 |
| 3 | Grades 9–11 |
| 4 | Grade 12 / GED |
| 5 | Some college / technical school |
| 6 | College graduate / 4+ years |

**Income**

| Code | Category |
|---:|---|
| 1 | Less than $10,000 |
| 2 | $10,000–$14,999 |
| 3 | $15,000–$19,999 |
| 4 | $20,000–$24,999 |
| 5 | $25,000–$34,999 |
| 6 | $35,000–$49,999 |
| 7 | $50,000–$74,999 |
| 8 | $75,000 or more |

The implementation must retain both machine-readable codes and human-readable labels.

---

## 7. Dataset Versioning and Duplicate Policy

The project must never silently remove duplicates.

Create explicit dataset variants:

```text
raw_v1
    ↓
full_clean_v1
    ↓
unique_profile_v1
    ↓
balanced_train_v1
```

### `raw_v1`

The original downloaded CSV, preserved unchanged.

### `full_clean_v1`

A verified analytical representation of the complete supplied CSV:

- no missing cells;
- no row deletion;
- no silent recoding;
- validated columns and values;
- exact duplicate count retained as metadata.

### `unique_profile_v1`

`full_clean_v1` after complete-row duplicate removal.

This variant is the **primary ML source** because no respondent identifier is available and identical rows can otherwise cross the train/test boundary, making independent test evaluation difficult to interpret.

### `balanced_train_v1`

A training-only balanced derivative created from the training partition of `unique_profile_v1`.

### Primary analytical rules

1. **Descriptive and inferential analysis:** use `full_clean_v1` as the primary sample so the supplied dataset is represented faithfully.
2. **Sensitivity analysis:** where duplicate handling could materially affect a statistical result, compare with `unique_profile_v1`.
3. **Primary ML:** use `unique_profile_v1`.
4. **Train/test split:** perform a stratified 80/20 split after duplicate handling.
5. **Test set:** never balance, resample, fit, or tune against the test set.
6. **Sensitivity ML:** optionally evaluate `full_clean_v1` to quantify the effect of duplicate rows.
7. Every major result must identify its dataset variant.

### Why duplicates are not automatically called errors

The dataset does not provide a respondent identifier in the supplied CSV. Therefore, identical rows cannot be proven to represent erroneous duplicate records. The project should describe them as **exact duplicate feature rows**, document their prevalence, and treat their handling as an analytical design decision.

---

## 8. Download Strategy

Use `kagglehub` for reproducible acquisition.

```text
Kaggle → kagglehub → data/raw/ → validation → explicit dataset variants → analysis/modeling
```

The Streamlit application must not repeatedly download from Kaggle during ordinary navigation.

Dataset ID and expected filename belong in configuration, not hard-coded throughout the application.

---

## 9. Methodological Principles

### Association is not causation

Use “associated with,” “observed relationship,” or “predictive signal.” Do not use causal language such as “causes,” “prevents,” or “leads to.”

### Prediction is not diagnosis

The prediction page may show a model-estimated probability and classification at a selected threshold. It must never state that a person has diabetes or provide medical advice.

### Statistical significance is not practical importance

Do not label a variable important solely because `p < .05`. Present estimate, confidence interval, p-value, effect size, sample size, and context.

### Class imbalance

Class imbalance is a research question. The original test distribution must be retained for fair evaluation.

### Test-set integrity

The test set must not be balanced, oversampled, undersampled, or used for training. Any preprocessing that learns parameters must be fit using training data only.

### Duplicate leakage control

Exact duplicate feature rows must not be allowed to cross train/test partitions in the primary ML experiment. Duplicate removal or grouped splitting must happen before final test evaluation. The chosen strategy must be recorded in experiment metadata.

### BRFSS survey limitation

BRFSS is a complex survey and official population inference uses survey design and weights. This project uses a cleaned Kaggle analytical dataset and does **not** reproduce the full complex-survey analysis. Unweighted sample percentages must not be presented as official U.S. population prevalence estimates.

---

## 10. Statistical Analysis Plan

### Descriptive statistics

Continuous variables:

- n;
- mean;
- median;
- SD;
- variance;
- minimum;
- Q1;
- Q3;
- IQR;
- maximum.

Categorical/ordinal variables:

- category;
- count;
- percentage.

Report dataset variant and denominator.

### Probability

Implement:

- marginal probability;
- joint probability;
- conditional probability;
- Bayes theorem;
- profile-based probability.

Every result should expose numerator, denominator, conditions, target definition, dataset variant, and sample size.

### Confidence intervals

Implement:

- proportion CI;
- mean CI;
- difference-in-means CI.

Document the interval method and confidence level.

### Chi-square

Report:

- contingency table;
- chi-square statistic;
- df;
- p-value;
- Cramér’s V;
- n;
- expected-count diagnostics where appropriate.

### Welch t-test

Report:

- group n;
- means;
- SDs;
- mean difference;
- t;
- df;
- p-value;
- CI;
- effect size.

Welch's test is preferred over assuming equal variances.

### ANOVA

Report:

- group statistics;
- F;
- df;
- p-value;
- effect size;
- post-hoc comparisons where justified.

Where assumptions are materially questionable, document the limitation and consider an appropriate sensitivity analysis.

### Multiple comparisons

When a family of tests is performed, use a documented multiplicity strategy such as Benjamini-Hochberg/FDR.

Report both raw and adjusted p-values.

### Correlation

Support a documented correlation method such as Spearman for appropriate exploratory relationships.

Correlation must not be presented as causation.

---

## 11. Regression Plan

Primary binary outcome: `Diabetes_binary`.

Candidate predictors include:

`BMI`, `Age`, `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Sex`, `Education`, and `Income`.

Final feature selection must be documented.

### Variable treatment

- `BMI`: continuous.
- `MentHlth`, `PhysHlth`: bounded count-like variables; treatment must be documented.
- `Age`, `Education`, `Income`, `GenHlth`: ordinal variables. The implementation must explicitly document whether each is modeled numerically as an ordinal score or encoded categorically.
- Binary indicators: retain their verified 0/1 coding.
- Interaction: `BMI × Age`.

### Logistic regression outputs

- coefficients;
- standard errors;
- p-values;
- odds ratios;
- 95% CIs;
- sample size;
- model statistics;
- feature-treatment metadata;
- dataset variant.

### Interaction

At minimum evaluate BMI × Age.

Compare baseline and interaction models and report:

- interaction coefficient;
- odds ratio where appropriate;
- confidence interval;
- p-value;
- model-comparison statistics;
- interpretation limitations.

---

## 12. Machine-Learning Plan

### Models

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

### Primary split

Default:

- dataset: `unique_profile_v1`;
- 80/20 stratified train/test split;
- random seed: 42;
- test set remains untouched until final evaluation.

### Leakage checks

Before final evaluation, verify:

- no exact duplicate feature rows occur across train and test;
- target is not present in feature matrices;
- preprocessing transformers are fit only on training data;
- class balancing is applied only to training data;
- threshold selection is not silently optimized on the final test set.

### Metrics

- accuracy;
- precision;
- recall/sensitivity;
- specificity;
- F1;
- ROC-AUC;
- PR-AUC;
- confusion matrix;
- Brier score.

Because the target is imbalanced, accuracy must not be treated as the only performance measure.

### Comparison

Show metrics side by side.

The comparison function must not embed a winner/ranking. Interpretation belongs in the research discussion.

---

## 13. Class-Imbalance Experiments

Compare:

1. original training distribution;
2. class-weighted model;
3. balanced training data;
4. threshold adjustment.

Initial balancing method: undersampling, if retained after testing.

Only training data may be balanced.

The final test set must preserve the natural class distribution of the primary evaluation dataset.

Every imbalance experiment must record:

- source dataset variant;
- training counts before balancing;
- training counts after balancing;
- model;
- seed;
- threshold;
- test metrics.

---

## 14. Threshold Analysis

Evaluate configurable thresholds, for example 0.10 through 0.90 in 0.05 increments.

For each threshold report:

- threshold;
- precision;
- recall;
- specificity;
- F1;
- confusion-matrix counts.

The UI should expose trade-offs rather than silently selecting a “best” threshold.

If a threshold is selected for an illustrative scenario, document the selection rule and whether a separate validation set was used.

---

## 15. Calibration

Evaluate whether predicted probabilities correspond reasonably to observed event frequencies.

Include:

- Brier score;
- calibration table;
- calibration curve;
- number of bins;
- predicted probability per bin;
- observed frequency per bin;
- observations per bin.

Calibration is distinct from discrimination.

Calibration evaluation must use predictions generated without fitting the calibrator on the final test observations.

---

## 16. Model Interpretation

Provide:

- logistic-regression odds ratios;
- confidence intervals;
- permutation feature importance;
- optional native tree importance as supplementary information.

Interpretation remains descriptive and non-causal.

Permutation importance should be calculated on an explicitly documented evaluation split.

---

## 17. Streamlit Application Specification

Application: **Diabetes Health Analytics & Prediction Lab**.

The interface is an interactive research laboratory rather than only a dashboard.

### Pages

1. **Home / Executive Dashboard** — dataset overview, sample size, target distribution, selected findings, research questions, methodology warnings.
2. **Data Explorer** — schema, data types, missingness, duplicate rate, categories, quality, descriptive statistics.
3. **Exploratory Analysis** — distributions, target-group comparisons, prevalence, correlations.
4. **Probability Explorer** — marginal, joint, conditional, Bayes, profile probability.
5. **Statistical Testing Lab** — chi-square, Welch t-test, ANOVA, post-hoc, effect size, multiple comparisons.
6. **Regression Lab** — baseline logistic regression, interaction model, odds ratios, CIs, model comparison.
7. **Machine Learning Lab** — models, training, metrics, confusion matrix, ROC, PR, comparison.
8. **Imbalance & Threshold Lab** — class distribution, weighting, balancing, threshold curves.
9. **Calibration** — calibration curve/table and Brier score.
10. **Interpretation** — feature importance and odds ratios.
11. **Prediction Explorer** — feature profile, selected model, predicted probability, threshold, classification, model metadata.

### Prediction Explorer wording

Use labels such as:

- “Model-estimated probability of the positive `Diabetes_binary` class”
- “Model classification at selected threshold”
- “This output is a statistical/ML estimate from the project dataset and is not a medical diagnosis.”

Never use:

- “You have diabetes”
- “You are diagnosed”
- “The AI diagnosed you”

---

## 18. Software Architecture

```text
Streamlit UI
     ↓
Application / Service Layer
     ↓
Analytics Functions
     ↓
Data Layer
```

Streamlit pages must not implement statistical calculations directly.

### Repository

```text
diabetes-health-analytics/
├── app/
│   ├── app.py
│   ├── pages/
│   └── components/
├── src/
│   ├── common/
│   ├── data/
│   ├── statistics/
│   ├── modeling/
│   └── analysis/
├── tests/
├── data/raw/
├── data/processed/
├── outputs/figures/
├── outputs/tables/
├── outputs/models/
├── docs/
├── requirements.txt
└── README.md
```

Recommended explicit data artifacts:

```text
data/
├── raw/
├── processed/
│   ├── full_clean_v1/
│   ├── unique_profile_v1/
│   └── balanced_train_v1/
```

---

## 19. Function Contracts

### Data

```python
download_kaggle_dataset(dataset_id: str, output_directory: str) -> str
load_dataset(file_path: str) -> pd.DataFrame
validate_dataset(data, required_columns, target_column) -> dict
generate_data_quality_report(data) -> dict
prepare_analysis_data(
    data,
    target_column,
    feature_columns,
    drop_missing=False,
    dataset_variant="full_clean_v1",
) -> pd.DataFrame
```

### Descriptive

```python
summarize_numeric_variable(data, column) -> dict
summarize_categorical_variable(data, column, include_percentages=True) -> pd.DataFrame
compare_by_target(data, feature, target_column) -> pd.DataFrame
calculate_prevalence_table(data, feature, target_column) -> pd.DataFrame
```

### Probability

```python
calculate_probability(data, column, value) -> float
calculate_conditional_probability(
    data,
    target_column,
    target_value,
    condition_columns,
    condition_values,
) -> dict
calculate_joint_probability(data, conditions: dict) -> dict
verify_bayes_theorem(
    data,
    target_column,
    target_value,
    condition_column,
    condition_value,
) -> dict
calculate_risk_profile_probability(
    data,
    target_column,
    target_value,
    profile: dict,
) -> dict
```

### Confidence intervals

```python
proportion_confidence_interval(
    successes,
    total,
    confidence_level=0.95,
) -> dict

mean_confidence_interval(
    values,
    confidence_level=0.95,
) -> dict

difference_in_means_ci(
    group_a,
    group_b,
    confidence_level=0.95,
) -> dict
```

### Hypothesis testing

```python
run_chi_square_test(data, feature, target_column) -> dict
run_welch_t_test(
    data,
    numeric_feature,
    target_column,
    group_a,
    group_b,
) -> dict
run_anova(data, numeric_feature, grouping_feature) -> dict
run_posthoc_analysis(
    data,
    numeric_feature,
    grouping_feature,
    method="tukey",
) -> pd.DataFrame
adjust_multiple_comparisons(
    p_values,
    method="fdr_bh",
) -> pd.DataFrame
```

### Association

```python
calculate_correlation_matrix(
    data,
    columns,
    method="spearman",
) -> pd.DataFrame

calculate_effect_size(
    analysis_type,
    data,
    **kwargs,
) -> dict
```

### Regression

```python
fit_logistic_regression(
    data,
    target_column,
    feature_columns,
    interaction_terms=None,
) -> dict

compare_regression_models(
    baseline_model,
    interaction_model,
) -> dict

extract_odds_ratios(
    fitted_model,
) -> pd.DataFrame
```

### ML

```python
split_data(
    data,
    target_column,
    test_size=0.20,
    random_state=42,
    stratify=True,
) -> dict

train_logistic_classifier(
    X_train,
    y_train,
    class_weight=None,
    random_state=42,
) -> object

train_random_forest(
    X_train,
    y_train,
    class_weight=None,
    random_state=42,
    **model_parameters,
) -> object

train_gradient_boosting(
    X_train,
    y_train,
    random_state=42,
    **model_parameters,
) -> object

generate_predictions(
    model,
    X,
    threshold=0.50,
) -> dict

evaluate_classifier(
    y_true,
    y_pred,
    y_probability,
) -> dict

compare_models(
    evaluation_results: dict,
) -> pd.DataFrame
```

### Imbalance, threshold, calibration, interpretation

```python
inspect_class_distribution(y) -> dict

balance_training_data(
    X_train,
    y_train,
    method="undersample",
    random_state=42,
) -> dict

train_class_weighted_model(
    model_type,
    X_train,
    y_train,
    random_state=42,
) -> object

evaluate_thresholds(
    y_true,
    probabilities,
    thresholds: list[float],
) -> pd.DataFrame

generate_threshold_curve(
    threshold_results,
)

calculate_calibration_metrics(
    y_true,
    probabilities,
    n_bins=10,
) -> dict

generate_calibration_data(
    y_true,
    probabilities,
    n_bins=10,
) -> pd.DataFrame

calculate_feature_importance(
    model,
    X,
    y,
    method="permutation",
) -> pd.DataFrame
```

### Profiles and prediction

```python
create_analysis_profile(
    target_column,
    feature_columns,
    confidence_level=0.95,
)

build_risk_profile(
    feature_values: dict,
    metadata: dict,
) -> dict

predict_risk_profile(
    model,
    risk_profile,
    threshold=0.50,
) -> dict
```

---

## 20. Common Result Contract

Use a structured result contract, preferably a dataclass or Pydantic model:

```text
AnalysisResult
├── analysis_name
├── status
├── parameters
├── results
├── warnings
└── metadata
```

`metadata` should include, where applicable:

- dataset variant;
- dataset version;
- sample size;
- target definition;
- random seed;
- model;
- feature set;
- split strategy;
- training strategy;
- threshold;
- analysis timestamp.

Analytics functions return data and facts, not formatted Streamlit text.

---

## 21. Validation and Exceptions

Every public analytical function validates:

- required columns;
- types;
- valid values;
- non-empty groups;
- sample size where needed;
- nonzero denominators;
- required model features;
- confidence levels;
- thresholds;
- model parameters;
- target class count.

Custom exceptions:

```text
DataLoadError
DataValidationError
MissingColumnError
InvalidParameterError
UnsupportedAnalysisError
ModelTrainingError
```

The UI should convert expected exceptions into friendly messages rather than raw tracebacks.

---

## 22. Configuration and Metadata

Central configuration should include:

```python
PROJECT_CONFIG = {
    "target_column": "Diabetes_binary",
    "random_state": 42,
    "test_size": 0.20,
    "confidence_level": 0.95,
    "primary_ml_dataset_variant": "unique_profile_v1",
    "descriptive_dataset_variant": "full_clean_v1",
}
```

Dataset configuration:

```python
DATASET_CONFIG = {
    "kaggle_dataset": "alexteboul/diabetes-health-indicators-dataset",
    "primary_file": "diabetes_binary_health_indicators_BRFSS2015.csv",
    "target_column": "Diabetes_binary",
}
```

Variable metadata must include human-readable labels, type, valid range/category codes, and display labels.

---

## 23. Reproducibility and Artifacts

Use fixed random seeds, documented dataset source/version, explicit splits, documented preprocessing, feature lists, model parameters, thresholds, and experiment metadata.

Model artifacts should eventually contain:

```text
outputs/models/
├── model.pkl
├── metadata.json
└── metrics.json
```

Metadata:

- model type;
- training date;
- features;
- random state;
- training strategy;
- threshold;
- dataset variant;
- dataset version;
- metrics;
- preprocessing version.

Experiment contract:

```text
Experiment
├── ID
├── Model
├── Feature set
├── Dataset variant
├── Dataset version
├── Training strategy
├── Split strategy
├── Threshold
├── Random seed
├── Metrics
└── Timestamp
```

---

## 24. Testing Requirements

Each core function needs at least:

1. one normal case;
2. one invalid-input case;
3. one edge case.

Additional dataset/ML integrity tests are mandatory:

- expected 22-column schema;
- target is exactly `Diabetes_binary`;
- target contains only 0/1;
- no missing values in the verified supplied CSV;
- duplicate count is reported rather than silently discarded;
- `unique_profile_v1` row count is reproducible;
- train/test class distribution is documented;
- no exact duplicate feature rows cross train/test;
- target is excluded from feature matrices;
- test data are not used in balancing;
- preprocessing does not fit on test data;
- invalid threshold is rejected;
- single-class training data is rejected;
- missing prediction features are rejected.

Examples of ordinary edge cases:

- missing columns;
- empty groups;
- zero denominators;
- invalid confidence levels;
- invalid categories;
- single-class targets;
- invalid thresholds;
- missing model features.

---

## 25. Team Ownership

### Member 1 — Data & EDA

Owns:

- `src/data/*`;
- data dictionary;
- dataset verification;
- duplicate analysis;
- descriptive statistics;
- Data Explorer;
- EDA;
- data-quality outputs.

### Member 2 — Probability & Statistical Inference

Owns:

- probability;
- Bayes;
- confidence intervals;
- chi-square;
- Welch t-test;
- ANOVA;
- post-hoc tests;
- effect sizes;
- multiple comparisons;
- corresponding UI pages.

### Member 3 — AI & Predictive Modeling

Owns:

- regression;
- classifiers;
- duplicate-leakage controls;
- imbalance;
- thresholds;
- calibration;
- interpretation;
- Prediction Explorer;
- corresponding UI pages.

### Shared

All members share:

- integration;
- code review;
- testing;
- report;
- presentation;
- final validation;
- documentation consistency.

---

## 26. Report Requirements

The final report must contain:

1. Introduction and research objective.
2. Dataset/source and verification.
3. Target definition and terminology.
4. Data-quality and duplicate analysis.
5. Descriptive statistics.
6. Probability and Bayes analysis.
7. Confidence intervals.
8. Hypothesis testing.
9. Effect sizes and multiple-comparison handling.
10. Logistic regression.
11. BMI × Age interaction.
12. ML models.
13. Class-imbalance experiments.
14. Threshold analysis.
15. Calibration.
16. Model interpretation.
17. Limitations.
18. Conclusion.
19. References.

Every major table and figure must identify the dataset variant and analysis method where relevant.

---

## 27. Key Limitations

The project must explicitly discuss:

1. The Kaggle dataset is derived from BRFSS 2015 and is not the raw complex-survey analytical environment.
2. Survey weights/design are not reproduced.
3. The target positive class combines prediabetes and diabetes.
4. The dataset is cross-sectional/observational for this project; causal claims are inappropriate.
5. Exact duplicate feature rows exist and have no respondent identifier for definitive deduplication.
6. Primary ML deduplication is an analytical choice, not proof that duplicated rows are erroneous.
7. Predictive performance on this dataset does not establish clinical utility or external validity.
8. Threshold selection is context-dependent.
9. Calibration and discrimination can change across populations and deployment settings.
10. Results should not be generalized to current populations without appropriate validation.

---

## 28. Definition of Done

The project is complete only when:

- the supplied dataset passes the verification gate;
- raw data are preserved unchanged;
- dataset variants are explicitly defined;
- duplicate handling is documented;
- target terminology consistently reflects prediabetes-or-diabetes;
- descriptive and inferential analyses use the documented dataset variant;
- primary ML uses duplicate-controlled data;
- no duplicate feature rows cross the primary train/test split;
- test data remain untouched during training and balancing;
- all major statistical results expose appropriate uncertainty/effect-size information;
- ML results include multiple metrics;
- threshold and calibration analyses are reproducible;
- Streamlit pages use the service layer;
- expected errors are handled cleanly;
- unit/integration tests pass;
- report results trace to code;
- README and documentation agree with implementation;
- the project runs from a clean environment;
- no personal absolute paths, secrets, or credentials are committed.

---

## 29. References to Maintain

The repository should maintain authoritative references for:

- CDC BRFSS 2015 annual data/documentation;
- the BRFSS 2015 codebook/variable definitions;
- the Kaggle Diabetes Health Indicators Dataset;
- statistical-method references used for confidence intervals, hypothesis tests, effect sizes, regression, calibration, and ML evaluation.

Use APA 7 formatting in the final report and `references.md`.

