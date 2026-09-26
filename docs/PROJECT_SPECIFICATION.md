# Diabetes Health Analytics & Prediction Lab
## Complete Project Specification

**Project:** Understanding and Predicting Diabetes Risk: A Statistical and Machine Learning Study of Health, Lifestyle, and Socioeconomic Factors  
**Application:** Diabetes Health Analytics & Prediction Lab  
**Team:** 3 members  
**Duration:** 20 days  
**Framework:** Python + Streamlit  
**Academic context:** University of San Diego — M.S. in Applied Artificial Intelligence  
**Specification version:** 1.0

---

## 1. Purpose

This document is the authoritative specification for the project. It defines the research objectives, questions, hypotheses, dataset requirements, statistical and machine-learning methodology, software architecture, UI requirements, function contracts, validation, testing, team ownership, report requirements, limitations, and definition of done.

The companion document `IMPLEMENTATION_PLAN.md` defines the sequence and 20-day execution schedule.

If implementation decisions conflict with this document, update the specification before silently changing the design.

---

## 2. Executive Summary

The project investigates relationships between demographic, health, lifestyle, and socioeconomic indicators and diabetes status, then evaluates whether those variables can predict diabetes status using interpretable statistical and machine-learning methods.

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

---

## 3. Research Objective

### Primary objective

Investigate relationships between demographic, health, behavioral, and socioeconomic indicators and diabetes status, and evaluate whether those indicators can predict diabetes status using interpretable statistical and machine-learning methods.

### Secondary objectives

1. Understand dataset structure and quality.
2. Quantify diabetes distribution in the analytical sample.
3. Demonstrate marginal, joint, conditional, and posterior probabilities.
4. Quantify uncertainty with confidence intervals.
5. Test associations and group differences.
6. Report practical effect sizes alongside p-values.
7. Estimate adjusted associations using logistic regression.
8. Investigate BMI × Age interaction.
9. Compare three classification models.
10. Investigate class imbalance.
11. Examine threshold trade-offs.
12. Assess probability calibration.
13. Provide transparent model interpretation.
14. Build a reusable Streamlit research laboratory.

---

## 4. Research Questions

- **RQ1:** What demographic, health, lifestyle, and socioeconomic characteristics describe the analytical sample?
- **RQ2:** How is `Diabetes_binary` distributed?
- **RQ3:** How does diabetes probability change under selected conditions such as high blood pressure, high cholesterol, physical inactivity, and demographic categories?
- **RQ4:** Can Bayes theorem be demonstrated and independently verified from the observed data?
- **RQ5:** What confidence intervals describe selected proportions, means, and group differences?
- **RQ6:** Which variables show statistically detectable associations or differences with diabetes status?
- **RQ7:** How large are the observed effects, and how should they be interpreted alongside p-values?
- **RQ8:** Which variables remain associated with diabetes after simultaneous adjustment?
- **RQ9:** Does the BMI–diabetes relationship vary by age?
- **RQ10:** How do Logistic Regression, Random Forest, and Gradient Boosting compare across multiple metrics?
- **RQ11:** How does class imbalance affect model behavior?
- **RQ12:** How do thresholds affect precision, recall, specificity, and F1, and are predicted probabilities reasonably calibrated?

---

## 5. Hypotheses

| ID | Alternative/research hypothesis | Null hypothesis |
|---|---|---|
| H1 | Diabetes status is associated with HighBP. | Diabetes status and HighBP are independent. |
| H2 | Diabetes status is associated with HighChol. | Diabetes status and HighChol are independent. |
| H3 | Mean BMI differs between diabetes groups. | Mean BMI is equal between groups. |
| H4 | Diabetes status is associated with physical activity. | Diabetes status and physical activity are independent. |
| H5 | Diabetes status is associated with smoking. | Diabetes status and smoking are independent. |
| H6 | Diabetes status differs across age categories. | Diabetes status is independent of age category. |
| H7 | Diabetes status differs across income categories. | Diabetes status is independent of income category. |
| H8 | Selected predictors remain associated after adjustment. | Selected predictors have no adjusted association. |
| H9 | The BMI–diabetes association varies with age. | There is no BMI × Age interaction. |
| H10 | Class-imbalance strategy changes predictive behavior. | Relevant predictive behavior is unchanged. |

H10 is treated primarily as an experimental/modeling question rather than a conventional population hypothesis test.

---

## 6. Dataset Specification

### Source

Primary dataset: Kaggle **Diabetes Health Indicators Dataset** associated with BRFSS 2015.

Expected Kaggle identifier:

`alexteboul/diabetes-health-indicators-dataset`

Expected primary binary file:

`diabetes_binary_health_indicators_BRFSS2015.csv`

These values are **provisional until the downloaded dataset is inspected**.

### Download strategy

Use `kagglehub` for reproducible acquisition. The data flow is:

```text
Kaggle → kagglehub → data/raw/ → validation → analysis-ready data
```

The Streamlit application should not repeatedly download from Kaggle during ordinary navigation.

### Raw-data rule

Never manually edit the raw file. Do not silently delete records, rename columns, or recode values in place. Transformations belong in the data-preparation layer.

### Target

Expected target: `Diabetes_binary`.

The exact target coding **must be verified against the downloaded CSV before implementation**. Do not use a guessed column such as `diabetes`.

### Mandatory verification gate

Before full analysis implementation, verify:

- exact filename;
- row and column counts;
- exact column names;
- target column and unique values;
- class counts;
- missing values;
- duplicates;
- data types;
- binary coding;
- Age mapping;
- Sex mapping;
- Education mapping;
- Income mapping;
- categorical encodings;
- unexpected values.

The verified result becomes `docs/DATA_DICTIONARY.md`.

---

## 7. Methodological Principles

### Association is not causation

Use “associated with,” “observed relationship,” or “predictive signal.” Do not use causal language such as “causes,” “prevents,” or “leads to.”

### Prediction is not diagnosis

The prediction page may show a model-estimated probability and classification at a selected threshold. It must never state that a person has diabetes or provide medical advice.

### Statistical significance is not practical importance

Do not label a variable important solely because `p < .05`. Present estimate, confidence interval, p-value, effect size, sample size, and context.

### Class imbalance

Class imbalance is a research question. The original test distribution must be retained for fair evaluation.

### Test-set integrity

The test set must not be balanced, oversampled, undersampled, or used for training. Preprocessing that learns parameters must be fit using training data only.

### BRFSS survey limitation

BRFSS is a complex survey and official population inference uses survey design and weights. This project uses a cleaned Kaggle analytical dataset and does **not** reproduce the full complex-survey analysis. Unweighted sample percentages must not be presented as official U.S. population prevalence estimates.

---

## 8. Statistical Analysis Plan

### Descriptive statistics

Continuous variables: n, mean, median, SD, variance, minimum, Q1, Q3, IQR, maximum.

Categorical variables: category, count, percentage.

### Probability

Implement marginal probability, joint probability, conditional probability, Bayes theorem, and profile-based probability. Report numerator and denominator.

### Confidence intervals

Implement proportion CI, mean CI, and difference-in-means CI. Document the method.

### Chi-square

Report contingency table, chi-square statistic, df, p-value, Cramér’s V, and n.

### Welch t-test

Report group n, means, SDs, mean difference, t, df, p-value, CI, and effect size.

### ANOVA

Report group statistics, F, df, p-value, effect size, and post-hoc comparisons where justified.

### Multiple comparisons

When a family of tests is performed, consider Benjamini-Hochberg/FDR. Report original and adjusted p-values and state the correction strategy.

### Correlation

Support a documented correlation method such as Spearman for appropriate exploratory relationships.

---

## 9. Regression Plan

Primary binary outcome: `Diabetes_binary`.

Candidate predictors include BMI, Age, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Education, and Income, subject to verification and final feature selection.

### Logistic regression outputs

- coefficients;
- standard errors;
- p-values;
- odds ratios;
- 95% CIs;
- model statistics;
- design metadata.

### Interaction

At minimum evaluate BMI × Age. Compare baseline and interaction models and report the interaction coefficient, CI, p-value, and appropriate model-comparison statistics.

---

## 10. Machine-Learning Plan

### Models

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

### Split

Default: 80/20 stratified train/test split, random seed 42.

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

Because the target may be imbalanced, accuracy must not be treated as the only performance measure.

### Comparison

Show metrics side by side. The comparison function must not embed a winner/ranking. Interpretation belongs in the research discussion.

---

## 11. Class-Imbalance Experiments

Compare:

1. original training distribution;
2. class-weighted model;
3. balanced training data;
4. threshold adjustment.

Initial balancing method: undersampling, if retained after testing.

Only training data may be balanced. Test data remains unchanged.

---

## 12. Threshold Analysis

Evaluate configurable thresholds, for example 0.10 through 0.90 in 0.05 increments.

For each threshold report threshold, precision, recall, specificity, and F1.

The UI should expose trade-offs rather than silently selecting a “best” threshold.

---

## 13. Calibration

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

---

## 14. Model Interpretation

Provide logistic-regression odds ratios and permutation feature importance. Native tree importance may be supplementary.

Interpretation remains descriptive and non-causal.

---

## 15. Streamlit Application Specification

Application: **Diabetes Health Analytics & Prediction Lab**.

The interface is an interactive research laboratory rather than only a dashboard.

### Pages

1. **Home / Executive Dashboard** — dataset overview, sample size, target distribution, selected findings, research questions, methodology warning.
2. **Data Explorer** — schema, data types, missingness, categories, quality, descriptive statistics.
3. **Exploratory Analysis** — distributions, target-group comparisons, prevalence, correlations.
4. **Probability Explorer** — marginal, joint, conditional, Bayes, profile probability.
5. **Statistical Testing Lab** — chi-square, Welch t-test, ANOVA, post-hoc, effect size, multiple comparisons.
6. **Regression Lab** — baseline logistic regression, interaction model, odds ratios, CIs, model comparison.
7. **Machine Learning Lab** — models, training, metrics, confusion matrix, ROC, PR, comparison.
8. **Imbalance & Threshold Lab** — class distribution, weighting, balancing, threshold curves.
9. **Calibration** — calibration curve/table and Brier score.
10. **Interpretation** — feature importance and odds ratios.
11. **Prediction Explorer** — feature profile, selected model, predicted probability, threshold, classification, model metadata.

The prediction page must never use diagnosis language.

---

## 16. Software Architecture

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

---

## 17. Function Contracts

### Data

```python
download_kaggle_dataset(dataset_id: str, output_directory: str) -> str
load_dataset(file_path: str) -> pd.DataFrame
validate_dataset(data, required_columns, target_column) -> dict
generate_data_quality_report(data) -> dict
prepare_analysis_data(data, target_column, feature_columns, drop_missing=False) -> pd.DataFrame
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
calculate_conditional_probability(data, target_column, target_value, condition_columns, condition_values) -> dict
calculate_joint_probability(data, conditions: dict) -> dict
verify_bayes_theorem(data, target_column, target_value, condition_column, condition_value) -> dict
calculate_risk_profile_probability(data, target_column, target_value, profile: dict) -> dict
```

### Confidence intervals

```python
proportion_confidence_interval(successes, total, confidence_level=0.95) -> dict
mean_confidence_interval(values, confidence_level=0.95) -> dict
difference_in_means_ci(group_a, group_b, confidence_level=0.95) -> dict
```

### Hypothesis testing

```python
run_chi_square_test(data, feature, target_column) -> dict
run_welch_t_test(data, numeric_feature, target_column, group_a, group_b) -> dict
run_anova(data, numeric_feature, grouping_feature) -> dict
run_posthoc_analysis(data, numeric_feature, grouping_feature, method="tukey") -> pd.DataFrame
adjust_multiple_comparisons(p_values, method="fdr_bh") -> pd.DataFrame
```

### Association

```python
calculate_correlation_matrix(data, columns, method="spearman") -> pd.DataFrame
calculate_effect_size(analysis_type, data, **kwargs) -> dict
```

### Regression

```python
fit_logistic_regression(data, target_column, feature_columns, interaction_terms=None) -> dict
compare_regression_models(baseline_model, interaction_model) -> dict
extract_odds_ratios(fitted_model) -> pd.DataFrame
```

### ML

```python
split_data(data, target_column, test_size=0.20, random_state=42, stratify=True) -> dict
train_logistic_classifier(X_train, y_train, class_weight=None, random_state=42) -> object
train_random_forest(X_train, y_train, class_weight=None, random_state=42, **model_parameters) -> object
train_gradient_boosting(X_train, y_train, random_state=42, **model_parameters) -> object
generate_predictions(model, X, threshold=0.50) -> dict
evaluate_classifier(y_true, y_pred, y_probability) -> dict
compare_models(evaluation_results: dict) -> pd.DataFrame
```

### Imbalance, threshold, calibration, interpretation

```python
inspect_class_distribution(y) -> dict
balance_training_data(X_train, y_train, method="undersample", random_state=42) -> dict
train_class_weighted_model(model_type, X_train, y_train, random_state=42) -> object
evaluate_thresholds(y_true, probabilities, thresholds: list[float]) -> pd.DataFrame
generate_threshold_curve(threshold_results)
calculate_calibration_metrics(y_true, probabilities, n_bins=10) -> dict
generate_calibration_data(y_true, probabilities, n_bins=10) -> pd.DataFrame
calculate_feature_importance(model, X, y, method="permutation") -> pd.DataFrame
```

### Profiles and prediction

```python
create_analysis_profile(target_column, feature_columns, confidence_level=0.95)
build_risk_profile(feature_values: dict, metadata: dict) -> dict
predict_risk_profile(model, risk_profile, threshold=0.50) -> dict
```

---

## 18. Common Result Contract

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

Analytics functions return data and facts, not formatted Streamlit text.

---

## 19. Validation and Exceptions

Every public analytical function validates required columns, types, valid values, non-empty groups, sample size where needed, nonzero denominators, required model features, confidence levels, thresholds, and model parameters.

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

## 20. Configuration and Metadata

Central configuration should include:

```python
PROJECT_CONFIG = {
    "target_column": "Diabetes_binary",
    "random_state": 42,
    "test_size": 0.20,
    "confidence_level": 0.95,
}
```

Also maintain dataset, model, UI, and variable metadata. Exact category mappings must come from the verified dataset.

---

## 21. Reproducibility and Artifacts

Use fixed random seeds, documented dataset source/version, explicit splits, documented preprocessing, feature lists, model parameters, thresholds, and experiment metadata.

Model artifacts should eventually contain:

```text
outputs/models/
├── model.pkl
├── metadata.json
└── metrics.json
```

Metadata: model type, training date, features, random state, training strategy, threshold, dataset version, metrics.

Experiment contract:

```text
Experiment
├── ID
├── Model
├── Feature set
├── Training strategy
├── Threshold
├── Dataset version
├── Random seed
├── Metrics
└── Timestamp
```

---

## 22. Testing Requirements

Each core function needs at least one normal case, one invalid-input case, and one edge case.

Examples include missing columns, empty groups, zero denominators, invalid confidence levels, invalid categories, single-class targets, invalid thresholds, and missing model features.

---

## 23. Team Ownership

### Member 1 — Data & EDA

Owns `src/data/*`, descriptive statistics, data dictionary, data explorer, EDA, and data-quality outputs.

### Member 2 — Probability & Statistical Inference

Owns probability, Bayes, confidence intervals, chi-square, Welch t-test, ANOVA, post-hoc tests, effect sizes, multiple comparisons, and corresponding UI pages.

### Member 3 — AI & Predictive Modeling

Owns regression, classifiers, imbalance, thresholds, calibration, interpretation, prediction explorer, and corresponding UI pages.

### Shared

All members share integration, code review, testing, report, presentation, and final validation.

---

## 24. GitHub Workflow

Recommended branches:

```text
main
develop
feature/member1-data
feature/member2-statistics
feature/member3-modeling
```

Workflow: issue → feature branch → implementation → tests → commit → pull request → review → merge.

Never commit virtual environments, `.DS_Store`, credentials, API keys, private data, or unnecessary secrets.

---

## 25. Report Requirements

The report should cover title, abstract, introduction, problem, objectives, research questions, hypotheses, dataset, preparation, EDA, probability, confidence intervals, hypothesis testing, effect sizes, multiple comparisons, logistic regression, interaction analysis, ML methodology, imbalance, threshold, calibration, interpretation, results, discussion, limitations, responsible use, conclusion, references, and appendix.

Every major reported number should be traceable to reproducible code/output.

---

## 26. Presentation Requirements

Target 15–18 slides covering problem, objectives, dataset, research questions, data quality, EDA, probability/Bayes, inference, regression, interaction, ML, imbalance, threshold/calibration, application, findings, limitations, and conclusion.

---

## 27. Limitations

Explicitly discuss observational data, inability to establish causality, self-reported measures where applicable, 2015 temporal limitation, processed Kaggle data, source preprocessing, class imbalance, model assumptions, generalizability, lack of clinical validation, lack of patient-level context, and BRFSS complex-survey weighting/design limitations.

---

## 28. Responsible Use

The application must clearly state its educational/research purpose; avoid medical recommendations and diagnosis claims; expose uncertainty and threshold settings; distinguish association from prediction; and avoid implying that a model probability is an individual's true medical probability.

---

## 29. Definition of Done

The project is complete when:

- dataset acquisition is reproducible;
- schema and target coding are verified;
- data dictionary is finalized;
- data-quality checks pass;
- statistical analyses are implemented and tested;
- regression and ML models work;
- imbalance, threshold, calibration, and interpretation are complete;
- all Streamlit pages work;
- UI uses the service/analytics architecture;
- tests pass;
- README and documentation are complete;
- report and presentation are complete;
- APA 7 references are included;
- no personal paths or credentials are committed;
- a clean environment can run the project.

---

## 30. Guiding Principle

> **Analytics functions return facts. UI functions present facts. Neither layer invents conclusions.**
