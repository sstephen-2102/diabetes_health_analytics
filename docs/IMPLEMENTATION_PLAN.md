# Diabetes Health Analytics & Prediction Lab
## Complete 20-Day Implementation Plan

**Specification:** `docs/PROJECT_SPECIFICATION.md`  
**Plan version:** 1.1  
**Duration:** 20 days  
**Team:** 3 members  
**Primary stack:** Python, pandas, NumPy, SciPy/statsmodels, scikit-learn, Plotly, Streamlit, kagglehub  
**Dataset verification:** COMPLETE — supplied CSV verified on 2026-09-29

---

## 1. Implementation Philosophy

Build the system in layers. Do not begin by putting statistical calculations into Streamlit pages.

Required dependency direction:

```text
APP → SERVICE → ANALYTICS → DATA
```

Analytics functions must be reusable by Streamlit, tests, notebooks, and report-generation workflows.

### Non-negotiable principles

1. Raw data are immutable.
2. Dataset variants are explicit.
3. The positive target class is described as **prediabetes or diabetes**, not confirmed diabetes.
4. Exact duplicates are measured and documented, never silently discarded.
5. Primary ML uses `unique_profile_v1`.
6. The primary train/test split is stratified 80/20 with `random_state=42`.
7. Exact duplicate feature rows must not cross the primary train/test boundary.
8. Test data remain untouched during balancing, preprocessing fitting, threshold selection, and model fitting.
9. Statistical results report practical context, not p-values alone.
10. The Streamlit layer presents results; it does not invent analytical conclusions.

---

## 2. Build Sequence

1. Research design and repository alignment
2. Dataset verification gate — **COMPLETE**
3. Data dictionary and metadata
4. Configuration and schemas
5. Data layer and explicit dataset variants
6. Descriptive statistics
7. Probability
8. Inferential statistics and assumption checks
9. Regression
10. Machine learning
11. Class imbalance
12. Thresholds
13. Calibration
14. Interpretation
15. Service layer
16. Streamlit core UI
17. Advanced Streamlit UI
18. Integration and QA
19. Report and presentation
20. Final audit and rehearsal

---

## 3. Day 1 — Research Design and Repository Alignment

### All members

Confirm:

- project title;
- application name;
- research objective;
- research questions;
- hypotheses;
- dataset;
- scope;
- ownership;
- documentation structure;
- terminology rules.

### Member 1

Review dataset source and acquisition process.

### Member 2

Map every research question to a statistical method and required output.

### Member 3

Map predictive research questions to models, metrics, imbalance experiments, threshold analysis, calibration, and interpretation.

### Deliverables

- approved specification;
- approved implementation plan;
- repository structure;
- initial GitHub issues.

### Gate

All members agree that `Diabetes_binary=1` means **prediabetes or diabetes** and that the project is not a clinical diagnostic system.

---

## 4. Day 2 — Dataset Verification Gate — COMPLETE

The supplied CSV has been inspected.

### Verified facts

- **253,680 rows**
- **22 columns**
- **0 missing cells**
- **24,206 exact duplicate rows**
- **9.5419% exact duplicate-row rate**
- **229,474 rows after complete-row deduplication**
- target `Diabetes_binary`
- target counts: 218,334 zeros and 35,346 ones
- all 22 columns are numeric in the supplied CSV.

### Member 1

Preserve the raw file unchanged and generate the initial data-quality report.

### Member 2

Confirm that the verified schema supports the planned probability and inferential analyses.

### Member 3

Confirm candidate ML features, target encoding, and preprocessing requirements.

### Deliverables

- raw dataset under `data/raw/`;
- verification notes;
- initial quality report;
- `docs/DATA_DICTIONARY.md`;
- dataset-version decision record.

### Gate

The verification gate is complete. Future runs must still validate the same conditions and fail fast if the downloaded file differs unexpectedly.

---

## 5. Day 3 — Data Dictionary, Configuration, and Package Foundation

Implement:

- package `__init__` files;
- `config.py`;
- `schemas.py`;
- `exceptions.py`;
- dataset configuration;
- model configuration;
- variable metadata;
- human-readable category labels;
- `.gitignore`;
- requirements.

### Required configuration

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

### Deliverable

The project imports cleanly from a fresh environment and the verified schema is represented centrally.

---

## 6. Day 4 — Data Layer and Dataset Variants

Member 1 implements:

```python
download_kaggle_dataset()
load_dataset()
validate_dataset()
generate_data_quality_report()
prepare_analysis_data()
```

### Required variants

```text
raw_v1
full_clean_v1
unique_profile_v1
balanced_train_v1
```

### Rules

- `raw_v1`: exact downloaded CSV.
- `full_clean_v1`: verified complete dataset with no silent row removal.
- `unique_profile_v1`: complete-row duplicate removal.
- `balanced_train_v1`: derived only from a training partition.

### Validation must detect

- missing files;
- malformed CSVs;
- missing columns;
- unexpected target values;
- invalid categories;
- empty datasets;
- unexpected schema changes;
- duplicate-count changes.

### Deliverable

Reproducible raw-to-analysis-ready data pipeline.

---

## 7. Day 5 — Descriptive Statistics and EDA

Implement numeric and categorical summaries, target comparisons, prevalence tables, and initial visualizations.

### Required EDA

- target distribution;
- duplicate-rate summary;
- demographic summaries;
- health/lifestyle summaries;
- socioeconomic summaries;
- target-group comparisons;
- selected conditional prevalence tables;
- distributions;
- correlation exploration.

### Sensitivity requirement

For important findings, compare `full_clean_v1` and `unique_profile_v1` where duplicate handling could change the interpretation.

### Deliverable

Baseline EDA tables and figures with dataset-variant labels.

---

## 8. Day 6 — Probability

Member 2 implements:

```python
calculate_probability()
calculate_conditional_probability()
calculate_joint_probability()
verify_bayes_theorem()
calculate_risk_profile_probability()
```

Every result should expose:

- numerator;
- denominator;
- probability;
- target definition;
- conditions;
- dataset variant;
- sample size.

### Required demonstrations

At least one reproducible example each for:

- marginal probability;
- joint probability;
- conditional probability;
- Bayes theorem;
- multi-feature/profile probability.

### Deliverable

Verified probability examples suitable for report and UI.

---

## 9. Day 7 — Confidence Intervals

Implement:

- proportion confidence interval;
- mean confidence interval;
- difference-in-means confidence interval.

### Validation

Test:

- standard cases;
- small samples where the chosen method remains valid;
- zero denominators;
- invalid confidence levels;
- empty inputs.

Document the interval method.

### Deliverable

Tested CI module.

---

## 10. Day 8 — Hypothesis Testing and Assumption Checks

Implement:

- chi-square;
- Welch t-test;
- ANOVA;
- post-hoc analysis;
- effect sizes;
- multiple-comparison adjustment.

### Required reporting

- test statistic;
- df where applicable;
- p-value;
- effect size;
- sample size;
- confidence interval where applicable;
- assumption/diagnostic notes.

### Multiple comparisons

Use a documented Benjamini-Hochberg/FDR strategy for planned families of tests where appropriate.

### Assumption checks

Document relevant assumptions and limitations rather than mechanically applying tests without checking data characteristics.

### Deliverable

Complete inferential-statistics core.

---

## 11. Day 9 — Association and Statistical Integration

Implement:

- Spearman correlation;
- unified statistical result structures;
- report-ready tables;
- warnings/metadata.

Review:

- variable treatment;
- assumptions;
- effect-size interpretation;
- multiple-comparison strategy;
- dataset-variant labeling.

### Deliverable

All planned core statistical outputs can be generated outside Streamlit.

---

## 12. Day 10 — Logistic Regression

Member 3 implements:

```python
fit_logistic_regression()
extract_odds_ratios()
compare_regression_models()
```

Build:

1. baseline model;
2. BMI × Age interaction model.

### Required decisions

Document whether:

- Age is treated as ordinal numeric or categorical;
- Education is treated as ordinal numeric or categorical;
- Income is treated as ordinal numeric or categorical;
- GenHlth is treated as ordinal numeric or categorical;
- MentHlth and PhysHlth are modeled as count-like numeric variables.

### Outputs

- coefficients;
- standard errors;
- p-values;
- odds ratios;
- 95% CIs;
- sample size;
- model-comparison statistics;
- feature-treatment metadata;
- dataset variant.

### Deliverable

Reproducible regression results.

---

## 13. Day 11 — Primary Classification Models

Implement:

```python
split_data()
train_logistic_classifier()
train_random_forest()
train_gradient_boosting()
generate_predictions()
evaluate_classifier()
compare_models()
```

### Primary ML protocol

1. Start from `unique_profile_v1`.
2. Create stratified 80/20 train/test split.
3. Fix `random_state=42`.
4. Keep test set untouched.
5. Fit preprocessing on training data only.
6. Verify no exact duplicate feature rows cross train/test.
7. Train the three baseline models.
8. Evaluate on the untouched test set.

### Leakage tests

Add an automated assertion that no exact duplicate feature vectors occur across train and test.

### Deliverable

Three baseline classifiers and common metrics with a recorded experiment configuration.

---

## 14. Day 12 — Class Imbalance

Implement:

```python
inspect_class_distribution()
balance_training_data()
train_class_weighted_model()
```

Compare:

1. original training distribution;
2. class-weighted model;
3. balanced training data;
4. threshold adjustment as a separate experiment.

### Rules

- balance training data only;
- never rebalance the final test set;
- preserve the natural test distribution;
- record before/after training counts.

### Deliverable

Comparable imbalance experiment table.

---

## 15. Day 13 — Threshold and Calibration

Implement:

```python
evaluate_thresholds()
generate_threshold_curve()
calculate_calibration_metrics()
generate_calibration_data()
```

### Thresholds

Default grid:

```text
0.10, 0.15, 0.20, ..., 0.90
```

Report:

- threshold;
- precision;
- recall;
- specificity;
- F1;
- confusion-matrix counts.

### Calibration

Report:

- Brier score;
- calibration curve;
- calibration table;
- bin size/count;
- predicted probability;
- observed event frequency.

### Important control

Do not select a final threshold using the final test set without documenting the resulting evaluation bias. If threshold selection is optimized, introduce a validation strategy or cross-validation and reserve the final test set for one-time evaluation.

### Deliverable

Threshold and calibration outputs.

---

## 16. Day 14 — Model Interpretation

Implement permutation feature importance and odds-ratio extraction.

### Requirements

- use an explicitly documented evaluation split;
- report uncertainty/context where available;
- avoid causal language;
- distinguish statistical association from predictive importance.

### Deliverable

Transparent interpretation outputs.

---

## 17. Day 15 — Service Layer

Implement:

```python
get_dashboard_summary()
get_eda_summary()
get_probability_analysis()
get_statistical_test_result()
get_model_comparison()
get_prediction_result()
```

The service layer orchestrates existing analytics; it must not duplicate statistical formulas.

### Service result requirements

Include:

- status;
- results;
- warnings;
- dataset variant;
- metadata;
- target definition where relevant.

### Deliverable

Streamlit can consume stable service contracts.

---

## 18. Day 16 — Streamlit Core UI

Build:

- application shell;
- navigation;
- Home / Executive Dashboard;
- Data Explorer;
- Exploratory Analysis;
- shared components.

Shared components should include:

- metric cards;
- result tables;
- probability result cards;
- statistical result cards;
- confusion matrices;
- warning/limitation panels.

### Data Explorer must show

- row/column count;
- missingness;
- duplicate count/rate;
- target distribution;
- variable types;
- category mappings;
- dataset variant.

### Deliverable

Core UI connected to real analytics.

---

## 19. Day 17 — Advanced Streamlit UI

Build:

- Probability Explorer;
- Statistical Testing Lab;
- Regression Lab;
- Machine Learning Lab;
- Imbalance & Threshold Lab;
- Calibration;
- Interpretation;
- Prediction Explorer.

### Prediction Explorer

Show:

- selected model;
- model-estimated probability of the positive `Diabetes_binary` class;
- threshold;
- model classification;
- model metadata;
- methodology warning.

Do not use diagnostic language.

### Deliverable

Complete research laboratory.

---

## 20. Day 18 — Integration and QA

Merge team branches and run:

- unit tests;
- integration tests;
- complete analytics pipeline;
- Streamlit smoke tests;
- prediction-page validation;
- invalid-input tests;
- duplicate-leakage test;
- clean-environment test.

### Final data audit

Confirm:

- raw dataset unchanged;
- schema matches expected 22 columns;
- target is `Diabetes_binary`;
- target wording is correct;
- duplicate counts are documented;
- `unique_profile_v1` is reproducible;
- primary ML uses `unique_profile_v1`;
- no exact duplicate feature rows cross train/test;
- test set is not balanced;
- all outputs identify dataset variant.

### Deliverable

Release candidate.

---

## 21. Day 19 — Report and Presentation

Complete:

1. methodology;
2. dataset verification;
3. duplicate analysis;
4. descriptive results;
5. probability/Bayes results;
6. confidence intervals;
7. hypothesis testing;
8. effect sizes;
9. multiple-comparison handling;
10. logistic regression;
11. BMI × Age interaction;
12. ML results;
13. imbalance experiments;
14. threshold analysis;
15. calibration;
16. model interpretation;
17. limitations;
18. conclusion;
19. references.

### Traceability rule

Every major reported value must be traceable to a reproducible code output.

### Presentation

Target 15–18 slides covering:

- problem;
- dataset;
- research questions;
- statistical methodology;
- key results;
- ML methodology;
- imbalance/threshold/calibration;
- limitations;
- application demonstration;
- conclusions.

### Deliverable

Draft final report and presentation.

---

## 22. Day 20 — Final Audit and Rehearsal

Perform:

- clean-environment installation;
- test-suite run;
- Streamlit launch;
- full pipeline execution;
- output inspection;
- GitHub review;
- documentation review;
- report-to-code traceability check;
- presentation rehearsal.

### Final audit checklist

- no unresolved critical defects;
- no personal absolute paths;
- no secrets/API keys;
- no raw-data mutation;
- no unsupported clinical claims;
- no “diagnosis” language;
- target consistently described as prediabetes-or-diabetes outcome;
- duplicate handling documented;
- primary ML leakage control verified;
- statistical methods documented;
- effect sizes included;
- multiple comparisons addressed;
- test-set integrity preserved;
- model metadata stored;
- references formatted in APA 7.

### Deliverable

Final release candidate, report, presentation, and reproducibility package.

---

## 23. Team Parallelization

```text
                 COMMON CONTRACTS
                       │
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
   Member 1         Member 2         Member 3
   Data/EDA         Statistics        AI/ML
       │               │               │
       └───────────────┼───────────────┘
                       ↓
                  Integration
                       ↓
                  Streamlit
                       ↓
              Report + Presentation
```

Shared contracts should be frozen before parallel implementation begins.

---

## 24. GitHub Workflow

Branches:

```text
main
develop
feature/member1-data
feature/member2-statistics
feature/member3-modeling
```

Workflow:

```text
Issue → feature branch → implementation → tests → commit → pull request → review → merge
```

Substantial changes should receive teammate review.

Suggested commit prefixes:

```text
feat:
fix:
test:
docs:
refactor:
```

Never commit:

- `.venv/`;
- `.DS_Store`;
- credentials;
- API keys;
- personal absolute paths;
- private/raw data that should not be distributed.

---

## 25. GitHub Issue Backlog

### Setup

- SETUP-001 Repository structure
- SETUP-002 Dependencies
- SETUP-003 Configuration
- SETUP-004 Exceptions and schemas

### Data

- DATA-001 Kaggle download
- DATA-002 Schema verification
- DATA-003 Data dictionary
- DATA-004 Validation
- DATA-005 Quality report
- DATA-006 Dataset variants
- DATA-007 Duplicate analysis
- DATA-008 Preprocessing

### Statistics

- STAT-001 Descriptive statistics
- STAT-002 Probability
- STAT-003 Bayes theorem
- STAT-004 Confidence intervals
- STAT-005 Chi-square
- STAT-006 Welch t-test
- STAT-007 ANOVA
- STAT-008 Post-hoc analysis
- STAT-009 Effect sizes
- STAT-010 Multiple comparisons
- STAT-011 Assumption checks
- STAT-012 Sensitivity analysis

### Modeling

- ML-001 Duplicate-controlled split
- ML-002 Logistic classifier
- ML-003 Random Forest
- ML-004 Gradient Boosting
- ML-005 Evaluation
- ML-006 Model comparison
- ML-007 Imbalance
- ML-008 Thresholds
- ML-009 Calibration
- ML-010 Interpretation
- ML-011 Prediction Explorer
- ML-012 Leakage tests

### UI

- UI-001 App shell
- UI-002 Dashboard
- UI-003 Data Explorer
- UI-004 EDA
- UI-005 Probability
- UI-006 Statistics
- UI-007 Regression
- UI-008 ML
- UI-009 Imbalance
- UI-010 Calibration
- UI-011 Interpretation
- UI-012 Prediction

### QA

- QA-001 Unit tests
- QA-002 Integration tests
- QA-003 Dataset integrity tests
- QA-004 Leakage tests
- QA-005 Clean environment test
- QA-006 Documentation review
- QA-007 Report-to-code traceability
- QA-008 Final reproducibility audit

---

## 26. Function Definition of Done

A function is complete only when:

- its signature matches the contract;
- its docstring defines behavior;
- inputs are validated;
- normal, invalid, and edge cases are tested;
- its return structure is documented;
- it contains no UI-specific code;
- it contains no personal absolute paths;
- results are reproducible;
- dataset variant is propagated where relevant.

---

## 27. UI Definition of Done

A page is complete when it:

- loads successfully;
- uses the service/analytics layer;
- validates inputs;
- handles expected errors cleanly;
- labels charts and statistics;
- identifies the dataset variant;
- avoids causal/diagnostic claims;
- uses the correct positive-class terminology;
- works with the verified dataset.

---

## 28. Statistical Result Definition of Done

Every major result should state or expose:

- what was analyzed;
- dataset variant;
- sample size;
- estimate;
- uncertainty;
- test statistic where applicable;
- p-value where applicable;
- effect size where applicable;
- assumptions/limitations where applicable.

---

## 29. ML Result Definition of Done

Every model experiment should record:

- model type;
- feature set;
- dataset variant;
- split strategy;
- seed;
- training strategy;
- threshold;
- accuracy;
- precision;
- recall;
- specificity;
- F1;
- ROC-AUC;
- PR-AUC;
- Brier score where applicable;
- confusion matrix;
- leakage checks.

---

## 30. Dataset Decision Record

### Decision 1 — Target terminology

**Decision:** `Diabetes_binary=1` is documented as **prediabetes or diabetes**.

**Reason:** The target definition must not be overstated as confirmed clinical diabetes.

### Decision 2 — Duplicate rows

**Decision:** Preserve raw data, report exact duplicates, and create `unique_profile_v1` for primary ML.

**Reason:** The dataset has 24,206 exact duplicate rows and no respondent identifier. Silent removal would hide an important analytical decision; leaving identical rows across train/test could make model evaluation overly optimistic.

### Decision 3 — Dataset variants

**Decision:** Use explicit versioned variants rather than implicit data transformations.

**Reason:** Every report result and experiment must be reproducible.

### Decision 4 — Primary ML split

**Decision:** Stratified 80/20 split from `unique_profile_v1`, seed 42, with test data untouched.

**Reason:** Preserve class representation while reducing duplicate-row leakage.

### Decision 5 — Full-data sensitivity analysis

**Decision:** Retain `full_clean_v1` for descriptive/inferential analysis and optionally evaluate ML sensitivity against it.

**Reason:** This preserves the supplied dataset for statistical description while making the predictive experiment more conservative.

---

## 31. Testing Matrix

| Area | Normal | Invalid | Edge / integrity |
|---|---|---|---|
| Data loading | valid CSV | missing/malformed file | empty file |
| Schema | expected 22 columns | missing column | unexpected extra/missing schema |
| Target | 0/1 | unexpected value | single-class target |
| Duplicates | reproducible count | invalid variant | exact duplicates across split |
| Probability | valid event | invalid category | zero denominator |
| CI | valid sample | invalid confidence | small/empty sample |
| Chi-square | valid contingency table | missing feature | sparse expected counts |
| Welch t-test | two groups | invalid group | empty/small group |
| ANOVA | 3+ groups | invalid grouping | empty group |
| Regression | valid features | missing feature | separation/degenerate design |
| ML split | stratified split | invalid test size | duplicate leakage |
| Balancing | training-only | invalid method | single-class input |
| Threshold | 0–1 | outside range | threshold at 0 or 1 |
| Calibration | valid probabilities | invalid probability | empty/single-class test |
| Prediction | complete profile | missing feature | boundary category |
| UI | valid inputs | invalid inputs | failed model / empty result |

---

## 32. Required Final Artifacts

By the end of Day 20:

```text
docs/
├── PROJECT_SPECIFICATION.md
├── IMPLEMENTATION_PLAN.md
├── DATA_DICTIONARY.md
├── references.md
└── report_template.md

src/
├── common/
├── data/
├── statistics/
├── modeling/
└── analysis/

app/
├── app.py
├── pages/
└── components/

tests/

data/
├── raw/
└── processed/

outputs/
├── figures/
├── tables/
└── models/

README.md
requirements.txt
```

The repository must make it possible for a teammate or evaluator to understand:

1. exactly which dataset was analyzed;
2. what the target means;
3. how duplicates were handled;
4. how the train/test split was created;
5. how statistics were computed;
6. how models were trained;
7. how performance was evaluated;
8. how results map to the final report.

---

## 33. Final Success Criteria

The project succeeds when the team can demonstrate a complete, reproducible chain:

```text
RAW DATA
   ↓
VERIFICATION
   ↓
EXPLICIT DATASET VARIANTS
   ↓
EDA + DATA QUALITY
   ↓
PROBABILITY
   ↓
CONFIDENCE INTERVALS
   ↓
HYPOTHESIS TESTING
   ↓
EFFECT SIZES
   ↓
LOGISTIC REGRESSION
   ↓
BMI × AGE INTERACTION
   ↓
DUPLICATE-CONTROLLED ML
   ↓
CLASS-IMBALANCE EXPERIMENTS
   ↓
THRESHOLD ANALYSIS
   ↓
CALIBRATION
   ↓
MODEL INTERPRETATION
   ↓
STREAMLIT RESEARCH LAB
   ↓
REPORT + PRESENTATION
   ↓
REPRODUCIBLE GITHUB REPOSITORY
```

The application and report must remain aligned: **analytics functions return facts, service functions orchestrate them, and the UI presents them without inventing conclusions.**
