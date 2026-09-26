# Diabetes Health Analytics & Prediction Lab
## Complete 20-Day Implementation Plan

**Specification:** `docs/PROJECT_SPECIFICATION.md`  
**Duration:** 20 days  
**Team:** 3 members  
**Primary stack:** Python, pandas, NumPy, SciPy/statsmodels, scikit-learn, Plotly, Streamlit, kagglehub

---

## 1. Implementation Philosophy

Build the system in layers. Do not begin by putting statistical calculations into Streamlit pages.

Required dependency direction:

```text
APP → SERVICE → ANALYTICS → DATA
```

Analytics functions must be reusable by Streamlit, tests, notebooks, and report-generation workflows.

---

## 2. Build Sequence

1. Repository setup
2. Dataset verification
3. Data dictionary
4. Configuration and schemas
5. Data layer
6. Descriptive statistics
7. Probability
8. Inferential statistics
9. Regression
10. Machine learning
11. Class imbalance
12. Thresholds
13. Calibration
14. Interpretation
15. Service layer
16. Streamlit UI
17. Integration
18. Testing
19. Report
20. Presentation and final audit

Do not skip the dataset-verification gate.

---

## 3. Day 1 — Research Design and Repository Alignment

### All members

Confirm project title, application name, research objective, research questions, hypotheses, dataset, scope, ownership, and documentation structure.

### Member 1

Review dataset source and download process.

### Member 2

Map every research question to a statistical method and required output.

### Member 3

Map predictive research questions to models, metrics, imbalance experiments, threshold analysis, and calibration.

### Deliverables

- approved specification;
- approved implementation plan;
- repository structure;
- initial GitHub issues.

---

## 4. Day 2 — Dataset Verification Gate

### Member 1

Download using `kagglehub` and inspect the actual CSV.

Verify exact filename, dimensions, columns, target, target coding, class counts, missing values, duplicates, data types, binary values, Age, Sex, Education, Income, and unexpected values.

### Member 2

Confirm the verified schema supports planned probability and inferential analyses.

### Member 3

Confirm candidate ML features, target encoding, and preprocessing requirements.

### Deliverables

- raw dataset under `data/raw/`;
- verification notes;
- initial quality report;
- `docs/DATA_DICTIONARY.md`.

### Gate

No full analysis implementation until schema and target coding are verified.

---

## 5. Day 3 — Configuration and Package Foundation

Implement:

- package `__init__` files;
- `config.py`;
- `schemas.py`;
- `exceptions.py`;
- dataset configuration;
- model configuration;
- variable metadata;
- `.gitignore`;
- requirements.

Confirm the project imports cleanly from a fresh environment.

---

## 6. Day 4 — Data Layer

Member 1 implements:

```python
download_kaggle_dataset()
load_dataset()
validate_dataset()
generate_data_quality_report()
prepare_analysis_data()
```

Validation must detect missing files, malformed CSVs, missing columns, unexpected target values, invalid categories, and empty datasets.

**Deliverable:** reproducible raw-to-analysis-ready data pipeline.

---

## 7. Day 5 — Descriptive Statistics and EDA

Implement numeric and categorical summaries, target comparisons, and prevalence tables.

Generate target distribution, demographic summaries, health/lifestyle summaries, group comparisons, and initial visualizations.

**Deliverable:** baseline EDA tables and figures.

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

Every result should expose numerator, denominator, estimate, conditions, and relevant sample size.

**Deliverable:** verified probability examples suitable for report and UI.

---

## 9. Day 7 — Confidence Intervals

Implement proportion, mean, and difference-in-means confidence intervals. Test against simple known cases and edge cases.

**Deliverable:** tested CI module.

---

## 10. Day 8 — Hypothesis Testing

Implement chi-square, Welch t-test, ANOVA, post-hoc analysis, effect sizes, and multiple-comparison adjustment.

Report test statistic, df where applicable, p-value, effect size, and sample size.

**Deliverable:** complete inferential-statistics core.

---

## 11. Day 9 — Association and Statistical Integration

Implement correlation and unified statistical result structures. Produce report-ready tables.

Review whether assumptions and test choices are documented.

**Deliverable:** all planned core statistical outputs can be generated outside Streamlit.

---

## 12. Day 10 — Logistic Regression

Member 3 implements:

```python
fit_logistic_regression()
extract_odds_ratios()
compare_regression_models()
```

Build baseline and BMI × Age interaction models. Report coefficients, odds ratios, confidence intervals, p-values, and model comparison statistics.

**Deliverable:** reproducible regression results.

---

## 13. Day 11 — Classification Models

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

Use stratification and a fixed seed. Keep the test set untouched.

**Deliverable:** three baseline classifiers and common metrics.

---

## 14. Day 12 — Class Imbalance

Implement:

```python
inspect_class_distribution()
balance_training_data()
train_class_weighted_model()
```

Compare original, class-weighted, and balanced-training approaches.

Only training data may be balanced.

**Deliverable:** comparable imbalance experiment table.

---

## 15. Day 13 — Threshold and Calibration

Implement:

```python
evaluate_thresholds()
generate_threshold_curve()
calculate_calibration_metrics()
generate_calibration_data()
```

Evaluate threshold trade-offs and calibration/Brier score.

**Deliverable:** threshold and calibration outputs.

---

## 16. Day 14 — Model Interpretation

Implement permutation feature importance and odds-ratio extraction. Generate interpretation-ready tables and figures.

**Deliverable:** transparent model interpretation outputs.

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

**Deliverable:** Streamlit can consume stable service contracts.

---

## 18. Day 16 — Streamlit Core UI

Build application shell, navigation, dashboard, Data Explorer, EDA, and shared components.

Shared components should include metric cards, result tables, confusion matrices, probability results, and statistical results.

**Deliverable:** core UI connected to real analytics.

---

## 19. Day 17 — Advanced UI

Build Probability Explorer, Statistical Testing Lab, Regression Lab, Machine Learning Lab, Imbalance & Threshold Lab, Calibration, Interpretation, and Prediction Explorer.

Add user-input validation and friendly error handling.

**Deliverable:** complete research laboratory.

---

## 20. Day 18 — Integration and QA

Merge team branches. Test all modules and pages, including invalid inputs, empty groups, missing columns, failed models, invalid thresholds, and incomplete prediction profiles.

Run the complete test suite.

**Deliverable:** release candidate.

---

## 21. Day 19 — Report and Presentation

Complete methodology, results, discussion, limitations, figures, tables, references, and presentation.

Every major reported value must be traceable to code/output.

**Deliverable:** draft final report and 15–18 slide presentation.

---

## 22. Day 20 — Final Audit and Rehearsal

Perform a clean-environment install and run. Launch Streamlit, run tests, verify outputs, inspect documentation, check GitHub, and rehearse the presentation.

No unresolved critical defects should remain.

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

Suggested commit prefixes: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`.

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
- DATA-006 Preprocessing

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

### Modeling

- ML-001 Data split
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
- QA-003 Clean environment test
- QA-004 Documentation review
- QA-005 Final reproducibility audit

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
- results are reproducible.

---

## 27. UI Definition of Done

A page is complete when it loads successfully, uses the service/analytics layer, validates inputs, handles expected errors cleanly, labels charts and statistics, avoids causal/diagnostic claims, and works with the verified dataset.

---

## 28. Statistical Result Definition of Done

Every major result should state or expose what was analyzed, sample size, estimate, uncertainty, test statistic where applicable, p-value where applicable, effect size where applicable, and limitations.

---

## 29. ML Result Definition of Done

Every model experiment should record model type, feature set, split, seed, training strategy, threshold, accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, Brier score where applicable, and confusion matrix.

---

## 30. Report-to-Code Traceability

| Report section | Primary module | Output |
|---|---|---|
| Dataset | `src/data/*` | quality tables |
| EDA | `descriptive.py` | figures/tables |
| Probability | `probability.py` | probability tables |
| Confidence intervals | `confidence_intervals.py` | CI tables |
| Hypothesis tests | `hypothesis_testing.py` | test tables |
| Regression | `regression.py` | odds-ratio tables |
| ML | `classification.py` | metrics |
| Imbalance | `imbalance.py` | experiment tables |
| Threshold | `thresholds.py` | threshold data/figures |
| Calibration | `calibration.py` | calibration data |
| Interpretation | `interpretation.py` | importance tables |

---

## 31. Output Convention

```text
outputs/
├── figures/
│   ├── target_distribution.*
│   ├── bmi_by_diabetes.*
│   ├── correlation_matrix.*
│   ├── roc_curve.*
│   ├── precision_recall_curve.*
│   ├── threshold_curve.*
│   └── calibration_curve.*
├── tables/
│   ├── descriptive_statistics.*
│   ├── prevalence.*
│   ├── probability_results.*
│   ├── hypothesis_tests.*
│   ├── regression_odds_ratios.*
│   ├── model_comparison.*
│   └── calibration_metrics.*
└── models/
    ├── model.pkl
    ├── metadata.json
    └── metrics.json
```

---

## 32. Testing Matrix

| Module | Normal case | Invalid case | Edge case |
|---|---|---|---|
| Loader | valid CSV | missing file | empty CSV |
| Validation | valid schema | missing column | unexpected value |
| Probability | valid event | missing condition | zero denominator |
| CI | valid sample | invalid confidence | tiny/degenerate sample |
| Chi-square | valid groups | missing feature | empty category |
| t-test | two groups | invalid group | tiny group |
| ANOVA | 3+ groups | invalid grouping | empty group |
| Regression | valid features | missing feature | constant predictor |
| Classification | valid split | invalid target | single class |
| Threshold | valid 0–1 | >1 or <0 | duplicate thresholds |
| Calibration | valid probabilities | invalid probability | sparse bin |
| Prediction | complete profile | missing input | invalid category |

---

## 33. Integration Checklist

- [ ] Dataset path configurable
- [ ] Kaggle dataset ID documented
- [ ] No personal absolute paths
- [ ] No credentials/API keys committed
- [ ] Raw dataset preserved
- [ ] Target verified as `Diabetes_binary`
- [ ] Data dictionary matches actual file
- [ ] Metadata matches verified coding
- [ ] Analytics independent of Streamlit
- [ ] UI uses service layer
- [ ] Tests pass
- [ ] Streamlit starts
- [ ] All pages render
- [ ] Model artifacts load
- [ ] Report values match outputs
- [ ] Presentation matches final results

---

## 34. Clean-Environment Test

Create a fresh virtual environment and install from `requirements.txt`.

Typical macOS/Linux commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Run the complete test suite as well.

---

## 35. Reproducibility Audit

A teammate who did not implement a component should attempt to reproduce the main results using only the repository, documented dependencies, dataset acquisition instructions, and project documentation.

Record any undocumented manual steps and eliminate them before final submission.

---

## 36. Final Review Questions

### Research

- Do research questions map to analyses?
- Are H0/H1 explicit?
- Are conclusions supported by results?

### Statistics

- Are assumptions considered?
- Are effect sizes reported?
- Are confidence intervals reported?
- Are multiple comparisons addressed where appropriate?

### ML

- Was test leakage avoided?
- Was the test set kept unchanged?
- Was balancing restricted to training data?
- Are multiple metrics reported?
- Is calibration evaluated?

### Software

- Is the architecture modular?
- Are UI and analytics separated?
- Are expected errors handled cleanly?
- Are core functions tested?

### Reproducibility

- Can another teammate run the project?
- Is the dataset source documented?
- Are seeds and model parameters documented?
- Are experiments reproducible?

### Responsible use

- Are causal claims avoided?
- Are diagnosis claims avoided?
- Are limitations visible?
- Is model output clearly educational/research-oriented?

---

## 37. Priority if Time Is Lost

### Priority 1 — Must have

Dataset verification, data quality, descriptive statistics, probability, confidence intervals, hypothesis testing, logistic regression, three baseline ML models, core metrics, core Streamlit pages, report.

### Priority 2 — Strongly recommended

Class imbalance, threshold analysis, calibration, permutation importance, BMI × Age interaction.

### Priority 3 — Optional

Additional models, extensive hyperparameter search, advanced experiment tracking, additional UI polish.

Do not sacrifice reproducibility, testing, or core statistics for optional features.

---

## 38. Final System Flow

```text
Research Question
       ↓
Dataset Acquisition
       ↓
Data Quality & Validation
       ↓
Descriptive Analysis
       ↓
Probability
       ↓
Inferential Statistics
       ↓
Association / Regression
       ↓
Prediction
       ↓
Class Imbalance
       ↓
Threshold Analysis
       ↓
Calibration
       ↓
Interpretation
       ↓
Streamlit Research Lab
       ↓
Report + Presentation
```

The final deliverable should demonstrate both statistical/research competence and AI/software-engineering competence.
