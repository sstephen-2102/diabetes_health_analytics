# Diabetes Health Analytics & Prediction Lab

University of San Diego M.S. Applied AI team project: statistics, inference, regression and machine
learning on the CDC BRFSS 2015 Diabetes Health Indicators dataset (UCI Machine Learning Repository,
dataset 891; 253,680 survey responses, 21 features), delivered as an interactive Streamlit app.

**Target.** `Diabetes_binary`: 0 = neither prediabetes nor diabetes; 1 = **prediabetes or diabetes**
(project-defined outcome). Category 1 is not a diagnosis; never describe it as "diabetic".

**Dataset variants.** Every result names the variant it uses.

| Variant | Rows | Used for |
|---|---|---|
| `full_clean_v1` | 253,680 (all rows) | Descriptive statistics and inference |
| `unique_profile_v1` | 229,474 (24,206 repeated rows removed) | Machine learning, with a grouped split so no profile is in both training and test sets |

Architecture: Streamlit UI → service layer (`app/services.py`) → analytics (`src/`) → data layer.
Pages never compute statistics themselves.

## Quick start

```text
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The raw CSV and the fitted model files are not in Git. On first launch the app downloads the verified
UCI dataset to `data/raw/cdc_diabetes.csv` and retrains any missing models (a few minutes). Exported
tables and figures in `outputs/` are committed, so every page works straight after the download.

Always launch `streamlit_app.py` from the repository root; `streamlit run app/app.py` breaks imports.
The dev container (`.devcontainer/`) starts the app automatically in GitHub Codespaces.

## App pages

| Section | Page | Owner |
|---|---|---|
| Data | Executive Dashboard: key figures, selected findings, research questions, methodology notes | Member 1 |
| Data | Data Explorer: schema, category mappings, descriptive statistics, data quality | Member 1 |
| Data | Exploratory Analysis: per-variable views, binary indicators, correlations, full vs unique sample | Member 1 |
| Statistics | Probability Explorer: marginal, joint and conditional probability; Bayes' theorem check | Member 2 |
| Statistics | Statistical Testing Lab: H1-H7, chi-square and Welch tests, effect sizes, confidence intervals | Member 2 |
| Statistics | Regression Lab: baseline and BMI × Age interaction logistic regression, odds ratios | Member 3 |
| Modeling | Machine Learning Lab, Imbalance & Threshold Lab, Calibration, Interpretation | Member 3 |
| Modeling | Prediction Explorer: model-estimated probability for a feature profile (not a diagnosis) | Member 3 |

## Reproduce the outputs

Regenerate every table, figure and model in `outputs/` (downloads the CSV if it is missing):

```text
python -m src.analysis.pipeline --csv data/raw/cdc_diabetes.csv
```

Or run one step at a time, in this order:

```text
python -m src.analysis.export_eda --csv data/raw/cdc_diabetes.csv
python -m src.analysis.export_member2 --csv data/raw/cdc_diabetes.csv
python -m src.analysis.export_regression --csv data/raw/cdc_diabetes.csv
python -m src.analysis.export_modeling --csv data/raw/cdc_diabetes.csv
python -m src.analysis.export_duplicate_sensitivity --csv data/raw/cdc_diabetes.csv
```

`notebooks/CDC_DIABETES_EDA.ipynb` walks through the exploratory analysis and downloads the dataset in
its first cell. Every export validates the CSV schema first; all except the statistics export also check it
against the reference dimensions and counts.

## Tests

```text
python -m pytest tests/ -q
```

Tests use small synthetic CSVs, so they need neither the real dataset nor a network connection.

## Documentation

| File | Contents |
|---|---|
| `docs/PROJECT_SPECIFICATION.md` | Authoritative requirements: research questions, hypotheses, methods, definition of done |
| `docs/DATA_DICTIONARY.md` | Every variable, its coding and labels |
| `docs/MEMBER1_HANDOFF.md` | Data layer and EDA: contracts, decisions, test scope |
| `docs/MEMBER2_HANDOFF.md` | Probability and inference: contracts, H1-H7 results, scope decisions |
| `docs/MEMBER3_HANDOFF.md` | Regression and modeling: contracts, design decisions, test scope |
| `outputs/EDA_RESULTS.md` | EDA results summary |
| `docs/MODELING_REPORT.md` | Regression, classification, imbalance, threshold, calibration and interpretation results |
| `docs/FINAL_REPORT.md` | Final report scaffold: every number, table, figure and caption, with a traceability table |
| `docs/report_template.md` | Original section-by-section guidance for the final report |
| `docs/references.md` | APA 7 references and which report section cites each |

## Scope and limitations

- Unweighted sample statistics: BRFSS survey weights and design are not applied, so results describe
  this dataset, not U.S. population prevalence.
- Cross-sectional, self-reported data: associations are not causal.
- No multiple-comparison adjustment is applied to H1-H7, and ANOVA and post-hoc tests are not
  implemented (team decision; see `docs/MEMBER2_HANDOFF.md`).
- Model performance on this dataset does not establish clinical utility; predictions are statistical
  estimates, not medical advice.
