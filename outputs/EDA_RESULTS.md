# EDA results and handoff

Source: UCI dataset 891. SHA-256 is recorded in eda_manifest.json.
Target definition: 0 = neither prediabetes nor diabetes; 1 = prediabetes or diabetes (project-defined outcome). This follows the project contract and
imported UCI dictionary; the original binary preprocessing was not independently audited.

## Verification
- full_clean_v1: 253,680 rows, 22 columns, 0 missing cells.
- Repeated extra rows: 24,206 (9.5419%).
- unique_profile_v1: 229,474 complete-row distinct profiles.
- BMI IQR flags: 9,847; all retained.

## Duplicate sensitivity
Category 1 represents 13.9333% of full_clean_v1 and 15.2945% of
unique_profile_v1, a change of +1.3612 percentage points.
The full_vs_unique_sensitivity.csv file compares all categorical conditional
percentages and target-group means. Deduplication changes the represented sample;
it does not establish which respondents were duplicated.

## Main patterns
The full sample is imbalanced. Category 1 has higher typical BMI and more reported
poor physical-health days. Its share generally increases through age group 11,
then declines. Higher education/income categories generally have lower shares.
Reported high BP, high cholesterol, stroke, heart disease/attack and walking
difficulty have higher category-1 shares; activity has a lower share.
General-health category-1 shares increase from Excellent to Poor.
Consult labeled CSVs for exact estimates; these are unadjusted comparisons.

## Correlation interpretation
The matrix uses Spearman correlation for numeric counts, BMI, ordered categories
and the binary target. Age/Education/Income are ordered codes, not exact quantities.
The matrix is descriptive, not an importance ranking or adjusted causal model.

## Decisions and limitations
Raw data are preserved. Main EDA uses full_clean_v1; unique_profile_v1 is a labeled
sensitivity sample and the project-specified primary ML input. No balancing,
outlier removal, imputation or model fitting occurs here. Survey weights/design
are not applied; results describe the sample, not national prevalence. Reporting
error, unequal group sizes and confounding remain limitations. BMI extremes are
unverified. Complete-row deduplication does not guarantee distinct feature vectors
across train/test: the modeling team must check that separately.

## Reproduce
From the repository root, install requirements then run:
`python -m src.analysis.export_eda --csv data/raw/cdc_diabetes.csv`
See docs/MEMBER1_HANDOFF.md for contracts, changes, test scope and next steps.
