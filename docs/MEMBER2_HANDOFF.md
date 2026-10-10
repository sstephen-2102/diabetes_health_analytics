# Member 2: Probability and statistical inference handoff

This change implements the probability, confidence-interval, hypothesis-test, effect-size and
Bayes-verification contracts in `src/statistics/`, plus an exporter that writes the H1-H7 and Bayes
results for both dataset variants. It builds on Member 1's data layer. It does not implement the
Streamlit pages, the service-layer wiring, ANOVA, post-hoc analysis, FDR adjustment, or any modeling.

## Implemented contracts

| Function | Return and behavior |
|---|---|
| `calculate_probability(data, column, value)` | Float: marginal P(column = value). Empty data and unknown columns raise Member 1's validation errors. |
| `calculate_conditional_probability(data, target_column, target_value, condition_columns, condition_values)` | Dict with numerator, denominator, probability, Wilson confidence interval, target definition, dataset variant and sample size. A zero denominator returns `probability=None` and a warning, not an error. |
| `calculate_joint_probability(data, conditions)` | Dict with numerator, denominator (all rows), probability and metadata. Empty conditions raise `InvalidParameterError`. |
| `verify_bayes_theorem(data, target_column, target_value, condition_column, condition_value, tolerance=1e-9)` | Dict comparing the directly counted P(target given condition) with likelihood x prior / evidence, plus `matches`. Undefined cases (condition absent) return `matches=None` and a warning. |
| `calculate_risk_profile_probability(data, target_column, target_value, profile)` | Observed conditional probability for a feature profile, labelled as not a model prediction. |
| `proportion_confidence_interval(successes, total, confidence_level=0.95)` | Dict with estimate, bounds, method (Wilson), successes and total. |
| `mean_confidence_interval(values, confidence_level=0.95)` | t-interval with n, standard deviation and the count of missing values dropped. |
| `difference_in_means_ci(group_a, group_b, confidence_level=0.95)` | Welch-consistent interval for group_a minus group_b, with standard error and Satterthwaite degrees of freedom. |
| `run_chi_square_test(data, feature, target_column)` | Dict with contingency table, expected counts, chi-square, df, p-value, Cramer's V, n and an expected-count warning. No continuity correction. |
| `run_welch_t_test(data, numeric_feature, target_column, group_a, group_b, confidence_level=0.95)` | Dict with group sizes, means, standard deviations, mean difference (a minus b), t, df, p-value, confidence interval, Cohen's d and skewness warnings (absolute skewness above 1). |
| `calculate_effect_size(analysis_type, data, **kwargs)` | `"cohens_d"` (pooled SD), `"cramers_v"`, or `"odds_ratio"` (Woolf interval; Haldane-Anscombe correction with a warning if a cell is zero). |
| `run_hypothesis_suite(data, target_column='Diabetes_binary')` | One row per hypothesis H1-H7 with statistic, df, p-value, effect size, odds ratio where defined, and a plain restatement of the data. No interpretation text is generated. |
| `run_bayes_suite(data, target_column='Diabetes_binary', features=None, feature_value=1)` | One row per feature with prior, likelihood, evidence, Bayes right-hand side, direct value, match flag and lift over the prior. |
| `export_member2(csv_path, output_directory='outputs')` | Validates the CSV, runs both suites on both variants and writes two CSVs to `outputs/tables/`. |

`calculate_correlation_matrix` in `association.py` is Member 1's and is unchanged.

Existing signatures are retained. New modules: `hypothesis_suite.py`, `bayes_suite.py` (in `src/statistics/`) and `export_member2.py` (in `src/analysis/`).

## Usage

From the repository root:

```python
from src.common.config import FEATURE_COLUMNS, EXPECTED_COLUMNS
from src.data.loader import load_dataset
from src.data.preprocessing import prepare_analysis_data
from src.statistics.hypothesis_testing import run_chi_square_test, run_welch_t_test
from src.statistics.probability import calculate_conditional_probability

raw = load_dataset('data/raw/cdc_diabetes.csv')
full = prepare_analysis_data(raw, 'Diabetes_binary', FEATURE_COLUMNS)

chi = run_chi_square_test(full, 'HighBP', 'Diabetes_binary')
welch = run_welch_t_test(full, 'BMI', 'Diabetes_binary', 1, 0)   # category 1 minus category 0
cond = calculate_conditional_probability(full, 'Diabetes_binary', 1, ['HighBP'], [1])
```

The sample rule from the specification is followed: `full_clean_v1` is the primary sample for
inference; `unique_profile_v1` is the sensitivity check. Every result carries its `dataset_variant`.

### Notes for service-layer integration

- `get_probability_analysis(data, target_column, target_value, conditions)` can call
  `calculate_conditional_probability(data, target_column, target_value, list(conditions), list(conditions.values()))`.
- `get_statistical_test_result(data, test_type, feature, target_column, parameters)` can map
  `"chi_square"` to `run_chi_square_test` and `"welch"` to `run_welch_t_test`, taking `group_a` and
  `group_b` from `parameters`.
- Chi-square results contain DataFrames (contingency and expected counts), so they are not directly
  JSON-serializable; convert them before sending over an API boundary.

## Reproduce

```text
python -m pip install -r requirements.txt
python -m unittest discover tests
python -m src.analysis.export_member2 --csv data/raw/cdc_diabetes.csv
```

The CSV is not in the repository. Run `notebooks/CDC_DIABETES_EDA.ipynb` (its first cell downloads
and checks it) or place the verified file at `data/raw/cdc_diabetes.csv`.
Outputs: `outputs/tables/member2_hypothesis_results.csv` and `outputs/tables/member2_bayes_results.csv`.

## Results summary (full sample; unique-profile sample in the exported tables)

All seven tests reject independence at p < 0.001 in both samples, so effect sizes carry the comparison.

| Hypothesis | Feature | Effect size | Notes |
|---|---|---|---|
| H1 | HighBP | V = 0.26 | 24.45% vs 6.04% in category 1; risk ratio 4.05; odds ratio 5.04 |
| H2 | HighChol | V = 0.20 | 22.01% vs 7.98%; odds ratio 3.25 |
| H3 | BMI | d = 0.64 | means 31.94 vs 27.81, difference 4.14; BMI is right-skewed |
| H4 | PhysActivity | V = 0.12 | active 11.61% vs inactive 21.14%; odds ratio 0.49 |
| H5 | Smoker | V = 0.06 | 16.29% vs 12.06%; odds ratio 1.42 (lifetime 100+ cigarettes) |
| H6 | Age | V = 0.19 | 1.37% in band 1 to 21.85% in band 11, then lower |
| H7 | Income | V = 0.17 | not a smooth decline; most sensitive to duplicate handling |

On the unique-profile sample, all seven keep their direction and significance. Effects shrink most
for H5 (V 0.061 to 0.046), H7 (0.166 to 0.142) and H4 (0.118 to 0.100). 77% of the removed repeated
rows sit in the highest income band, which is why H7 moves most.

Bayes' theorem was verified for seven binary features on both samples: the directly counted
P(category 1 given feature) and the Bayes right-hand side agree in every case.

## Validation performed

- The full unit-test suite passes (69 tests at the time of writing, including Member 1's 25).
- Every Member 2 function has at least a normal, an invalid-input and an edge-case test.
- The Welch test, the difference-in-means interval and the mean interval agree with scipy; the
  odds ratio agrees with scipy's sample odds ratio; chi-square agrees with a hand calculation.
- The exporter was run on the real 253,680-row dataset for both variants. HighBP, HighChol,
  PhysActivity and Age results reproduce Member 1's published prevalence tables.

## Scope boundaries and unresolved points

- ANOVA, post-hoc analysis and multiple-comparison (FDR) adjustment remain stubs in
  `hypothesis_testing.py`. They are not required by H1-H7, were not taught in the course, and the
  team agreed they are out of scope. No multiplicity adjustment is applied; with all p < 0.001 it
  would not change any conclusion.
- All results are unadjusted associations from a self-reported survey without survey weights.
  They do not establish causation. Age is a likely confounder (see H8 for adjusted effects).
- With n = 253,680, p-values underflow to 0.0; report them as p < 0.001.
- BMI extremes remain unverified (carried over from Member 1). Skewness warnings are raised for H3.
- The positive class is category 1 (prediabetes or diabetes); do not describe it as "diabetic".
- The Streamlit pages and the service functions (`get_probability_analysis`,
  `get_statistical_test_result`) remain skeletons.
- Modeling, calibration and prediction are not part of this contribution.

## Review and collaboration

Use the team's `feature/member2-statistics` branch, then open a pull request against `main`.

Suggested PR title: `feat: Member 2 statistics (probability, CIs, chi-square, Welch, effect sizes, Bayes)`

AI assistance (Claude) was used to implement the contracts, prepare tests and draft documentation. The author ran the code on the full dataset, checked the results against Member 1's published tables, and wrote the hypothesis interpretations, with wording reviewed using AI.
