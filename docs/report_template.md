# Final report template

Skeleton for the final report (spec section 26). Copy it into the team document,
replace each *Write:* prompt with prose, and delete the guidance lines before
submission.

Each section lists its owner, the research questions it answers (spec section
4), where its numbers come from, and the tables or figures it needs.

## Conventions (apply everywhere)

- **Target wording.** Call class 1 the "prediabetes-or-diabetes outcome" or the
  "`Diabetes_binary` outcome". Never write "diabetes diagnosis" or "the model
  diagnoses".
- **Association, not causation.** Use "associated with", "higher odds", "predicts".
  Never write "causes", "increases risk" or "protective".
- **Dataset variant.** Every table and figure caption names its variant
  (`full_clean_v1` or `unique_profile_v1`), its n, and its method.
- **Traceability.** Every reported number must come from a file in `outputs/`.
  Fill in the traceability table at the end before submission.
- **Effect sizes with p-values.** Report a confidence interval or effect size
  next to every p-value. With n above 200,000, almost every p-value is tiny.
- **Citations.** APA 7; the reference list and usage map are in
  `docs/references.md`.
- **Numbers.** Three decimals for metrics, two for odds ratios (three for BMI),
  and thousands separators for counts.

## Ownership

| Section | Owner | Source material |
|---|---|---|
| 1. Introduction and research objective | Shared | Spec sections 2–5 |
| 2–5. Dataset, target, data quality, descriptive statistics | Member 1 | `outputs/EDA_RESULTS.md`, `docs/DATA_DICTIONARY.md`, `docs/MEMBER1_HANDOFF.md` |
| 6–9. Probability, confidence intervals, hypothesis tests, effect sizes | Member 2 | Statistics exports (not produced yet) |
| 10–16. Regression, ML, imbalance, thresholds, calibration, interpretation | Member 3 | `docs/MODELING_REPORT.md` |
| 17. Limitations | Shared | Spec section 27, plus each owner's section notes |
| 18. Conclusion | Shared | One answer per research question |
| 19. References | Shared | `docs/references.md` |

---

## Title page

- Title: Diabetes Health Analytics & Prediction Lab
- Course, team members and roles, date
- Repository URL and the commit hash the report describes

## Abstract (optional, about 200 words)

*Write:* objective, data (BRFSS 2015 via UCI/Kaggle, 253,680 records), methods
(statistics, logistic regression, three classifiers), two or three headline
results with uncertainty, and the main limitation.

## 1. Introduction and research objective

**Owner:** shared. **Answers:** frames RQ1–RQ13.

- Public-health context for prediabetes and diabetes indicators (cite
  Xie et al., 2019, for prior ML work on this survey).
- Primary objective, quoted or paraphrased from spec section 3.
- The research questions (RQ1–RQ13) and hypotheses (H1–H11), as a table or list.
- Report roadmap.

*Write:*

## 2. Dataset, source and verification

**Owner:** Member 1. **Sources:** `outputs/eda_manifest.json` (SHA-256,
reference checks), `outputs/tables/full_clean_v1_validation.json`,
`docs/DATA_DICTIONARY.md`.

- Provenance: CDC BRFSS 2015, then the Kaggle cleaned release (Teboul, 2021),
  then UCI dataset 891 (*CDC Diabetes Health Indicators*, 2017).
- Verification gate: 253,680 rows, 22 columns, 0 missing cells, SHA-256 recorded.
- Variable dictionary summary (21 features: binary, ordinal, counts, BMI).

**Table:** variable summary (name, type, coding).

*Write:*

## 3. Target definition and terminology

**Owner:** Member 1. **Sources:** spec section 2 (critical terminology rule),
`docs/DATA_DICTIONARY.md`.

- Class 0 is neither condition; class 1 is prediabetes or diabetes.
- Why "diagnosis" language is avoided; the original binary preprocessing was not
  independently audited (`outputs/EDA_RESULTS.md`).

*Write:*

## 4. Data quality and duplicate analysis

**Owner:** Member 1 (duplicates) with Member 3 (split integrity).
**Answers:** RQ13, H11.
**Sources:**
- `outputs/tables/full_clean_v1_data_quality.json`
- `outputs/tables/full_vs_unique_sensitivity.csv`
- `outputs/EDA_RESULTS.md`
- the `leakage_check` entry in `outputs/modeling_manifest.json`

- 24,206 repeated rows (9.54%), 229,474 distinct profiles; no respondent ID, so
  duplicates are not called errors.
- Positive share is 13.93% in `full_clean_v1` vs 15.29% in `unique_profile_v1`.
- 9,847 BMI values flagged by the IQR rule, all retained.
- Dataset-variant policy: `full_clean_v1` for statistics and regression;
  `unique_profile_v1` for ML, with a grouped split and 0 shared profiles.

**Open item:** H11 asks whether duplicate handling changes model evaluation. The
optional full-data ML comparison (spec section 7, item 6) has not been run. Either
run it or state in sections 4 and 17 that only split integrity was tested.

*Write:*

## 5. Descriptive statistics

**Owner:** Member 1. **Answers:** RQ1, RQ2.
**Sources:**
- `outputs/tables/full_clean_v1_numeric_summary.csv`
- `outputs/tables/full_clean_v1_prevalence_*.csv`
- `outputs/tables/full_clean_v1_spearman_correlation.csv`

**Figures** (all in `outputs/figures/`):
- `full_clean_v1_target_distribution.png`
- `full_clean_v1_bmi_by_target.png`
- `full_clean_v1_prevalence_{HighBP,HighChol,GenHlth,Age,Income}.png`
- `full_clean_v1_spearman_correlation.png`

- These comparisons are unadjusted; state that the correlation matrix is not an
  importance ranking.

*Write:*

## 6. Probability and Bayes analysis

**Owner:** Member 2. **Answers:** RQ3, RQ4. **Sources:** statistics exports (not
produced yet).

- Marginal, joint and conditional probabilities for the selected conditions
  (HighBP, HighChol, PhysActivity, demographics).
- Bayes' theorem check: direct conditional probability vs the Bayes formula.

*Write:*

## 7. Confidence intervals

**Owner:** Member 2. **Answers:** RQ5.

- Proportion intervals (name the method: Wilson or Agresti–Coull), mean
  intervals, and Welch mean-difference intervals.

*Write:*

## 8. Hypothesis testing

**Owner:** Member 2. **Answers:** RQ6, H1–H7.

- Chi-square tests (H1, H2, H4–H7), Welch t-test (H3), ANOVA with post-hoc tests.
- Assumption checks.

**Table:** hypothesis, test, statistic, df, p, adjusted p, decision.

*Write:*

## 9. Effect sizes and multiple comparisons

**Owner:** Member 2. **Answers:** RQ7.

- Cramér's V and Cohen's d (or the measures actually implemented), with
  interpretation thresholds.
- Benjamini–Hochberg adjustment across the test family.

*Write:*

## 10. Logistic regression

**Owner:** Member 3. **Answers:** RQ8, H8. **Outline:** `docs/MODELING_REPORT.md`
section 10.

**Figure:** `outputs/figures/full_clean_v1_regression_odds_ratios.png`.
**Table:** odds ratios with 95% intervals.

*Write:*

## 11. BMI × Age interaction

**Owner:** Member 3. **Answers:** RQ9, H9. **Outline:** `docs/MODELING_REPORT.md`
section 11.

**Figure:** `outputs/figures/full_clean_v1_regression_bmi_odds_by_age.png`.
**Table:** model comparison (log-likelihood, AIC, BIC, pseudo-R², likelihood-ratio test).

*Write:*

## 12. Machine-learning models

**Owner:** Member 3. **Answers:** RQ10. **Outline:** `docs/MODELING_REPORT.md`
section 12.

**Figures:** `outputs/figures/unique_profile_v1_roc_curves.png` and
`unique_profile_v1_precision_recall_curves.png`.
**Table:** `outputs/tables/model_comparison.csv`.

*Write:*

## 13. Class-imbalance experiments

**Owner:** Member 3. **Answers:** RQ11, H10. **Outline:** `docs/MODELING_REPORT.md`
section 13.

**Table:** `outputs/tables/imbalance_experiments.csv`.

*Write:*

## 14. Threshold analysis

**Owner:** Member 3. **Answers:** RQ12 (thresholds). **Outline:**
`docs/MODELING_REPORT.md` section 14.

**Figure:** `outputs/figures/unique_profile_v1_threshold_tradeoffs.png`.

*Write:*

## 15. Calibration

**Owner:** Member 3. **Answers:** RQ12 (calibration). **Outline:**
`docs/MODELING_REPORT.md` section 15.

**Figure:** `outputs/figures/unique_profile_v1_calibration_curves.png`.

*Write:*

## 16. Model interpretation

**Owner:** Member 3. **Outline:** `docs/MODELING_REPORT.md` section 16.

**Figure:** `outputs/figures/unique_profile_v1_feature_importance.png`.

*Write:*

## 17. Limitations

**Owner:** shared. All ten items in spec section 27 must appear:

1. The Kaggle/UCI dataset is derived from BRFSS 2015, not the raw complex-survey
   environment.
2. Survey weights and design are not reproduced; results describe the sample.
3. The positive class combines prediabetes and diabetes.
4. The data are cross-sectional and observational; no causal claims.
5. Exact duplicate rows exist and there is no respondent identifier.
6. Deduplication for ML is an analytical choice, not proof of error.
7. Predictive performance does not establish clinical utility or external validity.
8. Threshold selection depends on context.
9. Calibration and discrimination can change across populations.
10. Results should not be generalized to current populations without validation.

Add the section-specific limitations from each owner (for modeling, see
`docs/MODELING_REPORT.md` section 17).

*Write:*

## 18. Conclusion

**Owner:** shared. Answer each research question in one or two sentences with
its key number, then note future work (validation set or cross-validation,
survey weights, external validation).

| RQ | Short answer | Key evidence (section) |
|---|---|---|
| RQ1–RQ2 | | 5 |
| RQ3–RQ4 | | 6 |
| RQ5 | | 7 |
| RQ6–RQ7 | | 8, 9 |
| RQ8 | | 10 |
| RQ9 | | 11 |
| RQ10 | | 12 |
| RQ11 | | 13 |
| RQ12 | | 14, 15 |
| RQ13 | | 4 |

*Write:*

## 19. References

Paste the list from `docs/references.md`, keeping only entries cited in the text.

## Appendix A. Reproducibility

```text
python -m pip install -r requirements.txt
python -m src.analysis.export_eda --csv data/raw/cdc_diabetes.csv
python -m src.analysis.export_regression
python -m src.analysis.export_modeling
python -m pytest tests/ -q
streamlit run streamlit_app.py
```

Record the commit hash, the dataset SHA-256 from `outputs/eda_manifest.json`,
and the software versions (see `docs/references.md`).

## Appendix B. Streamlit application

Screenshots of the main pages, each with one sentence on what it shows. Remind
readers that the Prediction Explorer gives a model estimate, not a diagnosis.

## Traceability table (fill before submission)

| Reported value | Section | Source file | Column/key |
|---|---|---|---|
| e.g. HighBP OR 2.13 (2.07–2.20) | 10 | `outputs/tables/full_clean_v1_regression_baseline_odds_ratios.csv` | `odds_ratio`, `ci_lower`, `ci_upper` |
| | | | |

## Final checks (from the day-20 audit)

- [ ] No "diagnosis" language except disclaimers; target wording consistent.
- [ ] Every figure and table caption names its dataset variant.
- [ ] Every p-value has an interval or effect size next to it.
- [ ] Duplicate handling and leakage control described (sections 4, 12).
- [ ] Test set never used for tuning, balancing or threshold choice.
- [ ] All ten spec limitations present.
- [ ] Every major number appears in the traceability table.
- [ ] References in APA 7, all cited in the text, none missing.
