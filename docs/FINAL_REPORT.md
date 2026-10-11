# Diabetes Health Analytics & Prediction Lab: final report (scaffold)

> **How to use this file.** Every number, table, figure and caption below comes from a file in `outputs/`
> (see the traceability table at the end). Each section lists bullet-point findings; replace each
> *Write:* line with prose, keep the tables and figures you need, and delete these guidance blocks and
> the "Sources" lines before submission. Conventions from `docs/report_template.md` apply throughout:
> say "prediabetes-or-diabetes outcome" (never "diagnosis"), "associated with" (never "causes",
> "increases risk" or "protective"), and name the dataset variant in every caption.

---

## Title page

- **Title:** Diabetes Health Analytics & Prediction Lab
- **Course:** AAI-500 Probability and Statistics for AI
- **Team:** Arunava Banerjee (data and EDA); Binish Hamdani (probability and inference);
  Sanjay Varghese Stephen (regression and machine learning)
- **Date:** 19 October 2026
- **Repository:** https://github.com/sstephen-2102/diabetes_health_analytics, commit [FINAL COMMIT HASH]
- **Live application:** https://diabeteshealthanalytics-arunava-binish-sanjay.streamlit.app

## Abstract (optional, about 200 words)

Findings to cover:

- **Objective:** relate demographic, health, behavioral and socioeconomic indicators to the
  `Diabetes_binary` outcome (prediabetes or diabetes) and test how well they predict it.
- **Data:** CDC BRFSS 2015 via UCI dataset 891; 253,680 responses, 21 features; 13.93% in category 1.
- **Methods:** descriptive statistics, probability and Bayes, Wilson and t intervals, chi-square and
  Welch tests with effect sizes, logistic regression with a BMI × Age interaction, three classifiers,
  imbalance, threshold and calibration analysis.
- **Headline results:**
  - General health shows the strongest unadjusted association (Spearman ρ = 0.288), and high blood
    pressure has adjusted odds ratio 2.13 (95% CI 2.07–2.20).
  - Gradient boosting reaches ROC AUC 0.819 and PR AUC 0.441 on held-out data, against 0.810 and
    0.414 for logistic regression.
  - At the default 0.5 threshold, recall is only 0.15–0.18.
- **Main limitation:** unweighted, cross-sectional, self-reported data, so results are associations
  in this sample, not population prevalence or causal effects.

*Write:*

---

## 1. Introduction and research objective

**Owner:** shared. **Answers:** frames RQ1–RQ13. **Sources:** spec sections 2–5; Xie et al. (2019).

Points to cover:

- Public-health context: prediabetes and diabetes are common, and survey indicators are cheap to
  collect; Xie et al. (2019) built prediction models on the same BRFSS survey.
- **Primary objective** (spec section 3): investigate relationships between demographic, health,
  behavioral and socioeconomic indicators and the `Diabetes_binary` outcome, and evaluate whether those
  indicators can predict it using interpretable statistical and machine-learning methods.
- This is an educational research project, not a clinical diagnostic tool.

**Table 1.** Research questions and where they are answered.

| RQ | Question | Section |
|---|---|---|
| RQ1 | What characteristics describe the analytical sample? | 5 |
| RQ2 | How is `Diabetes_binary` distributed? | 5 |
| RQ3 | How does the probability of category 1 change under selected conditions? | 6 |
| RQ4 | Can Bayes' theorem be demonstrated and verified from the observed data? | 6 |
| RQ5 | What confidence intervals describe selected proportions, means and differences? | 7 |
| RQ6 | Which variables show statistically detectable associations with the outcome? | 8 |
| RQ7 | How large are the effects, and how should they be read alongside p-values? | 9 |
| RQ8 | Which variables remain associated after simultaneous adjustment? | 10 |
| RQ9 | Does the BMI–outcome relationship vary by age? | 11 |
| RQ10 | How do logistic regression, random forest and gradient boosting compare? | 12 |
| RQ11 | How does class imbalance affect model behavior? | 13 |
| RQ12 | How do thresholds affect precision, recall, specificity and F1; are probabilities calibrated? | 14, 15 |
| RQ13 | How does exact-duplicate handling affect train/test integrity and evaluation? | 4 |

**Table 2.** Hypotheses (spec section 5). H1–H7 are tested in section 8, H8 in 10, H9 in 11, H10 in
13 and H11 in 4.

| ID | Alternative hypothesis |
|---|---|
| H1 | The `Diabetes_binary` outcome is associated with HighBP. |
| H2 | The `Diabetes_binary` outcome is associated with HighChol. |
| H3 | Mean BMI differs between target groups. |
| H4 | The `Diabetes_binary` outcome is associated with physical activity. |
| H5 | The `Diabetes_binary` outcome is associated with smoking. |
| H6 | The target outcome differs across age categories. |
| H7 | The target outcome differs across income categories. |
| H8 | Selected predictors remain associated with the target after adjustment. |
| H9 | The BMI–target association varies with age. |
| H10 | Class-imbalance strategy changes predictive behavior. |
| H11 | Exact-duplicate handling changes some model evaluation results. |

*Write:* (context, objective, report roadmap)

---

## 2. Dataset, source and verification

**Owner:** Member 1. **Sources:** `outputs/eda_manifest.json`, `outputs/tables/full_clean_v1_validation.json`,
`outputs/tables/full_clean_v1_data_quality.json`, `docs/DATA_DICTIONARY.md`.

Findings:

- **Provenance:** CDC BRFSS 2015 (CDC, 2016a), then the cleaned Kaggle release (Teboul, 2021), then UCI
  dataset 891 (*CDC Diabetes Health Indicators*, 2017). The app and pipeline download the UCI copy.
- **Verification gate passed:**
  - 253,680 rows, 22 columns (21 features plus the target), 0 missing cells;
  - every reference check passes: rows, columns, missing cells, duplicate rows and target counts.
- **Version:** SHA-256 `0f8ff13c…a79774` for the CSV behind the reported results (see Appendix A).
- **Feature types:** 14 binary, 4 ordinal (GenHlth 1–5, Age 13 bands, Education 6 levels, Income 8
  levels), 2 day counts (MentHlth, PhysHlth, 0–30) and BMI (12–98).

**Table 3.** Variable summary (`full_clean_v1`, n = 253,680). Full coding is in `docs/DATA_DICTIONARY.md`.

| Group | Variables | Coding |
|---|---|---|
| Target | Diabetes_binary | 0 = neither prediabetes nor diabetes; 1 = prediabetes or diabetes |
| Cardiometabolic | HighBP, HighChol, CholCheck, Stroke, HeartDiseaseorAttack | 0 = no, 1 = yes |
| Behavior | Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump | 0 = no, 1 = yes |
| Access to care | AnyHealthcare, NoDocbcCost | 0 = no, 1 = yes |
| Health status | GenHlth (1 = excellent … 5 = poor), MentHlth, PhysHlth (days of 30), DiffWalk (0/1) | ordinal, count, binary |
| Body size | BMI | recorded value, 12–98 |
| Demographics | Sex (0 = female, 1 = male), Age (13 five-year bands), Education (1–6), Income (1–8) | binary, ordinal |

*Write:*

---

## 3. Target definition and terminology

**Owner:** Member 1. **Sources:** spec section 2, `docs/DATA_DICTIONARY.md`, `outputs/EDA_RESULTS.md`.

Findings:

- Category 0 = neither prediabetes nor diabetes; category 1 = prediabetes **or** diabetes, a
  project-defined outcome.
- Category 1 is self-reported survey status, not a clinical diagnosis; the original preprocessing that
  produced the binary target was not independently audited.
- Terms used throughout: "category 1", "prediabetes-or-diabetes outcome" or "`Diabetes_binary` outcome".

*Write:*

---

## 4. Data quality and duplicate analysis

**Owner:** Member 1 (duplicates) with Member 3 (split integrity). **Answers:** RQ13, H11.
**Sources:** `outputs/tables/full_clean_v1_data_quality.json`, `outputs/tables/full_vs_unique_sensitivity.csv`,
`outputs/modeling_manifest.json` (`leakage_check`), `outputs/tables/duplicate_sensitivity.csv`.

Findings:

- **Repeated rows:**
  - 24,206 rows (9.54%) repeat an earlier row;
  - 35,575 rows belong to a repeated pattern;
  - there are 229,474 distinct profiles.

  There is no respondent identifier, so repeats cannot be confirmed as the same person or called errors.
- **Effect on the sample:** the category 1 share is 13.93% in `full_clean_v1` and 15.29% in
  `unique_profile_v1` (+1.36 points). Removing repeats changes which respondents are represented:
  - the largest shifts are in older age bands (65–69: +1.91 points);
  - and in the highest education and income groups (college graduates +1.90; USD 75,000+ +1.87).
- **BMI outliers:** 9,847 values (3.88%) fall outside the IQR fences of 13.5 and 41.5 (27 below, 9,820
  above); all are retained as recorded values.
- **Variant policy:**
  - `full_clean_v1` is used for descriptive statistics, inference and regression;
  - `unique_profile_v1` is used for machine learning, with a split grouped by feature vector. The
    leakage check passes: 0 profiles are shared between train (183,579 rows) and test (45,895 rows).

**Duplicate-handling sensitivity (H11).** The three baselines were trained under four scenarios (two
variants × grouped or random split). Numbers are from `outputs/tables/duplicate_sensitivity.csv`;
the full outline is in `docs/MODELING_REPORT.md` section 12a.

**Table 4.** Test-set composition by scenario (seed 42, 80/20 stratified).

| Variant | Split | Train | Test | Test prevalence | Test rows seen in training |
|---|---|---|---|---|---|
| `unique_profile_v1` (primary) | grouped | 183,579 | 45,895 | 15.3% | 0 |
| `unique_profile_v1` | random | 183,579 | 45,895 | 15.3% | 502 (1.1%) |
| `full_clean_v1` | grouped | 202,943 | 50,737 | 13.9% | 0 |
| `full_clean_v1` | random | 202,944 | 50,736 | 13.9% | 6,836 (13.5%) |

**Table 5.** ROC AUC / PR AUC / Brier score on all test rows, threshold 0.5.

| Model | Unique, grouped (primary) | Unique, random | Full, grouped | Full, random |
|---|---|---|---|---|
| Logistic regression | 0.810 / 0.414 / 0.107 | 0.810 / 0.414 / 0.107 | 0.822 / 0.406 / 0.099 | 0.819 / 0.394 / 0.100 |
| Random forest | 0.790 / 0.396 / 0.110 | 0.778 / 0.358 / 0.114 | 0.804 / 0.389 / 0.101 | 0.796 / 0.369 / 0.104 |
| Gradient boosting | 0.819 / 0.441 / 0.105 | 0.819 / 0.446 / 0.105 | 0.831 / 0.434 / 0.096 | 0.826 / 0.421 / 0.098 |

- **H11 is supported:** duplicate handling changes evaluation results, mainly through test-set
  composition rather than leakage.
- **Keeping repeats flatters ROC AUC and Brier:** full vs unique, both grouped:
  - ROC AUC +0.012 to +0.014 and Brier about −0.009;
  - but PR AUC −0.007 to −0.009, because the extra rows are mostly easy negatives.
- **A random split did not inflate metrics here:** ROC AUC changed by −0.011 to +0.001. The grouped
  split remains the right design because it is the only one that tests performance on unseen profiles.

*Write:*

---

## 5. Descriptive statistics

**Owner:** Member 1. **Answers:** RQ1, RQ2. **Sources:** `outputs/tables/full_clean_v1_target_distribution.csv`,
`full_clean_v1_numeric_summary.csv`, `full_clean_v1_prevalence_*.csv`, `full_clean_v1_binary_comparisons.csv`,
`full_clean_v1_health_days_by_target.csv`, `full_clean_v1_spearman_correlation.csv`.

Findings:

- **RQ2:** 35,346 of 253,680 respondents (13.93%) are in category 1; 218,334 (86.07%) are in category 0.
  The classes are imbalanced.
- **Numeric variables:**
  - BMI: mean 28.38 (SD 6.61), median 27, IQR 24–31.
  - Mental-health days: mean 3.18, median 0.
  - Physical-health days: mean 4.24, median 0.

  Both day counts are heavily right-skewed.
- **By target group:**
  - Mean physical-health days are 7.95 in category 1 vs 3.64 in category 0; 16.2% vs 6.3% report all
    30 days.
  - Mean mental-health days are 4.46 vs 2.98.
- **Gradients in the category 1 share (unadjusted):**
  - General health: rises from 2.52% (excellent) to 37.89% (poor).
  - Age: rises from 1.37% (18–24) to 21.85% (70–74), then falls to 18.48% (80+).
  - Income: falls from 26.19% (USD 10,000–<15,000) to 7.96% (USD 75,000+).
  - Education: falls from 29.26% (grades 1–8) to 9.69% (college graduates).
- **Binary indicators with the largest gaps** (category 1 share, yes vs no):
  - heart disease or attack: 32.97% vs 11.95% (+21.0 points);
  - difficulty walking: 30.75% vs 10.53%;
  - stroke: 31.75% vs 13.18%;
  - high blood pressure: 24.45% vs 6.04%.

  Physical activity (11.61% vs 21.14%) and heavy alcohol use (5.84% vs 14.42%) go the other way.
- **Spearman correlation with the outcome:**
  - GenHlth ρ = 0.288, BMI 0.226, Age 0.178, PhysHlth 0.157, Income −0.163, Education −0.120.
  - Among features, the strongest pairs are GenHlth–PhysHlth (0.452) and Education–Income (0.452).

  The matrix is descriptive, not an importance ranking.

**Table 6.** Category 1 share by selected characteristics (`full_clean_v1`, n = 253,680; unadjusted).

| Characteristic | Group | n | Category 1 share |
|---|---|---|---|
| High blood pressure | No / Yes | 144,851 / 108,829 | 6.04% / 24.45% |
| High cholesterol | No / Yes | 146,089 / 107,591 | 7.98% / 22.01% |
| Physical activity | No / Yes | 61,760 / 191,920 | 21.14% / 11.61% |
| Sex | Female / Male | 141,974 / 111,706 | 12.97% / 15.16% |
| General health | Excellent / Good / Poor | 45,299 / 75,646 / 12,081 | 2.52% / 17.79% / 37.89% |
| Age | 18–24 / 70–74 / 80+ | 5,700 / 23,533 / 17,363 | 1.37% / 21.85% / 18.48% |
| Income | USD 10,000–<15,000 / USD 75,000+ | 11,783 / 90,385 | 26.19% / 7.96% |

**Figures** (each caption names `full_clean_v1`, n = 253,680):

![Figure 1. Target distribution (full_clean_v1, n = 253,680).](../outputs/figures/full_clean_v1_target_distribution.png)

![Figure 2. BMI distribution by target group (full_clean_v1, n = 253,680).](../outputs/figures/full_clean_v1_bmi_by_target.png)

![Figure 3. Category 1 share by general health (full_clean_v1, n = 253,680; unadjusted).](../outputs/figures/full_clean_v1_prevalence_GenHlth.png)

![Figure 4. Category 1 share by age group (full_clean_v1, n = 253,680; unadjusted).](../outputs/figures/full_clean_v1_prevalence_Age.png)

![Figure 5. Spearman rank correlations, average ranks (full_clean_v1, n = 253,680).](../outputs/figures/full_clean_v1_spearman_correlation.png)

Other available figures: `full_clean_v1_prevalence_{HighBP,HighChol,Income,Education,PhysActivity,Sex}.png`
and `full_clean_v1_health_day_distributions.png`.

*Write:*

---

## 6. Probability and Bayes analysis

**Owner:** Member 2. **Answers:** RQ3, RQ4. **Sources:** `outputs/tables/member2_probability_results.csv`,
`outputs/tables/member2_bayes_results.csv`.

Findings:

- **Marginal:** P(category 1) = 0.1393 (95% Wilson CI 0.1380–0.1407).
- **Conditional (RQ3):** the observed probability of category 1:
  - nearly doubles with high blood pressure (0.2445);
  - is 0.2971 with both high blood pressure and high cholesterol;
  - is 0.0351 for physically active respondents with neither condition.
- **Joint vs conditional:** P(HighBP = 1 and category 1) = 0.1049 and P(HighBP = 1) = 0.4290, so
  P(category 1 | HighBP = 1) = 0.1049 / 0.4290 = 0.2445.
- **Bayes' theorem (RQ4):** for seven yes/no features in both variants, likelihood × prior / evidence
  equals the directly counted conditional probability (absolute difference 0 to floating-point precision).
- **Lift over the prior:**
  - heart disease or attack 2.37, stroke 2.28, difficulty walking 2.21 and high blood pressure 1.75;
  - physical activity is below 1 (0.83).
- These are observed shares in the sample, not model predictions or individual risks.

**Table 7.** Marginal, joint and conditional probabilities of category 1 (`full_clean_v1`,
n = 253,680; Wilson 95% CI).

| Conditions | P(conditions) | P(conditions and category 1) | Respondents meeting conditions | P(category 1 \| conditions) | 95% CI |
|---|---|---|---|---|---|
| None (all respondents) | 1.0000 | 0.1393 | 253,680 | 0.1393 | 0.1380–0.1407 |
| HighBP = 1 | 0.4290 | 0.1049 | 108,829 | 0.2445 | 0.2419–0.2470 |
| HighChol = 1 | 0.4241 | 0.0934 | 107,591 | 0.2201 | 0.2177–0.2226 |
| PhysActivity = 1 | 0.7565 | 0.0879 | 191,920 | 0.1161 | 0.1147–0.1176 |
| HighBP = 1 and HighChol = 1 | 0.2549 | 0.0757 | 64,660 | 0.2971 | 0.2936–0.3007 |
| HighBP = 0, HighChol = 0, PhysActivity = 1 | 0.3268 | 0.0115 | 82,892 | 0.0351 | 0.0339–0.0364 |
| Sex = male | 0.4403 | 0.0668 | 111,706 | 0.1516 | 0.1495–0.1537 |
| Age 70–74 | 0.0928 | 0.0203 | 23,533 | 0.2185 | 0.2132–0.2238 |
| Income under USD 10,000 | 0.0387 | 0.0094 | 9,811 | 0.2429 | 0.2345–0.2515 |
| Income USD 75,000+ | 0.3563 | 0.0284 | 90,385 | 0.0796 | 0.0779–0.0814 |

**Table 8.** Bayes' theorem check, feature value 1 (`full_clean_v1`, n = 253,680; prior P(category 1) = 0.1393).

| Feature | Likelihood P(x \| cat. 1) | Evidence P(x) | Bayes result | Direct count | Lift over prior |
|---|---|---|---|---|---|
| HighBP | 0.7527 | 0.4290 | 0.2445 | 0.2445 | 1.75 |
| HighChol | 0.6701 | 0.4241 | 0.2201 | 0.2201 | 1.58 |
| PhysActivity | 0.6305 | 0.7565 | 0.1161 | 0.1161 | 0.83 |
| Smoker | 0.5182 | 0.4432 | 0.1629 | 0.1629 | 1.17 |
| Stroke | 0.0925 | 0.0406 | 0.3175 | 0.3175 | 2.28 |
| HeartDiseaseorAttack | 0.2229 | 0.0942 | 0.3297 | 0.3297 | 2.37 |
| DiffWalk | 0.3712 | 0.1682 | 0.3075 | 0.3075 | 2.21 |

The same tables for `unique_profile_v1` are in the two source files (prior 0.1529; for example
P(category 1 | HighBP = 1) = 0.2533).

*Write:*

---

## 7. Confidence intervals

**Owner:** Member 2. **Answers:** RQ5. **Source:** `outputs/tables/member2_confidence_intervals.csv`.

Findings:

- **Methods:**
  - Wilson score intervals for proportions (Wilson, 1927);
  - t intervals (df = n − 1) for means;
  - Welch intervals with Satterthwaite df for differences in means (Welch, 1947).

  All are 95% intervals.
- **Precision:** with n in the tens of thousands, intervals are narrow (for example ±0.13 points for
  the overall share). They describe sampling uncertainty only, not survey design or measurement error.
- **BMI:** mean 31.94 (31.87–32.02) in category 1 vs 27.81 (27.78–27.83) in category 0; difference
  4.14 (4.06–4.22).
- **Physical-health days:** difference 4.31 (4.19–4.44). **Mental-health days:** difference 1.48 (1.39–1.58).
- **Sensitivity:** in `unique_profile_v1` the BMI difference is 3.87 (3.79–3.95) and the
  physical-health-days difference is 3.93 (3.80–4.05).

**Table 9.** 95% confidence intervals (`full_clean_v1`, n = 253,680).

| Estimate | Group | n | Estimate | 95% CI | Method |
|---|---|---|---|---|---|
| Category 1 share | All respondents | 253,680 | 13.93% | 13.80–14.07% | Wilson |
| Category 1 share | HighBP no / yes | 144,851 / 108,829 | 6.04% / 24.45% | 5.91–6.16% / 24.19–24.70% | Wilson |
| Category 1 share | HighChol no / yes | 146,089 / 107,591 | 7.98% / 22.01% | 7.84–8.12% / 21.77–22.26% | Wilson |
| Category 1 share | PhysActivity no / yes | 61,760 / 191,920 | 21.14% / 11.61% | 20.82–21.47% / 11.47–11.76% | Wilson |
| Category 1 share | Smoker no / yes | 141,257 / 112,423 | 12.06% / 16.29% | 11.89–12.23% / 16.08–16.51% | Wilson |
| Category 1 share | GenHlth excellent / poor | 45,299 / 12,081 | 2.52% / 37.89% | 2.38–2.66% / 37.03–38.76% | Wilson |
| Mean BMI | Category 1 / 0 | 35,346 / 218,334 | 31.94 / 27.81 | 31.87–32.02 / 27.78–27.83 | t |
| BMI difference | 1 minus 0 | 253,680 | 4.14 | 4.06–4.22 | Welch |
| Mean PhysHlth days | Category 1 / 0 | 35,346 / 218,334 | 7.95 / 3.64 | 7.84–8.07 / 3.61–3.67 | t |
| PhysHlth difference | 1 minus 0 | 253,680 | 4.31 | 4.19–4.44 | Welch |
| Mean MentHlth days | Category 1 / 0 | 35,346 / 218,334 | 4.46 / 2.98 | 4.37–4.56 / 2.95–3.01 | t |
| MentHlth difference | 1 minus 0 | 253,680 | 1.48 | 1.39–1.58 | Welch |

*Write:*

---

## 8. Hypothesis testing

**Owner:** Member 2. **Answers:** RQ6, H1–H7. **Source:** `outputs/tables/member2_hypothesis_results.csv`.

Findings:

- **Tests:**
  - chi-square tests of independence without continuity correction for H1, H2 and H4–H7 (Pearson, 1900);
  - a Welch two-sample t-test for H3 (Welch, 1947).
- **All seven null hypotheses are rejected** at p < 0.001 in both variants. With n above 229,000 the
  p-values underflow to zero, so they are reported as p < 0.001 and the effect sizes in section 9
  carry the comparison.
- **Assumption checks:**
  - every expected cell count is at least 5;
  - BMI is strongly right-skewed in both groups (skewness 1.53 in category 1, 2.33 in category 0), but
    with these group sizes the Welch test of means remains valid; a mean difference can still hide
    differences in shape.
- **Multiple comparisons:** no adjustment is applied (team decision, `docs/MEMBER2_HANDOFF.md`). Every
  p-value is below the Bonferroni threshold 0.05 / 7 = 0.0071, so even the strictest common adjustment
  would not change a decision.
- **ANOVA and post-hoc tests** are not included: H1–H7 need only chi-square and Welch tests, and the
  team scoped them out. State this as a deviation from the specification.

**Table 10.** Hypothesis tests H1–H7 (`full_clean_v1`, n = 253,680).

| Hypothesis | Feature | Test | Statistic | df | p | Effect size | Decision |
|---|---|---|---|---|---|---|---|
| H1 | HighBP | Chi-square | χ² = 17,563.98 | 1 | < 0.001 | V = 0.263 | Reject H0 |
| H2 | HighChol | Chi-square | χ² = 10,175.25 | 1 | < 0.001 | V = 0.200 | Reject H0 |
| H3 | BMI | Welch t | t = 99.92 | 44,093 | < 0.001 | d = 0.641 | Reject H0 |
| H4 | PhysActivity | Chi-square | χ² = 3,540.21 | 1 | < 0.001 | V = 0.118 | Reject H0 |
| H5 | Smoker | Chi-square | χ² = 937.41 | 1 | < 0.001 | V = 0.061 | Reject H0 |
| H6 | Age | Chi-square | χ² = 8,795.05 | 12 | < 0.001 | V = 0.186 | Reject H0 |
| H7 | Income | Chi-square | χ² = 7,003.72 | 7 | < 0.001 | V = 0.166 | Reject H0 |

No multiplicity adjustment is applied; all p-values are below 0.0071 (Bonferroni, 7 tests).

*Write:*

---

## 9. Effect sizes and multiple comparisons

**Owner:** Member 2. **Answers:** RQ7. **Source:** `outputs/tables/member2_hypothesis_results.csv`.

Findings:

- **Measures:** Cramér's V for chi-square tests (Cramér, 1946), Cohen's d with pooled SD for H3
  (Cohen, 1988), and odds ratios with 95% Woolf intervals for two-category features.
- **Rules of thumb** (Cohen, 1988): V of 0.1 / 0.3 / 0.5 (df = 1) and d of 0.2 / 0.5 / 0.8 are small /
  medium / large. They are conventions, not clinical thresholds.
- **Ranking:**
  - BMI has a medium-to-large standardized difference (d = 0.64);
  - high blood pressure (V = 0.26) and high cholesterol (V = 0.20) are small-to-medium;
  - physical activity is small (V = 0.12);
  - smoking is negligible (V = 0.06) despite p < 0.001, a clear case of significance without practical
    importance.
- **Unadjusted odds ratios:**
  - high blood pressure 5.04 (4.91–5.17);
  - high cholesterol 3.25 (3.18–3.33);
  - smoking 1.42 (1.39–1.45);
  - physical activity 0.49 (0.48–0.50).

  These shrink after adjustment (section 10), for example high blood pressure from 5.04 to 2.13.
- **Duplicate sensitivity:** every effect keeps its direction in `unique_profile_v1`. Effects shrink most
  for H5 (V 0.061 to 0.046), H7 (0.166 to 0.142) and H4 (0.118 to 0.100).
- **Multiple comparisons:** see section 8 (no adjustment; every p-value passes the Bonferroni threshold).

**Table 11.** Effect sizes by dataset variant (`full_clean_v1`, n = 253,680; `unique_profile_v1`, n = 229,474).

| Hypothesis | Feature | Measure | Full | Unique | Unadjusted OR, full (95% CI) |
|---|---|---|---|---|---|
| H1 | HighBP | Cramér's V | 0.263 | 0.254 | 5.04 (4.91–5.17) |
| H2 | HighChol | Cramér's V | 0.200 | 0.195 | 3.25 (3.18–3.33) |
| H3 | BMI | Cohen's d | 0.641 | 0.582 | — |
| H4 | PhysActivity | Cramér's V | 0.118 | 0.100 | 0.49 (0.48–0.50) |
| H5 | Smoker | Cramér's V | 0.061 | 0.046 | 1.42 (1.39–1.45) |
| H6 | Age | Cramér's V | 0.186 | 0.189 | — |
| H7 | Income | Cramér's V | 0.166 | 0.142 | — |

*Write:*

---

## 10. Logistic regression

**Owner:** Member 3. **Answers:** RQ8, H8. **Sources:** `outputs/tables/full_clean_v1_regression_baseline_odds_ratios.csv`,
`outputs/regression_manifest.json`; outline in `docs/MODELING_REPORT.md` section 10.

Findings:

- **Model:**
  - statsmodels logistic regression on all 21 features (`full_clean_v1`, n = 253,680);
  - BMI and Age centered at 28.38 and age code 8.03;
  - ordinal codes treated as numeric scores.

  It converged; McFadden pseudo-R² = 0.208.
- **H8 is supported:** most features remain associated after adjustment. The largest adjusted odds ratios:
  - HighBP 2.13 (2.07–2.20);
  - HighChol 1.78 (1.74–1.83);
  - GenHlth 1.71 per step worse (1.68–1.74);
  - BMI 1.063 per point, about 1.36 per 5 points (1.061–1.065);
  - Age 1.13 per band (1.13–1.14).
- **Not distinguishable from 1:** Smoker 0.99 (0.96–1.02) and NoDocbcCost 1.02 (0.97–1.07). Smoking's
  unadjusted association (OR 1.42) disappears after adjustment.
- **Wording cautions:**
  - **CholCheck** (OR 3.47) probably reflects reverse direction: people who already know they have the
    condition are checked more often.
  - **Heavy alcohol use** (OR 0.46) is likely confounding or reporting bias; do not call it protective.

**Table 12.** Adjusted odds ratios (`full_clean_v1`, n = 253,680; 95% CI; all other features held fixed).

| Feature | OR | 95% CI | Unit |
|---|---|---|---|
| CholCheck | 3.47 | 3.03–3.97 | had a check vs not |
| HighBP | 2.13 | 2.07–2.20 | yes vs no |
| HighChol | 1.78 | 1.74–1.83 | yes vs no |
| GenHlth | 1.71 | 1.68–1.74 | per step worse (1–5) |
| Sex | 1.29 | 1.26–1.33 | male vs female |
| HeartDiseaseorAttack | 1.25 | 1.20–1.29 | yes vs no |
| Stroke | 1.14 | 1.09–1.20 | yes vs no |
| Age | 1.13 | 1.13–1.14 | per 5-year band |
| DiffWalk | 1.13 | 1.09–1.17 | yes vs no |
| BMI | 1.063 | 1.061–1.065 | per BMI point |
| Income | 0.95 | 0.94–0.96 | per income category |
| PhysActivity | 0.95 | 0.92–0.98 | yes vs no |
| HvyAlcoholConsump | 0.46 | 0.43–0.50 | yes vs no |
| Smoker | 0.99 | 0.96–1.02 | yes vs no (CI includes 1) |
| NoDocbcCost | 1.02 | 0.97–1.07 | yes vs no (CI includes 1) |

All 21 odds ratios are in the source file.

![Figure 6. Adjusted odds ratios with 95% CIs, baseline logistic regression (full_clean_v1, n = 253,680).](../outputs/figures/full_clean_v1_regression_odds_ratios.png)

*Write:*

---

## 11. BMI × Age interaction

**Owner:** Member 3. **Answers:** RQ9, H9. **Sources:** `outputs/tables/full_clean_v1_regression_model_comparison.csv`,
`full_clean_v1_regression_bmi_odds_by_age.csv`; outline in `docs/MODELING_REPORT.md` section 11.

Findings:

- **H9 is supported on the odds scale:** interaction odds ratio 1.0050 (95% CI 1.0043–1.0056).
- **Fit:**
  - likelihood-ratio χ² = 253.4 on 1 df, p ≈ 5 × 10⁻⁵⁷;
  - AIC falls by 251.4 and BIC by 240.9;
  - pseudo-R² rises only from 0.2083 to 0.2095, so the interaction is real but modest.
- **BMI odds ratio per point by age:** 1.026 at 18–24, 1.047 at 40–44, 1.068 at 60–64 and 1.089 at 80+.
  Per 5 points that is about 1.14, 1.26, 1.39 and 1.53.
- **Reading cautions:** this is an interaction on the log-odds scale; Age is in bands and treated as
  linear; the cross-sectional data say nothing about BMI change over a lifetime.

**Table 13.** Model comparison (`full_clean_v1`, n = 253,680).

| Model | Log-likelihood | AIC | BIC | Pseudo-R² |
|---|---|---|---|---|
| Baseline | −81,091.8 | 162,227.6 | 162,457.4 | 0.2083 |
| Baseline + BMI × Age | −80,965.1 | 161,976.2 | 162,216.5 | 0.2095 |

![Figure 7. Odds ratio per 1-unit BMI increase by age group, BMI × Age interaction model (full_clean_v1, n = 253,680).](../outputs/figures/full_clean_v1_regression_bmi_odds_by_age.png)

*Write:*

---

## 12. Machine-learning models

**Owner:** Member 3. **Answers:** RQ10. **Source:** `outputs/tables/model_comparison.csv`; outline in
`docs/MODELING_REPORT.md` section 12.

Findings:

- **Setup:**
  - three default scikit-learn models: logistic regression, random forest with 100 trees, and gradient
    boosting with 100 trees of depth 3;
  - trained on 183,579 rows and evaluated once on 45,895 held-out rows of `unique_profile_v1`;
  - 80/20 split, stratified and grouped by profile, seed 42.
- **Accuracy is uninformative:** about 0.85 for every model, close to the 0.847 from always predicting 0.
- **Gradient boosting leads slightly:** ROC AUC 0.819, PR AUC 0.441, Brier 0.105. Logistic regression is
  close behind (0.810, 0.414, 0.107); the random forest is weakest (0.789, 0.396, 0.110).
- **PR AUC is 2.7–2.9 times the no-skill baseline** of 0.153 (the test prevalence).
- **Recall at threshold 0.5 is low:** 0.15–0.18, so every model misses more than 80% of category 1
  respondents (motivates sections 13–14).
- **Caution:** one split and no confidence intervals, so differences of about 0.01 AUC may not be meaningful.

**Table 14.** Test-set performance at threshold 0.5 (`unique_profile_v1`, test n = 45,895).

| Model | Accuracy | ROC AUC | PR AUC | Brier | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.852 | 0.810 | 0.414 | 0.107 | 0.153 | 0.554 | 0.978 | 0.239 |
| Random forest | 0.850 | 0.789 | 0.396 | 0.110 | 0.184 | 0.524 | 0.970 | 0.272 |
| Gradient boosting | 0.855 | 0.819 | 0.441 | 0.105 | 0.164 | 0.588 | 0.979 | 0.256 |

![Figure 8. ROC curves on the held-out test set (unique_profile_v1, n = 45,895).](../outputs/figures/unique_profile_v1_roc_curves.png)

![Figure 9. Precision-recall curves on the held-out test set (unique_profile_v1, n = 45,895; no-skill line at 0.153).](../outputs/figures/unique_profile_v1_precision_recall_curves.png)

*Write:*

---

## 13. Class-imbalance experiments

**Owner:** Member 3. **Answers:** RQ11, H10. **Source:** `outputs/tables/imbalance_experiments.csv`;
outline in `docs/MODELING_REPORT.md` section 13.

Findings:

- **Setup:**
  - logistic regression on 155,501 negatives and 28,078 positives;
  - strategies: original, class-weighted (`"balanced"`) and random undersampling to 28,078 each;
  - all scored on the same untouched test set.
- **H10 is supported for classification behavior but not for ranking:**
  - recall rises from 0.153 to 0.762, precision falls from 0.554 to 0.322, and F1 rises from 0.239
    to about 0.45;
  - ROC AUC stays at 0.810.
- **Equivalent to a lower threshold:** the original model at threshold 0.15 gives recall 0.770 and
  precision 0.318.
- **Probabilities get worse:** Brier rises from 0.107 to 0.184. Undersampling matches class weighting
  while discarding 127,423 negative rows.

**Table 15.** Imbalance strategies, logistic regression, threshold 0.5 (`unique_profile_v1`, test n = 45,895).

| Strategy | Training negatives / positives | Recall | Precision | Specificity | F1 | ROC AUC | Brier |
|---|---|---|---|---|---|---|---|
| Original | 155,501 / 28,078 | 0.153 | 0.554 | 0.978 | 0.239 | 0.810 | 0.107 |
| Class-weighted | 155,501 / 28,078 (weighted) | 0.762 | 0.322 | 0.711 | 0.453 | 0.810 | 0.184 |
| Undersampling | 28,078 / 28,078 | 0.762 | 0.322 | 0.710 | 0.452 | 0.810 | 0.184 |

*Write:*

---

## 14. Threshold analysis

**Owner:** Member 3. **Answers:** RQ12 (thresholds). **Source:** `outputs/tables/thresholds_analysis_*.csv`;
outline in `docs/MODELING_REPORT.md` section 14.

Findings:

- **Setup:** thresholds 0.10–0.90 in steps of 0.05 on the test set, reported as a trade-off. **No
  threshold is selected**, because choosing one on the test set would bias the results.
- **0.5 is a poor default** when 15% of respondents are positive. F1 peaks well below it:
  - logistic regression 0.465 at 0.20;
  - gradient boosting 0.474 at 0.25;
  - random forest 0.450 at 0.25.
- **The right threshold depends on costs:** screening favors recall (a low threshold); limited follow-up
  capacity favors precision.
- **Above about 0.8** very few rows are flagged, so precision rests on small counts.

**Table 16.** Recall / precision by threshold (`unique_profile_v1`, test n = 45,895).

| Threshold | Logistic regression | Gradient boosting | Random forest |
|---|---|---|---|
| 0.10 | 0.876 / 0.270 | 0.880 / 0.276 | 0.854 / 0.258 |
| 0.20 | 0.661 / 0.358 | 0.683 / 0.363 | 0.689 / 0.325 |
| 0.30 | 0.447 / 0.427 | 0.480 / 0.435 | 0.514 / 0.392 |
| 0.50 | 0.153 / 0.554 | 0.164 / 0.588 | 0.184 / 0.524 |

![Figure 10. Precision, recall, specificity and F1 across thresholds (unique_profile_v1, test n = 45,895).](../outputs/figures/unique_profile_v1_threshold_tradeoffs.png)

*Write:*

---

## 15. Calibration

**Owner:** Member 3. **Answers:** RQ12 (calibration). **Sources:** `outputs/tables/calibration_table_*.csv`,
`outputs/models/metrics.json`; outline in `docs/MODELING_REPORT.md` section 15.

Findings:

- **Gradient boosting is best calibrated** (Brier 0.1051, expected calibration error 0.0074), following
  the diagonal up to about 0.7.
- **Logistic regression** (Brier 0.1075, ECE 0.0117) is well calibrated below 0.4 and slightly
  overestimates above it (0.64 predicted vs 0.61 observed in the 0.6–0.7 bin).
- **Random forest** (Brier 0.1104, ECE 0.0213) overestimates from about 0.4 upward (0.73 vs 0.59 in the
  0.7–0.8 bin).
- **Overall scores mostly reflect the low-risk range:** about half of test rows fall in the 0–0.1 bin.
- **Discrimination and calibration are separate:** rebalanced models rank as well but are miscalibrated
  (section 13).
- Bins with fewer than 50 rows are not interpreted.

**Table 17.** Calibration summary (`unique_profile_v1`, test n = 45,895; 10 equal-width bins).

| Model | Brier score | Expected calibration error |
|---|---|---|
| Gradient boosting | 0.1051 | 0.0074 |
| Logistic regression | 0.1075 | 0.0117 |
| Random forest | 0.1104 | 0.0213 |

![Figure 11. Calibration curves, 10 equal-width bins (unique_profile_v1, test n = 45,895).](../outputs/figures/unique_profile_v1_calibration_curves.png)

*Write:*

---

## 16. Model interpretation

**Owner:** Member 3. **Sources:** `outputs/tables/feature_importance_*.csv`; outline in
`docs/MODELING_REPORT.md` section 16.

Findings:

- **Method:** permutation importance, the mean drop in ROC AUC over 10 shuffles of each feature, on
  10,000 test rows.
- **The models agree:** all three share the same top five features (GenHlth, BMI, HighBP, Age,
  HighChol), which are also the strongest regression associations.
- **Importance and odds ratios measure different things:** CholCheck has the largest odds ratio but low
  importance, because only 3.7% of respondents lack a check.
- **Cautions:**
  - correlated features (GenHlth, PhysHlth, DiffWalk) share credit;
  - importance is predictive usefulness for this model, not causation.

**Table 18.** Top five features by permutation importance, mean ROC AUC drop (`unique_profile_v1`,
10,000 test rows).

| Rank | Logistic regression | Random forest | Gradient boosting |
|---|---|---|---|
| 1 | GenHlth (0.060) | GenHlth (0.047) | GenHlth (0.052) |
| 2 | BMI (0.033) | BMI (0.041) | BMI (0.043) |
| 3 | HighBP (0.028) | Age (0.028) | Age (0.034) |
| 4 | Age (0.023) | HighBP (0.026) | HighBP (0.021) |
| 5 | HighChol (0.013) | HighChol (0.016) | HighChol (0.011) |

![Figure 12. Permutation importance, mean ROC AUC drop over 10 repeats (unique_profile_v1, 10,000 test rows).](../outputs/figures/unique_profile_v1_feature_importance.png)

*Write:*

---

## 17. Limitations

**Owner:** shared. All ten items from spec section 27 must appear.

1. The UCI/Kaggle dataset is derived from BRFSS 2015, not the raw complex-survey environment.
2. Survey weights and design are not reproduced: results describe this sample, not U.S. prevalence.
3. The positive class combines prediabetes and diabetes.
4. The data are cross-sectional and observational, so no causal claims are made.
5. 24,206 exact duplicate rows exist and there is no respondent identifier.
6. Deduplication for machine learning is an analytical choice, not proof that repeats are errors.
7. Predictive performance (ROC AUC up to 0.819) does not establish clinical utility or external validity.
8. Threshold selection depends on context and costs; none was chosen here.
9. Calibration and discrimination can change in other populations.
10. Results should not be generalized to current populations without validation.

Section-specific limitations:

- **Data:** self-reported measures carry reporting error; BMI extremes (up to 98) are unverified.
- **Inference:**
  - p-values underflow at this sample size;
  - no multiplicity adjustment, and ANOVA and post-hoc tests were not implemented (team decision);
  - intervals ignore survey design.
- **Regression:** ordinal codes are assumed linear on the log-odds scale; repeated rows in
  `full_clean_v1` can make standard errors too small.
- **Machine learning:**
  - one split, so no confidence intervals on the metrics;
  - no tuning and no validation set;
  - imbalance experiments cover logistic regression only.
- **Prediction Explorer:** 13 of 21 inputs default to the most common training values; the output is a
  model estimate, not a diagnosis.

*Write:*

---

## 18. Conclusion

**Owner:** shared. One or two sentences per research question, with its key number.

| RQ | Short answer | Key evidence (section) |
|---|---|---|
| RQ1–RQ2 | 13.93% in category 1; strong gradients by general health, age and income, plus higher shares with cardiometabolic conditions | 5 |
| RQ3–RQ4 | P(category 1) rises from 0.139 to 0.245 with high blood pressure and 0.297 with high blood pressure and high cholesterol; Bayes' theorem reproduces every direct count | 6 |
| RQ5 | Intervals are narrow at this n, e.g. BMI difference 4.14 (4.06–4.22) | 7 |
| RQ6–RQ7 | All of H1–H7 reject independence (p < 0.001); effects range from negligible (smoking, V = 0.06) to medium-large (BMI, d = 0.64) | 8, 9 |
| RQ8 | Most features stay associated after adjustment: HighBP OR 2.13, HighChol 1.78, GenHlth 1.71 per step; smoking does not | 10 |
| RQ9 | The BMI association strengthens with age on the odds scale (1.026 to 1.089 per point); modest fit gain | 11 |
| RQ10 | Gradient boosting leads narrowly (ROC AUC 0.819, PR AUC 0.441); logistic regression is close (0.810, 0.414) | 12 |
| RQ11 | Rebalancing trades precision for recall (recall 0.15 to 0.76) without improving ranking, and worsens calibration | 13 |
| RQ12 | F1 peaks at thresholds 0.20–0.25; gradient boosting is best calibrated (ECE 0.007) | 14, 15 |
| RQ13 | Grouped split leaves 0 shared profiles; keeping repeats raises ROC AUC by about 0.013 but lowers PR AUC | 4 |

Future work: a validation set or cross-validation for tuning and threshold choice; survey weights;
external validation on another survey year.

*Write:*

---

## 19. References

Paste these entries from `docs/references.md` (APA 7, alphabetical, hanging indent). Only entries cited
above are listed. Agresti & Coull (1998), Benjamini & Hochberg (1995) and Tukey (1949) are **not** used,
because Wilson intervals were implemented and multiplicity adjustment and post-hoc tests were scoped out.

- Data: *CDC Diabetes Health Indicators* (2017); CDC (2016a, 2016b); Teboul (2021)
- Context: Xie et al. (2019)
- Statistics: Cohen (1988); Cramér (1946); Pearson (1900); Spearman (1904); Welch (1947); Wilson (1927)
- Regression: Akaike (1974); Hosmer et al. (2013); McFadden (1974); Schwarz (1978); Wilks (1938)
- Machine learning and evaluation: Breiman (2001); Brier (1950); Fawcett (2006); Friedman (2001);
  He & Garcia (2009); Kaufman et al. (2012); Niculescu-Mizil & Caruana (2005); Pakdaman Naeini et al.
  (2015); Saito & Rehmsmeier (2015); Steyerberg et al. (2010); Strobl et al. (2008); Van Calster et al. (2019)
- Software: Harris et al. (2020); Hunter (2007); McKinney (2010); Pedregosa et al. (2011); Plotly (2015);
  Seabold & Perktold (2010); Snowflake (n.d.); Virtanen et al. (2020)

---

## Appendix A. Reproducibility

```text
python -m pip install -r requirements.txt
python -m src.analysis.pipeline --csv data/raw/cdc_diabetes.csv
python -m pytest tests/ -q
streamlit run streamlit_app.py
```

- The pipeline downloads the verified UCI dataset if it is missing, then runs the EDA, statistics,
  regression, modeling and duplicate-sensitivity exports in about 2.5 minutes.
- **Dataset version:** SHA-256 `0f8ff13c40c0908fc6e9bf43dc05b3be706c6195481ac595b8e0adb256a79774`
  (recorded in the regression, modeling and duplicate-sensitivity manifests).
  - `outputs/eda_manifest.json` records `4b702ecd…a6f60` because the EDA was first exported from a
    differently formatted copy of the same data. Both copies pass every reference check, and
    re-running the EDA export on the `0f8ff13c…` file reproduces every EDA table exactly.
- **Reproducibility check:** a full re-run reproduced 153 of the 154 exported tables exactly. In
  `duplicate_sensitivity.csv`, ROC AUC and PR AUC differ by at most 0.00001 (multithreaded training).
- **Software used to produce the results:** Python 3.14, scikit-learn 1.9.1, statsmodels 0.15.0,
  pandas 3.0, NumPy 2.5, SciPy 1.18, Matplotlib 3.11, Streamlit 1.64, Plotly 7.1 (see `docs/references.md`).
- **Tests:** 548 automated tests pass on synthetic data.
- Commit described by this report: [FINAL COMMIT HASH].

## Appendix B. Streamlit application

Live at https://diabeteshealthanalytics-arunava-binish-sanjay.streamlit.app. Add one screenshot per page with one sentence each:

| Page | What it shows |
|---|---|
| Executive Dashboard | Key figures, selected findings, research questions and methodology notes |
| Data Explorer | Schema, category mappings, descriptive statistics and data-quality checks |
| Exploratory Analysis | Per-variable distributions and category 1 shares, binary indicators, correlations, full vs unique comparison |
| Probability Explorer | Marginal, joint and conditional probability for up to four conditions; Bayes' theorem check |
| Statistical Testing Lab | H1–H7 with effect sizes and odds ratios; single chi-square or Welch tests; confidence intervals |
| Regression Lab | Adjusted odds ratios and the BMI × Age interaction |
| Machine Learning Lab | Metrics, confusion matrices, ROC and PR curves, duplicate-handling sensitivity |
| Imbalance & Threshold Lab | Rebalancing strategies and threshold trade-offs |
| Calibration | Calibration curves and Brier scores |
| Interpretation | Permutation importance alongside odds ratios |
| Prediction Explorer | Model-estimated probability for a feature profile; **a statistical estimate, not a diagnosis** |

---

## Traceability table

| Reported value | Section | Source file | Column / key |
|---|---|---|---|
| 253,680 rows, 22 columns, 0 missing | 2 | `outputs/tables/full_clean_v1_data_quality.json` | `rows`, `columns`, `missing_cells` |
| Reference checks pass | 2 | `outputs/eda_manifest.json` | `reference_checks` |
| 24,206 repeated rows (9.54%); 229,474 profiles; 35,575 rows in repeated patterns | 4 | `outputs/tables/full_clean_v1_data_quality.json` | `duplicate_rows`, `duplicate_rate_percent`, `distinct_patterns`, `rows_in_repeated_patterns` |
| Category 1 share 13.93% / 15.29% | 4, 5 | `outputs/tables/{full_clean_v1,unique_profile_v1}_target_distribution.csv` | `percentage` |
| Largest full-vs-unique shifts | 4 | `outputs/tables/full_vs_unique_sensitivity.csv` | `change_unique_minus_full` |
| 9,847 BMI IQR flags (fences 13.5 and 41.5) | 4 | `outputs/tables/full_clean_v1_data_quality.json` | `potential_outliers.BMI.flagged_count`, `lower_boundary`, `upper_boundary` |
| 0 shared profiles; 183,579 / 45,895 rows | 4, 12 | `outputs/modeling_manifest.json` | `leakage_check`, `train_rows`, `test_rows` |
| Duplicate scenarios (Tables 4–5) | 4 | `outputs/tables/duplicate_sensitivity.csv` | `scenario`, `roc_auc`, `pr_auc`, `brier_score`, row counts |
| BMI and health-day summaries | 5 | `outputs/tables/full_clean_v1_numeric_summary.csv`, `full_clean_v1_health_days_by_target.csv` | `mean`, `std`, `q1`, `q3`; `*_mean`, `*_30_percent` |
| Category 1 share by group (Table 6) | 5 | `outputs/tables/full_clean_v1_prevalence_*.csv` | `total_observations`, `target_percentage` |
| Binary indicator gaps | 5 | `outputs/tables/full_clean_v1_binary_comparisons.csv` | `category_1_percent_yes/no`, `difference_percentage_points` |
| Spearman ρ with the outcome | 5 | `outputs/tables/full_clean_v1_spearman_correlation.csv` | column `Diabetes_binary` |
| Marginal, joint, conditional probabilities (Table 7) | 6 | `outputs/tables/member2_probability_results.csv` | `p_conditions`, `p_conditions_and_category1`, `p_category1_given_conditions`, `ci_lower`, `ci_upper` |
| Bayes check and lift (Table 8) | 6 | `outputs/tables/member2_bayes_results.csv` | `likelihood_p_feature_given_category1`, `evidence_p_feature`, `bayes_rhs`, `direct_p_category1_given_feature`, `lift_over_prior` |
| Confidence intervals (Table 9) | 7 | `outputs/tables/member2_confidence_intervals.csv` | `estimate`, `ci_lower`, `ci_upper`, `method` |
| H1–H7 statistics, effect sizes, unadjusted ORs (Tables 10–11) | 8, 9 | `outputs/tables/member2_hypothesis_results.csv` | `statistic`, `dof`, `p_value`, `effect_value`, `odds_ratio`, `or_lower`, `or_upper` |
| BMI skewness 1.53 / 2.33 | 8 | `outputs/tables/member2_hypothesis_results.csv` | `warnings` (H3) |
| Adjusted ORs (Table 12) | 10 | `outputs/tables/full_clean_v1_regression_baseline_odds_ratios.csv` | `odds_ratio`, `ci_lower`, `ci_upper`, `p_value` |
| Centering means, convergence | 10 | `outputs/regression_manifest.json` | `centering_means`, `converged` |
| Model comparison, LR test (Table 13) | 11 | `outputs/tables/full_clean_v1_regression_model_comparison.csv`, `outputs/regression_manifest.json` | `log_likelihood`, `aic`, `bic`, `pseudo_r2`; `model_comparison`, `interaction_term` |
| BMI OR by age | 11 | `outputs/tables/full_clean_v1_regression_bmi_odds_by_age.csv` | `bmi_odds_ratio`, `ci_lower`, `ci_upper` |
| Classifier metrics (Table 14) | 12 | `outputs/tables/model_comparison.csv` | metric columns |
| Imbalance strategies (Table 15) | 13 | `outputs/tables/imbalance_experiments.csv` | metric columns, `train_*_after` |
| Threshold trade-offs (Table 16), F1 peaks, threshold 0.15 | 13, 14 | `outputs/tables/thresholds_analysis_*.csv` | `threshold`, `recall`, `precision`, `f1` |
| Brier, ECE (Table 17) | 15 | `outputs/models/metrics.json` | `test_metrics.<model>.brier_score`, `calibration.<model>.expected_calibration_error` |
| Calibration bins | 15 | `outputs/tables/calibration_table_*.csv` | `mean_predicted`, `observed_frequency`, `count` |
| Permutation importance (Table 18) | 16 | `outputs/tables/feature_importance_*.csv` | `importance_mean` |
| Dataset SHA-256 | A | `outputs/regression_manifest.json`, `outputs/modeling_manifest.json` | `dataset_version_sha256` |

## Final checks before submission

- [ ] Title page placeholders filled: course, names, date, commit hash, app URL.
- [ ] Every *Write:* line replaced with prose; guidance blocks and "Sources" lines deleted.
- [ ] No "diagnosis" language except disclaimers; target wording consistent.
- [ ] Every figure and table caption names its dataset variant and n.
- [ ] Every p-value has an interval or effect size next to it.
- [ ] Deviation from the spec stated: no ANOVA, post-hoc tests or multiplicity adjustment (team decision).
- [ ] All ten spec limitations present.
- [ ] References in APA 7, all cited in the text, none missing.
