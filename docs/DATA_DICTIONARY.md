# Data dictionary and provenance

Source: [UCI dataset 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators),
using ucimlrepo. This is the full 253,680-record binary dataset, not a balanced
70,692-record variant. The source CSV remains local and immutable.

The executable dictionary is `src/common/config.py:VARIABLE_METADATA`; the
notebook displays all definitions and labels. All 22 columns in the acquired CSV
are numeric, but many are category codes. The full dataset has no respondent ID
among its 21 analysis features and target.

## Target

The team's specification and imported dictionary define code 0 as neither
prediabetes nor diabetes, and code 1 as prediabetes or diabetes. This is the
project's agreed definition; original BRFSS preprocessing was not independently
audited. EDA charts retain Category 0/1 to avoid overstating the outcome.

## Types

- Numeric: BMI.
- Count-like integers: MentHlth and PhysHlth, days in 0–30; zero is valid.
- Ordered categories: GenHlth (1–5), Age (1–13), Education (1–6), Income (1–8).
- Binary features: HighBP, HighChol, CholCheck, Smoker, Stroke,
  HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
  AnyHealthcare, NoDocbcCost, DiffWalk and Sex.

Binary codes are 0=No and 1=Yes except Sex (0=Female, 1=Male). The Smoker code
describes lifetime consumption of at least 100 cigarettes, not current smoking.
PhysActivity excludes job activity and refers to the past 30 days. Fruits and
Veggies represent consumption at least once per day. NoDocbcCost describes a
needed visit prevented by cost in the preceding 12 months. CholCheck refers to
a check within five years. Income is annual household income in USD.

## Category mappings

| Age | Label |
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

| Education | Label |
|---:|---|
| 1 | No schooling / kindergarten only |
| 2 | Grades 1–8 |
| 3 | Grades 9–11 |
| 4 | Grade 12 / GED |
| 5 | Some college / technical school |
| 6 | College graduate / 4+ years |

| Income | Annual household income (USD) |
|---:|---|
| 1 | Under 10,000 |
| 2 | 10,000 to under 15,000 |
| 3 | 15,000 to under 20,000 |
| 4 | 20,000 to under 25,000 |
| 5 | 25,000 to under 35,000 |
| 6 | 35,000 to under 50,000 |
| 7 | 50,000 to under 75,000 |
| 8 | 75,000 or more |

GenHlth: 1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor.

The ordinal mappings agree with the supplied project specification and
[CDC BRFSS 2015 codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/CODEBOOK15_LLCP.pdf)
(EDUCA, INCOME2, _AGEG5YR). The UCI description contains inconsistent year/target
summary text; these differences are not proof of a different dataset. Source hash,
schema and reference counts are recorded for reproducibility.

## Variants and handling

full_clean_v1 retains all source rows. unique_profile_v1 removes complete-row
duplicates explicitly. Both preserve BMI extremes; IQR flags are not clinical
validity thresholds. No survey weights are present in the analysis columns or
applied here. Reported percentages describe this sample.
