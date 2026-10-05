# Diabetes Health Analytics & Prediction Lab

Function Signature Skeleton v1 for the University of San Diego M.S. Applied AI team project.

The repository began as a function-signature skeleton. Member 1 data validation,
explicit data variants, descriptive summaries and EDA exports are now implemented.
Inference, modeling and the full Streamlit application remain separate team work.

Architecture: Streamlit UI -> Service Layer -> Analytics -> Data.

Team: Member 1 = data/EDA; Member 2 = statistics; Member 3 = modeling; shared = common/analysis/app/tests.

Target: `Diabetes_binary`. Verify exact dataset columns and category mappings before implementation.

## Member 1 EDA

Open `notebooks/CDC_DIABETES_EDA.ipynb` inside this repository and run all cells.
It downloads UCI dataset 891 if the local raw CSV is absent, verifies the reference,
and uses the original full sample for EDA. Raw files are excluded from Git.

```text
python -m pip install -r requirements.txt
python -m unittest tests.test_member1_data_eda -v
python -m src.analysis.export_eda --csv data/raw/cdc_diabetes.csv
```

Outputs identify `full_clean_v1` and `unique_profile_v1`. Read
`outputs/EDA_RESULTS.md`, `docs/DATA_DICTIONARY.md` and
`docs/MEMBER1_HANDOFF.md` for results, signatures, decisions and test scope.

Target code 1 follows the project definition: **prediabetes or diabetes**.
This work is descriptive, unadjusted sample analysis; no causal or diagnostic claims.
