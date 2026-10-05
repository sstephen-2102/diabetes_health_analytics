# Member 1: Data and EDA handoff

This change integrates the supplied `CDC DIABETES.ipynb` into the downloaded
starter repository. It implements the data/EDA contracts and two EDA-related
service functions. It does not complete the project's full app, inference or ML.

## Function signatures explained

The signature is the function's first line: its name, arguments, defaults and
return type. For example:

```python
def load_dataset(file_path: str) -> pd.DataFrame:
```

The caller supplies a CSV filename and receives a pandas DataFrame. The former
`pass` placeholder is now an implementation that reads and checks a CSV.

## Implemented contracts

| Function | Return and behavior |
|---|---|
| `load_dataset(file_path)` | Numeric non-empty DataFrame; errors wrapped in DataLoadError; no row removal. |
| `download_kaggle_dataset(dataset_id, output_directory)` | Output directory string; optional kagglehub dependency; existing changed files cannot be overwritten. |
| `validate_dataset(data, required_columns, target_column)` | Dict with is_valid, errors, warnings, variables and separate reference_checks. required_columns includes target and defines exact schema. |
| `validate_variable(data, column, variable_metadata)` | Per-variable dict of validity, missingness, invalid count and dtype. Metadata may be one entry or the entire dictionary. |
| `generate_data_quality_report(data)` | JSON-compatible quality dict including duplicated extra rows, all repeated-pattern rows, missingness, constants and BMI IQR flags. |
| `prepare_analysis_data(data, target_column, feature_columns, drop_missing=False, dataset_variant='full_clean_v1')` | Independent DataFrame; optional explicit missing-row removal; complete-row deduplication only for unique_profile_v1. |
| `summarize_numeric_variable(data, column)` | Dict of sample-size and summary statistics; known category codes rejected. |
| `summarize_categorical_variable(data, column, include_percentages=True)` | DataFrame with category, label, count and optional percentage; missing rows remain visible. |
| `compare_by_target(data, feature, target_column)` | Numeric summaries by target, or categorical counts/percentages within target groups. |
| `calculate_prevalence_table(data, feature, target_column)` | DataFrame with category, label, total_observations, target_count and target_percentage. Count refers to target code 1. |
| `calculate_correlation_matrix(data, columns, method='spearman')` | Labeled matrix; constants yield undefined correlations. Effect-size placeholder in the same module remains untouched. |
| `build_eda_tables(data)` | Dict with quality, validation and tables for all 21 features. |
| `compare_dataset_variants(data)` | Full/unique data bundles and a sensitivity DataFrame. |
| `export_eda(csv_path, output_directory='outputs', check_reference=True)` | Run manifest; exports tables, quality reports, figures and results text. Strict full-source reference check is the default. |
| `get_dashboard_summary(data, target_column)` | Quality report and target table for the service layer. |
| `get_eda_summary(data, feature)` | Feature summary, comparison, optional prevalence table, metadata and warnings. |

Existing positional signatures are retained. The optional dataset_variant argument
was added to match the written specification while preserving four-argument calls.
Model code should request variants explicitly rather than expect the default to deduplicate.

## Usage

From the repository root:

```python
from src.common.config import FEATURE_COLUMNS, EXPECTED_COLUMNS
from src.data.loader import load_dataset
from src.data.validation import validate_dataset
from src.data.preprocessing import prepare_analysis_data
from src.statistics.descriptive import calculate_prevalence_table

raw = load_dataset('data/raw/cdc_diabetes.csv')
checks = validate_dataset(raw, EXPECTED_COLUMNS, 'Diabetes_binary')
if not checks['is_valid']:
    raise ValueError(checks['errors'])

full = prepare_analysis_data(raw, 'Diabetes_binary', FEATURE_COLUMNS)
unique = prepare_analysis_data(raw, 'Diabetes_binary', FEATURE_COLUMNS,
                               dataset_variant='unique_profile_v1')
table = calculate_prevalence_table(full, 'Age', 'Diabetes_binary')
```

The reference checks additionally detect changes from the full source's recorded
row count, missingness, duplicate count and target counts. These checks are not a
guarantee of identical file contents; the run manifest records SHA-256 separately.
Invalid schema/values fail the exporter before outputs are written.

Frame attrs include dataset_variant; returned tables carry target_definition where
relevant. Since CSV loses attrs, exports include an explicit dataset_variant column.
Sensitivity rows label full versus unique columns rather than a single variant.

## Reproduce

```text
python -m pip install -r requirements.txt
python -m unittest tests.test_member1_data_eda -v
python -m src.analysis.export_eda --csv data/raw/cdc_diabetes.csv
```

Open `notebooks/CDC_DIABETES_EDA.ipynb` from the project and run all cells. The first
cell fetches UCI dataset 891 only if the local CSV is absent. It checks the full
reference before caching. This notebook acquisition path was exercised previously
via a separate UCI download; notebook verification itself used the local verified CSV.

## Validation performed

- All 25 data/EDA tests passed using Python unittest.
- The 59-cell cleaned notebook executed from top to bottom without cell errors.
- Full UCI dataset: 253,680 rows, 22 columns, 0 missing cells and 24,206 extra repeated rows.
- All five full-source reference checks passed.
- Unique complete profiles: 229,474.
- Full target-1 percentage: 13.9333%; unique: 15.2945%; change: +1.3612 percentage points.
- Exported labeled results for both variants; representative figures visually inspected.

Tests cover valid/invalid CSVs, missing/extra columns, category/range failures,
missingness, target validity, no source mutation, category denominators, empty
categories, constants, correlation ties and explicit deduplication stages.
Kaggle acquisition was tested with a mocked downloader, not a live Kaggle account.
The other team's tests are still skeleton placeholders; a full-project passing
test run would not establish that inference/modeling/application modules work.

## Scope boundaries and unresolved points

- Source definitions follow the provided project and imported UCI dictionary.
  Original binary-target preprocessing is not independently audited.
- BMI extremes are retained and remain unverified.
- Complete-row deduplication still permits identical feature vectors with different
  target labels. The modeling team must enforce its separate train/test feature-vector
  leakage rule. This change does not implement a train/test split or balancing.
- Streamlit page renderers and the full orchestration pipeline remain skeletons;
  EDA service functions are ready for the shared UI integration.
- Inference, effect sizes, model training, calibration and prediction are not
  implemented by this contribution.
- Raw CSVs, personal paths, secrets, virtual environments and caches are excluded
  from the GitHub changes package. Data are reacquired locally when needed.

## Review and collaboration

Use the team's `feature/member1-data` branch, then open a pull request against the
agreed integration branch (develop if the team uses it, otherwise main).
The ZIP is based on the supplied snapshot, not current live GitHub state. Compare
the files against the latest checkout, particularly common/config.py,
statistics/association.py and app/services.py, before replacing concurrent work.

Suggested PR title: `feat: integrate Member 1 data validation and EDA`

AI assistance was used to adapt the notebook, implement contracts and prepare tests
and documentation. The user should review and understand the contribution before
presenting it as their project work.
