"""Reusable EDA tables and explicit full-vs-unique sensitivity comparisons."""
import pandas as pd
from src.common.config import FEATURE_COLUMNS, EXPECTED_COLUMNS, VARIABLE_METADATA, TARGET_DEFINITION
from src.common.exceptions import DataValidationError
from src.data.validation import validate_dataset
from src.data.preprocessing import prepare_analysis_data
from src.data.quality import generate_data_quality_report
from src.statistics.descriptive import (
    summarize_numeric_variable, summarize_categorical_variable,
    compare_by_target, calculate_prevalence_table,
)
from src.statistics.association import calculate_correlation_matrix


def build_eda_tables(data: pd.DataFrame) -> dict:
    """Return quality, validation and named DataFrames for all 21 features.

    This stage requires the complete valid schema. No transformations, fitting,
    hypothesis tests or clinical conclusions are performed. Frame attrs record
    dataset_variant; export code also writes it as a column because CSV loses attrs.
    """
    validation = validate_dataset(data, EXPECTED_COLUMNS, "Diabetes_binary")
    if not validation["is_valid"]:
        raise DataValidationError("; ".join(validation["errors"]))
    tables = {
        "target_distribution": summarize_categorical_variable(data, "Diabetes_binary"),
        "numeric_summary": pd.DataFrame({c: summarize_numeric_variable(data, c) for c in ["BMI", "MentHlth", "PhysHlth"]}).T,
    }
    for feature in FEATURE_COLUMNS:
        tables[f"by_target_{feature}"] = compare_by_target(data, feature, "Diabetes_binary")
        if VARIABLE_METADATA[feature]["type"] in {"binary", "ordinal"}:
            tables[f"distribution_{feature}"] = summarize_categorical_variable(data, feature)
            tables[f"prevalence_{feature}"] = calculate_prevalence_table(data, feature, "Diabetes_binary")
    health = data.groupby("Diabetes_binary").agg(
        respondents=("MentHlth", "size"),
        mental_mean=("MentHlth", "mean"), mental_median=("MentHlth", "median"),
        mental_any_percent=("MentHlth", lambda s: s.gt(0).mean() * 100),
        mental_30_percent=("MentHlth", lambda s: s.eq(30).mean() * 100),
        physical_mean=("PhysHlth", "mean"), physical_median=("PhysHlth", "median"),
        physical_any_percent=("PhysHlth", lambda s: s.gt(0).mean() * 100),
        physical_30_percent=("PhysHlth", lambda s: s.eq(30).mean() * 100),
    )
    tables["health_days_by_target"] = health
    binary_rows = []
    for feature in FEATURE_COLUMNS:
        if VARIABLE_METADATA[feature]["type"] != "binary":
            continue
        group = tables[f"prevalence_{feature}"].set_index("category")
        no, yes = group.loc[0], group.loc[1]
        binary_rows.append({"feature": feature, "no_respondents": int(no.total_observations),
                            "yes_respondents": int(yes.total_observations),
                            "category_1_percent_no": no.target_percentage,
                            "category_1_percent_yes": yes.target_percentage,
                            "difference_percentage_points": yes.target_percentage - no.target_percentage})
    tables["binary_comparisons"] = pd.DataFrame(binary_rows)
    # Sex is nominal: exclude arbitrary sex-code ranking from the ordinal/numeric matrix.
    correlation_columns = ["BMI", "MentHlth", "PhysHlth", "Age", "Education", "Income", "GenHlth", "Diabetes_binary"]
    tables["spearman_correlation"] = calculate_correlation_matrix(data, correlation_columns)
    for table in tables.values():
        table.attrs.update(dataset_variant=data.attrs.get("dataset_variant", "unspecified"), target_definition=TARGET_DEFINITION)
    return {"quality": generate_data_quality_report(data), "validation": validation, "tables": tables}


def compare_dataset_variants(data: pd.DataFrame) -> dict:
    """Build full_clean_v1 and complete-row unique_profile_v1 EDA plus sensitivity.

    Sensitivity includes overall target share, numeric target-group means and all
    categorical conditional percentages. It is not proof of duplicate respondents
    or a replacement for the modeling team's feature-vector leakage controls.
    """
    variants = {}
    for name in ["full_clean_v1", "unique_profile_v1"]:
        frame = prepare_analysis_data(data, "Diabetes_binary", FEATURE_COLUMNS, dataset_variant=name)
        variants[name] = {"data": frame, **build_eda_tables(frame)}
    rows = []
    for name, bundle in variants.items():
        frame = bundle["data"]
        rows.append({"dataset_variant": name, "feature": "Diabetes_binary", "group": "All",
                     "metric": "category_1_percent", "estimate": frame.Diabetes_binary.mean() * 100, "n": len(frame)})
        for feature in ["BMI", "MentHlth", "PhysHlth"]:
            for code, group in frame.groupby("Diabetes_binary"):
                rows.append({"dataset_variant": name, "feature": feature, "group": f"Target {code}",
                             "metric": "mean", "estimate": group[feature].mean(), "n": len(group)})
        for feature in FEATURE_COLUMNS:
            table = bundle["tables"].get(f"prevalence_{feature}")
            if table is None:
                continue
            for row in table.itertuples():
                rows.append({"dataset_variant": name, "feature": feature, "group": row.label,
                             "metric": "category_1_percent", "estimate": row.target_percentage, "n": row.total_observations})
    long = pd.DataFrame(rows)
    keys = ["feature", "group", "metric"]
    full = long.loc[long.dataset_variant == "full_clean_v1"].drop(columns="dataset_variant")
    unique = long.loc[long.dataset_variant == "unique_profile_v1"].drop(columns="dataset_variant")
    sensitivity = full.merge(unique, on=keys, suffixes=("_full", "_unique"), validate="one_to_one")
    sensitivity["change_unique_minus_full"] = sensitivity.estimate_unique - sensitivity.estimate_full
    return {"variants": variants, "sensitivity": sensitivity}
