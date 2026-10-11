"""Export Member 2 results (H1-H7, Bayes verification, selected probabilities and confidence intervals)
for both dataset variants.

Usage from the repository root:
    python -m src.analysis.export_member2 --csv data/raw/cdc_diabetes.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from src.common.config import EXPECTED_COLUMNS, FEATURE_COLUMNS
from src.data.loader import load_dataset
from src.data.preprocessing import prepare_analysis_data
from src.data.validation import validate_dataset
from src.statistics.bayes_suite import run_bayes_suite
from src.statistics.confidence_intervals import (difference_in_means_ci, mean_confidence_interval,
                                                 proportion_confidence_interval)
from src.statistics.hypothesis_suite import run_hypothesis_suite
from src.statistics.probability import calculate_conditional_probability, calculate_joint_probability

TARGET = "Diabetes_binary"
PROBABILITY_CONDITIONS = [
    {"HighBP": 1}, {"HighChol": 1}, {"PhysActivity": 1}, {"HighBP": 1, "HighChol": 1},
    {"HighBP": 0, "HighChol": 0, "PhysActivity": 1}, {"Sex": 1}, {"Age": 11}, {"Income": 1}, {"Income": 8},
]
PROPORTION_FEATURES = ["HighBP", "HighChol", "PhysActivity", "Smoker", "GenHlth"]
MEAN_FEATURES = ["BMI", "MentHlth", "PhysHlth"]


def run_probability_table(data: pd.DataFrame, conditions_list: list[dict] = PROBABILITY_CONDITIONS) -> pd.DataFrame:
    """Marginal, joint and conditional probability of category 1 for each set of conditions (Wilson 95% CI)."""
    total, positives = len(data), int((data[TARGET] == 1).sum())
    marginal = proportion_confidence_interval(positives, total)
    rows = [{"conditions": "none (all respondents)", "p_conditions": 1.0, "p_conditions_and_category1":
             positives / total, "numerator": positives, "denominator": total, "p_category1_given_conditions":
             marginal["estimate"], "ci_lower": marginal["lower"], "ci_upper": marginal["upper"]}]
    for conditions in conditions_list:
        conditional = calculate_conditional_probability(data, TARGET, 1, list(conditions), list(conditions.values()))
        interval = conditional["confidence_interval"] or {"lower": None, "upper": None}
        rows.append({
            "conditions": " & ".join(f"{column}={value}" for column, value in conditions.items()),
            "p_conditions": calculate_joint_probability(data, conditions)["probability"],
            "p_conditions_and_category1": calculate_joint_probability(data, {**conditions, TARGET: 1})["probability"],
            "numerator": conditional["numerator"], "denominator": conditional["denominator"],
            "p_category1_given_conditions": conditional["probability"],
            "ci_lower": interval["lower"], "ci_upper": interval["upper"]})
    table = pd.DataFrame(rows)
    table.insert(1, "dataset_variant", data.attrs.get("dataset_variant", "unspecified"))
    return table


def run_interval_table(data: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:
    """Wilson intervals for the category 1 share overall and by category of selected features; t intervals for
    selected means by target group and Welch intervals for their differences (category 1 minus category 0)."""
    rows = []
    def add(kind, feature, group, interval, estimate, method):
        rows.append({"estimate_type": kind, "feature": feature, "group": group, "n": interval.get("n"),
                     "estimate": estimate, "ci_lower": interval["lower"], "ci_upper": interval["upper"],
                     "method": method, "confidence_level": confidence_level})
    groups = [("all", "all respondents", data)] + [(feature, code, group) for feature in PROPORTION_FEATURES
                                                   for code, group in data.groupby(feature)]
    for feature, label, group in groups:
        interval = proportion_confidence_interval(int((group[TARGET] == 1).sum()), len(group), confidence_level)
        add("category_1_share", feature, label, {**interval, "n": interval["total"]}, interval["estimate"], "wilson")
    for feature in MEAN_FEATURES:
        values = {code: data.loc[data[TARGET] == code, feature] for code in (1, 0)}
        for code, series in values.items():
            interval = mean_confidence_interval(series, confidence_level)
            add("mean", feature, f"category {code}", interval, interval["mean"], interval["method"])
        difference = difference_in_means_ci(values[1], values[0], confidence_level)
        add("mean_difference", feature, "category 1 minus category 0",
            {**difference, "n": difference["n_a"] + difference["n_b"]}, difference["difference"], difference["method"])
    table = pd.DataFrame(rows)
    table.insert(1, "dataset_variant", data.attrs.get("dataset_variant", "unspecified"))
    return table


def export_member2(csv_path: str, output_directory: str = "outputs") -> dict:
    """Validate the CSV, run the suites and tables on full_clean_v1 and unique_profile_v1, write four CSVs."""
    raw = load_dataset(csv_path)
    checks = validate_dataset(raw, EXPECTED_COLUMNS, TARGET)
    if not checks["is_valid"]:
        raise ValueError(checks["errors"])
    samples = [
        prepare_analysis_data(raw, TARGET, FEATURE_COLUMNS),
        prepare_analysis_data(raw, TARGET, FEATURE_COLUMNS, dataset_variant="unique_profile_v1"),
    ]
    hypotheses = pd.concat([run_hypothesis_suite(sample) for sample in samples], ignore_index=True)
    bayes = pd.concat([run_bayes_suite(sample) for sample in samples], ignore_index=True)
    probabilities = pd.concat([run_probability_table(sample) for sample in samples], ignore_index=True)
    intervals = pd.concat([run_interval_table(sample) for sample in samples], ignore_index=True)

    table_directory = Path(output_directory) / "tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    paths = {"hypotheses": table_directory / "member2_hypothesis_results.csv",
             "bayes": table_directory / "member2_bayes_results.csv",
             "probabilities": table_directory / "member2_probability_results.csv",
             "intervals": table_directory / "member2_confidence_intervals.csv"}
    hypotheses.round(4).to_csv(paths["hypotheses"], index=False)
    bayes.to_csv(paths["bayes"], index=False)
    probabilities.round(6).to_csv(paths["probabilities"], index=False)
    intervals.round(6).to_csv(paths["intervals"], index=False)
    return {name: str(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to cdc_diabetes.csv")
    parser.add_argument("--output-directory", default="outputs")
    arguments = parser.parse_args()
    for name, path in export_member2(arguments.csv, arguments.output_directory).items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
