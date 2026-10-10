"""Export Member 2 results (H1-H7 and Bayes verification) for both dataset variants.

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
from src.statistics.hypothesis_suite import run_hypothesis_suite

TARGET = "Diabetes_binary"


def export_member2(csv_path: str, output_directory: str = "outputs") -> dict:
    """Validate the CSV, run both suites on full_clean_v1 and unique_profile_v1, write two CSVs."""
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

    table_directory = Path(output_directory) / "tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    paths = {"hypotheses": table_directory / "member2_hypothesis_results.csv",
             "bayes": table_directory / "member2_bayes_results.csv"}
    hypotheses.round(4).to_csv(paths["hypotheses"], index=False)
    bayes.to_csv(paths["bayes"], index=False)
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
