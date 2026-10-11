"""Regenerate every exported output (tables, figures, models) in one run.

Usage from the repository root:
    python -m src.analysis.pipeline --csv data/raw/cdc_diabetes.csv
"""
import argparse

from src.analysis.export_duplicate_sensitivity import export_duplicate_sensitivity
from src.analysis.export_eda import export_eda
from src.analysis.export_member2 import export_member2
from src.analysis.export_modeling import export_modeling
from src.analysis.export_regression import export_regression
from src.common.exceptions import InvalidParameterError
from src.data.loader import download_uci_dataset

STEPS = {
    "eda": lambda csv, output, check: export_eda(csv, output, check_reference=check),
    "statistics": lambda csv, output, check: export_member2(csv, output),
    "regression": lambda csv, output, check: export_regression(csv, output, check_reference=check),
    "modeling": lambda csv, output, check: export_modeling(csv, output, check_reference=check),
    "duplicate_sensitivity": lambda csv, output, check: export_duplicate_sensitivity(csv, output,
                                                                                     check_reference=check),
}


def run_analysis_pipeline(csv_path: str, output_directory: str = "outputs", check_reference: bool = True,
                          steps: list[str] | None = None) -> dict:
    """Run the exporters in order and return each step's result, keyed by step name.

    Downloads the verified UCI dataset to csv_path first when it is missing. The statistics exporter has no
    reference check of its own; check_reference=False is for fixtures and new dataset versions.
    """
    steps = list(STEPS) if steps is None else steps
    unknown = [step for step in steps if step not in STEPS]
    if not steps or unknown:
        raise InvalidParameterError(f"steps must be a non-empty subset of {list(STEPS)}; got {steps}.")
    csv_path = download_uci_dataset(csv_path)
    return {step: STEPS[step](csv_path, output_directory, check_reference) for step in steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Path to cdc_diabetes.csv (downloaded if missing)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), help="Run only these steps, in this order")
    arguments = parser.parse_args()
    for step in run_analysis_pipeline(arguments.csv, arguments.output, steps=arguments.steps):
        print(f"Finished {step}")
    print(f"Outputs written to {arguments.output}/")


if __name__ == "__main__":
    main()
