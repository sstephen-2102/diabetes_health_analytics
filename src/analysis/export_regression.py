"""Run from the project root: python -m src.analysis.export_regression --csv PATH."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from scipy import stats

from src.common.config import EXPECTED_COLUMNS, FEATURE_COLUMNS, PROJECT_CONFIG, TARGET_DEFINITION, VARIABLE_METADATA
from src.common.exceptions import DataValidationError, ModelTrainingError
from src.data.loader import load_dataset
from src.data.preprocessing import prepare_analysis_data
from src.data.validation import validate_dataset
from src.modeling.regression import compare_regression_models, extract_odds_ratios, fit_logistic_regression
from src.analysis.model_figures import bmi_by_age_figure, odds_ratio_figure

TARGET = PROJECT_CONFIG["target_column"]
VARIANT = PROJECT_CONFIG["descriptive_dataset_variant"]
CONFIDENCE = PROJECT_CONFIG["confidence_level"]
INTERACTION = ("BMI", "Age")
TERM = "BMI:Age"
FEATURE_TREATMENT = {
    "BMI": "continuous; mean-centered",
    "Age": "13-level ordinal code used as a numeric score (1 unit = one age band); mean-centered",
    "GenHlth, Education, Income": "ordinal codes used as numeric scores (linear on the log-odds scale)",
    "MentHlth, PhysHlth": "day counts 0-30 used as numeric",
    "other features": "binary 0/1 indicators as recorded",
}
LIMITATIONS = [
    "Associations, not causal effects; survey responses are self-reported.",
    "Survey weights and design are not applied; estimates describe this sample.",
    "With n above 250,000 almost every p-value is tiny; judge size by odds ratios and CIs.",
    "Ordinal codes are assumed to have a linear relationship with the log-odds.",
    "full_clean_v1 keeps repeated complete rows, which can understate standard errors.",
]

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot write {type(value).__name__} to JSON")


def bmi_odds_by_age(interaction_model: dict, age_mean: float) -> pd.DataFrame:
    """BMI odds ratio (per 1 BMI unit) within each age band, with CIs, from the interaction model."""
    coefficients = interaction_model["results"].set_index("feature")["coefficient"]
    covariance = interaction_model["covariance"]
    z = stats.norm.ppf(0.5 + CONFIDENCE / 2)
    rows = []
    for code, label in VARIABLE_METADATA["Age"]["category_labels"].items():
        offset = code - age_mean
        log_odds = coefficients["BMI"] + offset * coefficients[TERM]
        std_error = np.sqrt(covariance.loc["BMI", "BMI"] + offset ** 2 * covariance.loc[TERM, TERM]
                            + 2 * offset * covariance.loc["BMI", TERM])
        rows.append({"age_code": code, "age_group": label, "bmi_log_odds": log_odds, "std_error": std_error,
                     "bmi_odds_ratio": np.exp(log_odds),
                     "ci_lower": np.exp(log_odds - z * std_error), "ci_upper": np.exp(log_odds + z * std_error)})
    return pd.DataFrame(rows)

def export_regression(csv_path: str, output_directory: str = "outputs", check_reference: bool = True) -> dict:
    """Fit baseline and BMI x Age logistic regressions on full_clean_v1; export tables and a manifest.

    BMI and Age are mean-centered so main effects describe an average-BMI, average-age respondent.
    check_reference=False is the explicit opt-out for fixtures/new dataset versions.
    """
    # 1. Load, validate, build the statistics dataset and center BMI/Age
    raw = load_dataset(csv_path)
    validation = validate_dataset(raw, EXPECTED_COLUMNS, TARGET)
    if not validation["is_valid"]:
        raise DataValidationError("; ".join(validation["errors"]))
    if check_reference and not validation["matches_reference"]:
        raise DataValidationError("Full source differs from reference counts; review data before regression.")
    data = prepare_analysis_data(raw, TARGET, FEATURE_COLUMNS, dataset_variant=VARIANT)
    means = {column: float(data[column].mean()) for column in INTERACTION}
    centered = data.assign(**{column: data[column] - means[column] for column in INTERACTION})
    centered.attrs.update(data.attrs)

    # 2. Fit both specifications on the same rows
    baseline = fit_logistic_regression(centered, TARGET, FEATURE_COLUMNS)
    interaction = fit_logistic_regression(centered, TARGET, FEATURE_COLUMNS, [INTERACTION])
    if not (baseline["converged"] and interaction["converged"]):
        raise ModelTrainingError("Logistic regression did not converge; nothing exported.")
    comparison = compare_regression_models(baseline, interaction)

    # 3. Tables
    output = Path(output_directory)
    tables_dir, figures_dir = output / "tables", output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    files = []

    def save_table(table, name):
        path = tables_dir / f"{VARIANT}_regression_{name}.csv"
        table.assign(dataset_variant=VARIANT).to_csv(path, index=False)
        files.append(str(path.relative_to(output)))

    def save_figure(fig, name):
        fig.text(0.99, 0, f"dataset: {VARIANT}", ha="right", va="top", fontsize=8, color="dimgray")
        path = figures_dir / f"{VARIANT}_regression_{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        files.append(str(path.relative_to(output)))

    odds_ratios = extract_odds_ratios(baseline).query("feature != 'const'")
    bmi_by_age = bmi_odds_by_age(interaction, means["Age"])
    save_table(baseline["results"], "baseline_coefficients")
    save_table(interaction["results"], "interaction_coefficients")
    save_table(odds_ratios, "baseline_odds_ratios")
    save_table(pd.DataFrame([{"model": name, **{k: comparison[name][k] for k in
                                                ["n", "log_likelihood", "aic", "bic", "pseudo_r2"]}}
                            for name in ["baseline", "interaction"]]), "model_comparison")
    save_table(bmi_by_age, "bmi_odds_by_age")
    save_figure(odds_ratio_figure(odds_ratios), "odds_ratios")
    save_figure(bmi_by_age_figure(bmi_by_age), "bmi_odds_by_age")

    # 4. Manifest
    term = interaction["results"].set_index("feature").loc[TERM]
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": Path(csv_path).name,
        "dataset_version_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "dataset_variant": VARIANT,
        "n": baseline["n"],
        "target_definition": TARGET_DEFINITION,
        "features": FEATURE_COLUMNS,
        "feature_treatment": FEATURE_TREATMENT,
        "centering_means": means,
        "interaction_term": {"term": TERM, "coefficient": term["coefficient"], "odds_ratio": term["odds_ratio"],
                             "ci_lower": term["ci_lower"], "ci_upper": term["ci_upper"], "p_value": term["p_value"]},
        "model_comparison": {k: comparison[k] for k in ["delta_aic", "delta_bic", "lr_statistic", "lr_df", "lr_p_value"]},
        "converged": {"baseline": baseline["converged"], "interaction": interaction["converged"]},
        "confidence_level": CONFIDENCE,
        "limitations": LIMITATIONS,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "statsmodels": statsmodels.__version__},
        "files": files,
    }
    (output / "regression_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default),
                                                     encoding="utf-8")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    manifest = export_regression(args.csv, args.output)
    print(f"Exported {len(manifest['files'])} regression files to {args.output} "
          f"(interaction p = {manifest['interaction_term']['p_value']:.2e})")
