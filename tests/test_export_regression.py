"""Tests for src/analysis/export_regression.py on a small generated dataset (reference check disabled)."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.analysis.export_regression import bmi_odds_by_age, export_regression
from src.common.config import FEATURE_COLUMNS, VARIABLE_METADATA
from src.common.exceptions import DataValidationError
from src.modeling.regression import fit_logistic_regression

PREFIX = "full_clean_v1_regression_"
TABLES = ["baseline_coefficients", "interaction_coefficients", "baseline_odds_ratios",
          "model_comparison", "bmi_odds_by_age"]


def create_survey_csv(path, n_rows=3000, seed=0):
    """Schema-valid CDC-style CSV where the BMI effect grows with age; far smaller than the reference data."""
    rng = np.random.default_rng(seed)
    data = {}
    for column in FEATURE_COLUMNS:
        meta = VARIABLE_METADATA[column]
        if meta.get("allowed_values"):
            data[column] = rng.choice(meta["allowed_values"], n_rows)
        elif column == "BMI":
            data[column] = rng.integers(15, 51, n_rows)
        else:
            data[column] = rng.integers(meta["minimum"], meta["maximum"] + 1, n_rows)
    frame = pd.DataFrame(data)
    bmi, age = frame["BMI"] - 28, frame["Age"] - 7
    score = -1.5 + 0.05 * bmi + 0.15 * age + 0.02 * bmi * age + 0.8 * frame["HighBP"]
    frame["Diabetes_binary"] = rng.binomial(1, 1 / (1 + np.exp(-score)))
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("regression")
    csv = create_survey_csv(folder / "survey.csv")
    output = folder / "outputs"
    manifest = export_regression(str(csv), str(output), check_reference=False)
    return {"csv": csv, "output": output, "manifest": manifest}


def read_table(exported, name):
    return pd.read_csv(exported["output"] / "tables" / f"{PREFIX}{name}.csv")


# ---------------------------------------------------------------- files and manifest

def test_every_table_and_figure_is_written_and_listed(exported):
    files = exported["manifest"]["files"]

    assert files == ([f"tables/{PREFIX}{name}.csv" for name in TABLES]
                     + [f"figures/{PREFIX}{name}.png" for name in ["odds_ratios", "bmi_odds_by_age"]])
    assert all((exported["output"] / f).is_file() for f in files)
    for figure in files[-2:]:
        assert (exported["output"] / figure).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_manifest_file_matches_return_value(exported):
    saved = json.loads((exported["output"] / "regression_manifest.json").read_text())

    assert saved == exported["manifest"]


def test_manifest_records_data_and_method(exported):
    manifest = exported["manifest"]

    assert manifest["dataset_variant"] == "full_clean_v1"
    assert manifest["n"] == 3000
    assert manifest["dataset_version_sha256"] == hashlib.sha256(exported["csv"].read_bytes()).hexdigest()
    assert manifest["converged"] == {"baseline": True, "interaction": True}
    assert set(manifest["centering_means"]) == {"BMI", "Age"}
    assert "Age" in manifest["feature_treatment"]
    assert manifest["limitations"]


def test_manifest_centering_means_match_data(exported):
    data = pd.read_csv(exported["csv"])

    assert exported["manifest"]["centering_means"]["BMI"] == pytest.approx(data["BMI"].mean())
    assert exported["manifest"]["centering_means"]["Age"] == pytest.approx(data["Age"].mean())


def test_manifest_interaction_matches_coefficient_table(exported):
    table = read_table(exported, "interaction_coefficients").set_index("feature")
    term = exported["manifest"]["interaction_term"]

    assert term["term"] == "BMI:Age"
    assert term["odds_ratio"] == pytest.approx(table.loc["BMI:Age", "odds_ratio"])
    assert term["ci_lower"] < term["odds_ratio"] < term["ci_upper"]


def test_manifest_model_comparison_has_lr_test(exported):
    comparison = exported["manifest"]["model_comparison"]

    assert comparison["lr_df"] == 1
    assert comparison["lr_statistic"] > 0
    assert 0 <= comparison["lr_p_value"] <= 1


# ---------------------------------------------------------------- tables

def test_every_table_identifies_dataset_variant(exported):
    for name in TABLES:
        assert (read_table(exported, name)["dataset_variant"] == "full_clean_v1").all()


def test_coefficient_tables_have_expected_terms(exported):
    baseline = read_table(exported, "baseline_coefficients")
    interaction = read_table(exported, "interaction_coefficients")

    assert baseline["feature"].tolist() == ["const"] + FEATURE_COLUMNS
    assert interaction["feature"].tolist() == ["const"] + FEATURE_COLUMNS + ["BMI:Age"]


def test_baseline_odds_ratios_exclude_intercept_and_are_sorted(exported):
    table = read_table(exported, "baseline_odds_ratios")

    assert "const" not in table["feature"].tolist()
    assert len(table) == len(FEATURE_COLUMNS)
    assert table["odds_ratio"].is_monotonic_decreasing


def test_model_comparison_table_matches_manifest(exported):
    table = read_table(exported, "model_comparison").set_index("model")

    assert table.index.tolist() == ["baseline", "interaction"]
    assert (table["n"] == 3000).all()
    expected_delta = table.loc["baseline", "aic"] - table.loc["interaction", "aic"]
    assert exported["manifest"]["model_comparison"]["delta_aic"] == pytest.approx(expected_delta)


def test_interaction_detected_in_generated_data(exported):
    assert exported["manifest"]["model_comparison"]["lr_p_value"] < 0.001
    assert exported["manifest"]["interaction_term"]["coefficient"] > 0


def test_bmi_odds_table_covers_every_age_band(exported):
    table = read_table(exported, "bmi_odds_by_age")
    labels = VARIABLE_METADATA["Age"]["category_labels"]

    assert table["age_code"].tolist() == list(labels)
    assert table["age_group"].tolist() == list(labels.values())
    assert ((table["ci_lower"] < table["bmi_odds_ratio"]) & (table["bmi_odds_ratio"] < table["ci_upper"])).all()


def test_bmi_odds_rise_with_age_when_interaction_is_positive(exported):
    table = read_table(exported, "bmi_odds_by_age")

    assert table["bmi_odds_ratio"].is_monotonic_increasing


# ---------------------------------------------------------------- bmi_odds_by_age directly

def test_bmi_odds_at_mean_age_equals_bmi_main_effect():
    rng = np.random.default_rng(1)
    n = 2000
    data = pd.DataFrame({"BMI": rng.normal(0, 5, n), "Age": rng.integers(1, 14, n).astype(float)})
    data["Age"] = data["Age"] - 7.0
    score = 0.05 * data["BMI"] + 0.1 * data["Age"] + 0.01 * data["BMI"] * data["Age"]
    data["Diabetes_binary"] = rng.binomial(1, 1 / (1 + np.exp(-score)))
    model = fit_logistic_regression(data, "Diabetes_binary", ["BMI", "Age"], [("BMI", "Age")])

    table = bmi_odds_by_age(model, age_mean=7.0).set_index("age_code")

    bmi = model["results"].set_index("feature").loc["BMI"]
    assert table.loc[7, "bmi_odds_ratio"] == pytest.approx(bmi["odds_ratio"])
    assert table.loc[7, "ci_lower"] == pytest.approx(bmi["ci_lower"], rel=1e-6)
    assert table.loc[7, "ci_upper"] == pytest.approx(bmi["ci_upper"], rel=1e-6)


# ---------------------------------------------------------------- failures and CLI

def test_reference_check_rejects_small_dataset_before_writing(tmp_path):
    csv = create_survey_csv(tmp_path / "survey.csv", n_rows=300)
    output = tmp_path / "outputs"

    with pytest.raises(DataValidationError, match="reference"):
        export_regression(str(csv), str(output))

    assert not output.exists()


def test_missing_column_is_rejected(tmp_path):
    csv = create_survey_csv(tmp_path / "survey.csv", n_rows=300)
    pd.read_csv(csv).drop(columns=["Age"]).to_csv(csv, index=False)

    with pytest.raises(DataValidationError):
        export_regression(str(csv), str(tmp_path / "outputs"), check_reference=False)


def test_command_line_help_runs():
    result = subprocess.run([sys.executable, "-m", "src.analysis.export_regression", "--help"],
                            capture_output=True, text=True)

    assert result.returncode == 0
    assert "--csv" in result.stdout
