"""Tests for src/analysis/export_modeling.py on a small generated dataset (reference check disabled)."""
import hashlib
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from src.analysis.export_modeling import THRESHOLD_GRID, export_modeling
from src.common.config import FEATURE_COLUMNS, VARIABLE_METADATA
from src.common.exceptions import DataValidationError

MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]


def create_survey_csv(path, n_rows=600, seed=0):
    """Schema-valid CDC-style CSV with a learnable signal; far smaller than the reference data."""
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
    risk = 0.15 * frame["HighBP"] + 0.03 * frame["Age"] + rng.normal(0, 0.15, n_rows)
    frame["Diabetes_binary"] = (risk > np.quantile(risk, 0.8)).astype(int)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("modeling")
    csv = create_survey_csv(folder / "survey.csv")
    output = folder / "outputs"
    manifest = export_modeling(str(csv), str(output), check_reference=False, importance_sample=200)
    return {"csv": csv, "output": output, "manifest": manifest}


def read_table(exported, name):
    return pd.read_csv(exported["output"] / "tables" / f"{name}.csv")


# ---------------------------------------------------------------- files and manifest

def test_every_listed_file_exists(exported):
    files = exported["manifest"]["files"]

    assert len(files) == 27
    assert all((exported["output"] / f).is_file() for f in files)
    assert (exported["output"] / "modeling_manifest.json").is_file()


def test_figures_are_listed_png_files_named_by_variant(exported):
    figures = [f for f in exported["manifest"]["files"] if f.startswith("figures/")]

    assert figures == [f"figures/unique_profile_v1_{name}.png" for name in
                       ["roc_curves", "precision_recall_curves", "calibration_curves",
                        "threshold_tradeoffs", "feature_importance"]]
    for figure in figures:
        assert (exported["output"] / figure).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("model", MODELS)
def test_curve_tables_are_saved(exported, model):
    roc = read_table(exported, f"roc_curve_{model}")
    pr = read_table(exported, f"precision_recall_curve_{model}")

    assert list(roc.columns) == ["false_positive_rate", "true_positive_rate", "threshold"]
    assert list(pr.columns) == ["precision", "recall", "threshold"]
    assert roc["true_positive_rate"].iloc[-1] == 1


def test_manifest_records_split_and_dataset_version(exported):
    manifest = exported["manifest"]

    assert manifest["leakage_check"] == "PASS"
    assert manifest["dataset_variant"] == "unique_profile_v1"
    assert manifest["train_rows"] + manifest["test_rows"] == manifest["rows"]
    assert manifest["dataset_version_sha256"] == hashlib.sha256(exported["csv"].read_bytes()).hexdigest()


def test_manifest_file_matches_return_value(exported):
    saved = json.loads((exported["output"] / "modeling_manifest.json").read_text())

    assert saved == exported["manifest"]


# ---------------------------------------------------------------- tables

def test_model_comparison_has_one_row_per_baseline(exported):
    table = read_table(exported, "model_comparison")

    assert table["model"].tolist() == MODELS
    for column in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
                   "brier_score", "true_negatives", "false_positives", "false_negatives", "true_positives"]:
        assert column in table.columns


def test_imbalance_table_records_training_counts(exported):
    table = read_table(exported, "imbalance_experiments").set_index("strategy")

    assert table.index.tolist() == ["original", "class_weighted", "undersampling"]
    assert table["train_positives_before"].nunique() == 1
    assert table.loc["original", "train_negatives_after"] == table.loc["original", "train_negatives_before"]
    assert table.loc["undersampling", "train_negatives_after"] == table.loc["undersampling", "train_positives_after"]
    assert table.loc["undersampling", "train_positives_after"] == table.loc["undersampling", "train_positives_before"]


def test_imbalance_strategies_share_the_untouched_test_set(exported):
    table = read_table(exported, "imbalance_experiments")
    comparison = read_table(exported, "model_comparison")

    assert table["n"].nunique() == 1
    assert table["n"].iloc[0] == exported["manifest"]["test_rows"] == comparison["n"].iloc[0]
    positives = table["true_positives"] + table["false_negatives"]
    assert positives.nunique() == 1


def test_original_strategy_matches_logistic_baseline(exported):
    imbalance = read_table(exported, "imbalance_experiments").set_index("strategy")
    comparison = read_table(exported, "model_comparison").set_index("model")

    assert imbalance.loc["original", "roc_auc"] == pytest.approx(comparison.loc["logistic_regression", "roc_auc"])


@pytest.mark.parametrize("model", MODELS)
def test_threshold_table_covers_full_grid(exported, model):
    table = read_table(exported, f"thresholds_analysis_{model}")

    assert table["threshold"].tolist() == pytest.approx(THRESHOLD_GRID)
    assert THRESHOLD_GRID[0] == 0.10 and THRESHOLD_GRID[-1] == 0.90 and len(THRESHOLD_GRID) == 17


@pytest.mark.parametrize("model", MODELS)
def test_calibration_table_counts_every_test_row(exported, model):
    table = read_table(exported, f"calibration_table_{model}")

    assert table["count"].sum() == exported["manifest"]["test_rows"]
    assert table["observed_frequency"].between(0, 1).all()


@pytest.mark.parametrize("model", MODELS)
def test_feature_importance_lists_every_feature(exported, model):
    table = read_table(exported, f"feature_importance_{model}")

    assert sorted(table["feature"]) == sorted(FEATURE_COLUMNS)


# ---------------------------------------------------------------- metadata, metrics, models

def test_metadata_has_experiment_contract_fields(exported):
    metadata = json.loads((exported["output"] / "models" / "metadata.json").read_text())

    for key in ["timestamp", "dataset_version_sha256", "dataset_variant", "features", "random_state",
                "split", "threshold", "threshold_grid", "threshold_selection", "models", "versions"]:
        assert key in metadata
    assert metadata["split"]["leakage_check"]["status"] == "PASS"
    assert metadata["threshold_selection"].startswith("none")
    assert metadata["importance"]["sample_rows"] == min(200, exported["manifest"]["test_rows"])
    assert set(metadata["models"]) == set(MODELS)


def test_metadata_default_profile_is_valid_for_every_feature(exported):
    from src.analysis.profiles import build_risk_profile

    defaults = json.loads((exported["output"] / "models" / "metadata.json").read_text())["default_profile"]

    assert list(defaults) == FEATURE_COLUMNS
    for column in FEATURE_COLUMNS:
        if VARIABLE_METADATA[column]["allowed_values"]:
            assert defaults[column] in VARIABLE_METADATA[column]["allowed_values"]
    assert build_risk_profile(defaults, VARIABLE_METADATA)["defaulted_features"] == []


def test_metrics_json_matches_comparison_table(exported):
    metrics = json.loads((exported["output"] / "models" / "metrics.json").read_text())
    comparison = read_table(exported, "model_comparison").set_index("model")

    for model in MODELS:
        assert metrics["test_metrics"][model]["f1"] == pytest.approx(comparison.loc[model, "f1"])
    assert set(metrics["calibration"]) == set(MODELS)
    assert len(metrics["imbalance_experiments"]) == 3


@pytest.mark.parametrize("model", MODELS)
def test_saved_model_reloads_and_predicts(exported, model):
    loaded = joblib.load(exported["output"] / "models" / f"{model}.pkl")
    X = pd.read_csv(exported["csv"])[FEATURE_COLUMNS].head(20)

    probabilities = loaded.predict_proba(X)[:, 1]

    assert probabilities.shape == (20,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_export_is_reproducible(exported, tmp_path):
    export_modeling(str(exported["csv"]), str(tmp_path), check_reference=False, importance_sample=200)

    for name in ["model_comparison", "imbalance_experiments", "feature_importance_random_forest"]:
        first = read_table(exported, name)
        second = pd.read_csv(tmp_path / "tables" / f"{name}.csv")
        pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------- failures

def test_reference_check_rejects_small_dataset_before_writing(tmp_path):
    csv = create_survey_csv(tmp_path / "survey.csv", n_rows=200)
    output = tmp_path / "outputs"

    with pytest.raises(DataValidationError, match="reference"):
        export_modeling(str(csv), str(output))

    assert not output.exists()


def test_missing_column_is_rejected(tmp_path):
    csv = create_survey_csv(tmp_path / "survey.csv", n_rows=200)
    pd.read_csv(csv).drop(columns=["BMI"]).to_csv(csv, index=False)

    with pytest.raises(DataValidationError):
        export_modeling(str(csv), str(tmp_path / "outputs"), check_reference=False)


def test_command_line_help_runs():
    result = subprocess.run([sys.executable, "-m", "src.analysis.export_modeling", "--help"],
                            capture_output=True, text=True)

    assert result.returncode == 0
    assert "--csv" in result.stdout
