"""Tests for the Member 3 service functions in app/services.py."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from app.services import get_model_comparison, get_prediction_result, load_model_artifacts
from src.common.config import FEATURE_COLUMNS, TARGET_DEFINITION, VARIABLE_METADATA
from src.common.exceptions import DataLoadError, DataValidationError
from src.modeling.classification import evaluate_classifier, generate_predictions, train_logistic_classifier

USER = {"HighBP": 1, "HighChol": 1, "BMI": 34, "Smoker": 0, "PhysActivity": 0, "GenHlth": 4, "Sex": 1, "Age": 10}


def create_training_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.choice(VARIABLE_METADATA[c]["allowed_values"], n) if VARIABLE_METADATA[c]["allowed_values"]
                      else rng.integers(15 if c == "BMI" else 0, 50 if c == "BMI" else 31, n) for c in FEATURE_COLUMNS})
    y = pd.Series(((X["HighBP"] + X["GenHlth"] / 2 + rng.normal(0, 0.5, n)) > 2.5).astype(int))
    return X, y


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """A models folder shaped like export_modeling's output, with one small logistic model."""
    folder = tmp_path_factory.mktemp("models")
    X, y = create_training_data()
    model = train_logistic_classifier(X, y)
    joblib.dump(model, folder / "logistic_regression.pkl")
    predicted = generate_predictions(model, X)
    metrics = {"logistic_regression": evaluate_classifier(y, predicted["predictions"], predicted["probabilities"])}
    defaults = {c: int(X[c].mode().iloc[0]) if VARIABLE_METADATA[c]["allowed_values"] else float(X[c].median())
                for c in FEATURE_COLUMNS}
    metadata = {"timestamp": "2026-10-09T00:00:00+00:00", "dataset_variant": "unique_profile_v1",
                "dataset_version_sha256": "abc123", "default_profile": defaults,
                "models": {"logistic_regression": {"artifact": "models/logistic_regression.pkl"}}}
    (folder / "metadata.json").write_text(json.dumps(metadata, default=float))
    (folder / "metrics.json").write_text(json.dumps({"test_metrics": metrics}, default=float))
    return folder


# ---------------------------------------------------------------- load_model_artifacts

def test_load_model_artifacts_returns_models_metadata_and_metrics(artifacts):
    loaded = load_model_artifacts(str(artifacts))

    assert list(loaded["models"]) == ["logistic_regression"]
    assert hasattr(loaded["models"]["logistic_regression"], "predict_proba")
    assert loaded["metadata"]["dataset_variant"] == "unique_profile_v1"
    assert "logistic_regression" in loaded["metrics"]["test_metrics"]


def test_load_model_artifacts_missing_folder_explains_how_to_fix(tmp_path):
    with pytest.raises(DataLoadError, match="export_modeling"):
        load_model_artifacts(str(tmp_path / "missing"))


def test_load_model_artifacts_missing_model_file(artifacts, tmp_path):
    for name in ["metadata.json", "metrics.json"]:
        (tmp_path / name).write_text((artifacts / name).read_text())

    with pytest.raises(DataLoadError):
        load_model_artifacts(str(tmp_path))


def test_load_model_artifacts_corrupt_json(tmp_path):
    (tmp_path / "metadata.json").write_text("{not json")
    (tmp_path / "metrics.json").write_text("{}")

    with pytest.raises(DataLoadError):
        load_model_artifacts(str(tmp_path))


# ---------------------------------------------------------------- get_model_comparison

def test_model_comparison_accepts_saved_metrics_and_orders_columns(artifacts):
    metrics = load_model_artifacts(str(artifacts))["metrics"]["test_metrics"]

    table = get_model_comparison(metrics)

    assert list(table.columns[:8]) == ["accuracy", "precision", "recall", "specificity", "f1",
                                       "roc_auc", "pr_auc", "brier_score"]
    assert {"true_negatives", "false_positives", "false_negatives", "true_positives"} <= set(table.columns)
    assert list(table.index) == ["logistic_regression"]


def test_model_comparison_keeps_input_order_without_ranking(artifacts):
    metrics = load_model_artifacts(str(artifacts))["metrics"]["test_metrics"]["logistic_regression"]
    worse = {**metrics, "roc_auc": 0.5}

    table = get_model_comparison({"worse": worse, "better": metrics})

    assert list(table.index) == ["worse", "better"]


def test_model_comparison_rejects_incomplete_metrics():
    with pytest.raises(DataValidationError):
        get_model_comparison({"model": {"accuracy": 0.9}})


# ---------------------------------------------------------------- get_prediction_result

def test_prediction_result_follows_service_contract(artifacts):
    loaded = load_model_artifacts(str(artifacts))

    result = get_prediction_result(loaded["models"]["logistic_regression"], USER, 0.3, loaded["metadata"])

    assert result["status"] == "success"
    assert set(result) == {"status", "results", "warnings", "dataset_variant", "target_definition", "metadata"}
    assert result["dataset_variant"] == "unique_profile_v1"
    assert result["target_definition"] == TARGET_DEFINITION
    assert result["metadata"] == {"model_type": "LogisticRegression", "trained_at": "2026-10-09T00:00:00+00:00",
                                  "dataset_version_sha256": "abc123"}


def test_prediction_result_contains_probability_profile_and_defaults(artifacts):
    loaded = load_model_artifacts(str(artifacts))

    results = get_prediction_result(loaded["models"]["logistic_regression"], USER, 0.3, loaded["metadata"])["results"]

    assert 0 <= results["probability"] <= 1
    assert results["threshold"] == 0.3
    assert results["classification"] == int(results["probability"] >= 0.3)
    assert results["profile"]["High blood pressure"] == "Yes"
    assert len(results["defaulted_features"]) == 13


def test_prediction_result_warns_with_disclaimer_and_defaulted_inputs(artifacts):
    loaded = load_model_artifacts(str(artifacts))

    warnings = get_prediction_result(loaded["models"]["logistic_regression"], USER, 0.5, loaded["metadata"])["warnings"]

    assert "not a medical diagnosis" in warnings[0]
    assert "typical training-set values" in warnings[1]
    assert "Income" in warnings[1]


def test_prediction_result_with_all_inputs_has_no_default_warning(artifacts):
    loaded = load_model_artifacts(str(artifacts))
    everything = {**loaded["metadata"]["default_profile"], **USER}

    result = get_prediction_result(loaded["models"]["logistic_regression"], everything, 0.5, loaded["metadata"])

    assert result["results"]["defaulted_features"] == []
    assert len(result["warnings"]) == 1


def test_prediction_result_without_metadata_requires_all_inputs(artifacts):
    model = load_model_artifacts(str(artifacts))["models"]["logistic_regression"]

    with pytest.raises(DataValidationError, match="No value or default"):
        get_prediction_result(model, USER, 0.5)


def test_prediction_result_rejects_invalid_input(artifacts):
    loaded = load_model_artifacts(str(artifacts))

    with pytest.raises(DataValidationError, match="Age"):
        get_prediction_result(loaded["models"]["logistic_regression"], {**USER, "Age": 99}, 0.5, loaded["metadata"])
