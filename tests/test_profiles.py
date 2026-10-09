"""Tests for src/analysis/profiles.py: analysis profiles and Prediction Explorer inputs/outputs."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.profiles import (CLASSIFICATION_LABEL, DISCLAIMER, PROBABILITY_LABEL, AnalysisProfile,
                                   build_risk_profile, create_analysis_profile, predict_risk_profile)
from src.common.config import FEATURE_COLUMNS, VARIABLE_METADATA
from src.common.exceptions import DataValidationError, InvalidParameterError
from src.modeling.classification import train_logistic_classifier

FORM_INPUTS = [c for c in FEATURE_COLUMNS if VARIABLE_METADATA[c]["prediction_input"]]
OPTIONAL = [c for c in FEATURE_COLUMNS if c not in FORM_INPUTS]
USER = {"HighBP": 1, "HighChol": 1, "BMI": 34, "Smoker": 0, "PhysActivity": 0, "GenHlth": 4, "Sex": 1, "Age": 10}
DEFAULTS = {c: (VARIABLE_METADATA[c]["allowed_values"][0] if VARIABLE_METADATA[c]["allowed_values"] else 0.0)
            for c in FEATURE_COLUMNS}
DEFAULTS["BMI"] = 27.0


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({c: rng.choice(VARIABLE_METADATA[c]["allowed_values"], n) if VARIABLE_METADATA[c]["allowed_values"]
                      else rng.integers(15 if c == "BMI" else 0, 50 if c == "BMI" else 31, n) for c in FEATURE_COLUMNS})
    y = pd.Series(((X["HighBP"] + X["GenHlth"] / 2 + rng.normal(0, 0.5, n)) > 2.5).astype(int))
    return train_logistic_classifier(X, y)


def test_form_inputs_match_config():
    assert FORM_INPUTS == ["HighBP", "HighChol", "BMI", "Smoker", "PhysActivity", "GenHlth", "Sex", "Age"]


# ---------------------------------------------------------------- create_analysis_profile

def test_create_analysis_profile_returns_dataclass():
    profile = create_analysis_profile("Diabetes_binary", ["BMI", "Age"], 0.9)

    assert profile == AnalysisProfile("Diabetes_binary", ["BMI", "Age"], 0.9)


def test_create_analysis_profile_copies_feature_list():
    features = ["BMI"]
    profile = create_analysis_profile("Diabetes_binary", features)
    features.append("Age")

    assert profile.features == ["BMI"]


@pytest.mark.parametrize("target, features, level, message", [
    ("", ["BMI"], 0.95, "target_column"),
    ("Diabetes_binary", [], 0.95, "non-empty list"),
    ("Diabetes_binary", "BMI", 0.95, "non-empty list"),
    ("Diabetes_binary", ["BMI", "BMI"], 0.95, "duplicates"),
    ("Diabetes_binary", ["BMI", "Diabetes_binary"], 0.95, "leakage"),
    ("Diabetes_binary", ["BMI"], 1.0, "between 0 and 1"),
    ("Diabetes_binary", ["BMI"], 0, "between 0 and 1"),
    ("Diabetes_binary", ["BMI"], True, "between 0 and 1"),
])
def test_create_analysis_profile_rejects_bad_settings(target, features, level, message):
    with pytest.raises(InvalidParameterError, match=message):
        create_analysis_profile(target, features, level)


# ---------------------------------------------------------------- build_risk_profile

def test_build_profile_fills_optional_features_from_defaults():
    profile = build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS)

    assert set(profile["values"]) == set(FEATURE_COLUMNS)
    assert profile["defaulted_features"] == [c for c in VARIABLE_METADATA if c in OPTIONAL]
    assert profile["values"]["BMI"] == 34.0
    assert profile["values"]["Stroke"] == DEFAULTS["Stroke"]


def test_build_profile_uses_supplied_optional_values_over_defaults():
    profile = build_risk_profile({**USER, "Income": 3, "MentHlth": 5}, VARIABLE_METADATA, DEFAULTS)

    assert profile["values"]["Income"] == 3
    assert profile["values"]["MentHlth"] == 5.0
    assert "Income" not in profile["defaulted_features"]


def test_build_profile_with_every_feature_needs_no_defaults():
    profile = build_risk_profile({**DEFAULTS, **USER}, VARIABLE_METADATA)

    assert profile["defaulted_features"] == []


def test_build_profile_returns_ints_for_codes_and_floats_for_numbers():
    values = build_risk_profile({**USER, "HighBP": 1.0}, VARIABLE_METADATA, DEFAULTS)["values"]

    assert type(values["HighBP"]) is int
    assert type(values["GenHlth"]) is int
    assert type(values["BMI"]) is float


def test_build_profile_display_uses_readable_labels():
    display = build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS)["display"]

    assert display["High blood pressure"] == "Yes"
    assert display[VARIABLE_METADATA["GenHlth"]["label"]] == "Fair"
    assert display["Body mass index"] == 34.0


def test_build_profile_accepts_numpy_numbers():
    profile = build_risk_profile({**USER, "Age": np.int64(10), "BMI": np.float64(31.5)}, VARIABLE_METADATA, DEFAULTS)

    assert profile["values"]["Age"] == 10
    assert profile["values"]["BMI"] == 31.5


def test_build_profile_requires_every_form_input():
    partial = {k: v for k, v in USER.items() if k != "BMI"}

    with pytest.raises(DataValidationError, match=r"Missing required inputs: \['BMI'\]"):
        build_risk_profile(partial, VARIABLE_METADATA, DEFAULTS)


def test_build_profile_without_defaults_needs_optional_features():
    with pytest.raises(DataValidationError, match="No value or default"):
        build_risk_profile(USER, VARIABLE_METADATA)


def test_build_profile_rejects_unknown_feature_names():
    with pytest.raises(DataValidationError, match="Unknown features: \\['Bmi'\\]"):
        build_risk_profile({**USER, "Bmi": 30}, VARIABLE_METADATA, DEFAULTS)


def test_build_profile_rejects_the_target_as_input():
    with pytest.raises(DataValidationError, match="Unknown features"):
        build_risk_profile({**USER, "Diabetes_binary": 1}, VARIABLE_METADATA, DEFAULTS)


@pytest.mark.parametrize("change, message", [
    ({"HighBP": 2}, "one of"),
    ({"HighBP": 0.5}, "one of"),
    ({"Age": 14}, "one of"),
    ({"HighBP": True}, "must be a number"),
    ({"BMI": "34"}, "must be a number"),
    ({"BMI": float("nan")}, "must be a number"),
    ({"BMI": None}, "must be a number"),
    ({"BMI": 0}, "greater than 0"),
    ({"BMI": -5}, "at least"),
    ({"MentHlth": 31}, "at most"),
    ({"MentHlth": 2.5}, "whole number"),
])
def test_build_profile_rejects_invalid_values(change, message):
    with pytest.raises(DataValidationError, match=message):
        build_risk_profile({**USER, **change}, VARIABLE_METADATA, DEFAULTS)


def test_build_profile_validates_defaults_too():
    with pytest.raises(DataValidationError, match="Stroke"):
        build_risk_profile(USER, VARIABLE_METADATA, {**DEFAULTS, "Stroke": 7})


@pytest.mark.parametrize("values, metadata, defaults, error", [
    ("not a dict", VARIABLE_METADATA, None, DataValidationError),
    (USER, {}, None, InvalidParameterError),
    (USER, VARIABLE_METADATA, ["not", "a", "dict"], InvalidParameterError),
])
def test_build_profile_rejects_bad_containers(values, metadata, defaults, error):
    with pytest.raises(error):
        build_risk_profile(values, metadata, defaults)


# ---------------------------------------------------------------- predict_risk_profile

def test_predict_returns_probability_classification_and_spec_wording(model):
    result = predict_risk_profile(model, build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS), 0.5)

    assert 0 <= result["probability"] <= 1
    assert result["classification"] == int(result["probability"] >= 0.5)
    assert result["probability_label"] == PROBABILITY_LABEL
    assert result["classification_label"] == CLASSIFICATION_LABEL
    assert result["disclaimer"] == DISCLAIMER
    assert "not a medical diagnosis" in result["disclaimer"]


def test_predict_never_uses_diagnostic_language(model):
    profile = build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS)
    for threshold in (0.0, 1.0):
        text = " ".join(str(v) for v in predict_risk_profile(model, profile, threshold).values()).lower()
        assert "you have" not in text
        assert "diagnosed" not in text


def test_predict_matches_model_predict_proba(model):
    profile = build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS)
    row = pd.DataFrame([profile["values"]])[list(model.feature_names_in_)]

    result = predict_risk_profile(model, profile)

    assert result["probability"] == pytest.approx(model.predict_proba(row)[0, 1])


def test_predict_threshold_controls_classification(model):
    profile = build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS)

    assert predict_risk_profile(model, profile, 0.0)["classification"] == 1
    assert predict_risk_profile(model, profile, 1.0)["classification"] == int(
        predict_risk_profile(model, profile)["probability"] >= 1.0)


def test_predict_is_independent_of_input_order(model):
    profile = build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS)
    reversed_profile = {**profile, "values": dict(reversed(list(profile["values"].items())))}

    assert predict_risk_profile(model, reversed_profile)["probability"] == pytest.approx(
        predict_risk_profile(model, profile)["probability"])


def test_predict_rejects_profile_that_does_not_match_model(model):
    profile = build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS)
    values = {k: v for k, v in profile["values"].items() if k != "Income"}

    with pytest.raises(DataValidationError, match=r"missing \['Income'\]"):
        predict_risk_profile(model, {**profile, "values": values})


@pytest.mark.parametrize("threshold", [-0.1, 1.1, "0.5", True, float("nan")])
def test_predict_rejects_bad_threshold(model, threshold):
    profile = build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS)

    with pytest.raises(InvalidParameterError, match="threshold"):
        predict_risk_profile(model, profile, threshold)


def test_predict_rejects_unfitted_or_wrong_model():
    profile = build_risk_profile(USER, VARIABLE_METADATA, DEFAULTS)

    with pytest.raises(InvalidParameterError, match="fitted classifier"):
        predict_risk_profile(object(), profile)


@pytest.mark.parametrize("bad_profile", [None, {}, {"values": [1, 2]}])
def test_predict_rejects_raw_dict_instead_of_profile(model, bad_profile):
    with pytest.raises(DataValidationError, match="build_risk_profile"):
        predict_risk_profile(model, bad_profile)
