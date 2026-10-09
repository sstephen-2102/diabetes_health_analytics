"""Analysis profiles and Prediction Explorer inputs/outputs. Results are model estimates, never diagnoses."""
import math
import numbers
from dataclasses import dataclass

import pandas as pd

from src.common.exceptions import DataValidationError, InvalidParameterError

PROBABILITY_LABEL = "Model-estimated probability of the positive Diabetes_binary class"
CLASSIFICATION_LABEL = "Model classification at selected threshold"
DISCLAIMER = ("This output is a statistical/ML estimate from the project dataset "
              "and is not a medical diagnosis.")


@dataclass
class AnalysisProfile:
    """Configuration describing a reproducible analysis."""
    target: str
    features: list[str]
    confidence_level: float = 0.95


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and not math.isnan(value)


def create_analysis_profile(target_column: str, feature_columns: list[str], confidence_level: float = 0.95) -> AnalysisProfile:
    """Validate target/features and create an analysis profile."""
    if not isinstance(target_column, str) or not target_column:
        raise InvalidParameterError("target_column must be a non-empty string.")
    if (not isinstance(feature_columns, list) or not feature_columns
            or not all(isinstance(c, str) and c for c in feature_columns)):
        raise InvalidParameterError("feature_columns must be a non-empty list of column names.")
    if len(set(feature_columns)) != len(feature_columns):
        raise InvalidParameterError("feature_columns must not contain duplicates.")
    if target_column in feature_columns:
        raise InvalidParameterError("The target column cannot also be a feature (target leakage).")
    if not _is_number(confidence_level) or not 0 < confidence_level < 1:
        raise InvalidParameterError("confidence_level must be strictly between 0 and 1.")
    return AnalysisProfile(target_column, list(feature_columns), float(confidence_level))


def _check_value(feature: str, value, meta: dict):
    if not _is_number(value):
        raise DataValidationError(f"{feature} must be a number; got {value!r}.")
    if meta.get("allowed_values"):
        if value not in meta["allowed_values"]:
            raise DataValidationError(f"{feature} must be one of {meta['allowed_values']}; got {value}.")
        return int(value)
    if meta.get("minimum") is not None and value < meta["minimum"]:
        raise DataValidationError(f"{feature} must be at least {meta['minimum']}; got {value}.")
    if meta.get("maximum") is not None and value > meta["maximum"]:
        raise DataValidationError(f"{feature} must be at most {meta['maximum']}; got {value}.")
    if feature == "BMI" and value == 0:
        raise DataValidationError("BMI must be greater than 0.")
    if meta.get("integer_only") and value != int(value):
        raise DataValidationError(f"{feature} must be a whole number; got {value}.")
    return float(value)


def build_risk_profile(feature_values: dict, metadata: dict, defaults: dict | None = None) -> dict:
    """Validate user prediction inputs against metadata.

    Every prediction_input feature must be supplied. Other model features come from feature_values
    if given, otherwise from defaults (typical training-set values); defaulted features are listed.
    """
    if not isinstance(feature_values, dict):
        raise DataValidationError("feature_values must be a dictionary of feature -> value.")
    if not isinstance(metadata, dict) or not metadata:
        raise InvalidParameterError("metadata must be a non-empty variable-metadata dictionary.")
    if defaults is not None and not isinstance(defaults, dict):
        raise InvalidParameterError("defaults must be a dictionary or None.")
    defaults = defaults or {}
    features = [c for c, meta in metadata.items() if meta.get("role") == "feature" and meta.get("required_for_model")]

    unknown = sorted(set(feature_values) - set(features))
    if unknown:
        raise DataValidationError(f"Unknown features: {unknown}.")
    missing = [c for c in features if metadata[c].get("prediction_input") and c not in feature_values]
    if missing:
        raise DataValidationError(f"Missing required inputs: {missing}.")

    values, defaulted = {}, []
    for feature in features:
        if feature in feature_values:
            raw = feature_values[feature]
        elif feature in defaults:
            raw = defaults[feature]
            defaulted.append(feature)
        else:
            raise DataValidationError(f"No value or default available for {feature}.")
        values[feature] = _check_value(feature, raw, metadata[feature])

    display = {metadata[c]["label"]: (metadata[c].get("category_labels") or {}).get(values[c], values[c])
               for c in features}
    return {"values": values, "defaulted_features": defaulted, "display": display}


def predict_risk_profile(model, risk_profile: dict, threshold: float = 0.50) -> dict:
    """Return model probability/classification; never present it as a diagnosis."""
    if not _is_number(threshold) or not 0 <= threshold <= 1:
        raise InvalidParameterError("threshold must be between 0 and 1.")
    if not hasattr(model, "predict_proba") or not hasattr(model, "feature_names_in_"):
        raise InvalidParameterError("model must be a fitted classifier trained on named features.")
    if not isinstance(risk_profile, dict) or not isinstance(risk_profile.get("values"), dict):
        raise DataValidationError("risk_profile must be the result of build_risk_profile().")

    expected = list(model.feature_names_in_)
    supplied = set(risk_profile["values"])
    if supplied != set(expected):
        raise DataValidationError(f"Profile does not match the model's features: missing "
                                  f"{sorted(set(expected) - supplied)}, unexpected {sorted(supplied - set(expected))}.")
    row = pd.DataFrame([[risk_profile["values"][c] for c in expected]], columns=expected)
    probability = float(model.predict_proba(row)[0, 1])
    classification = int(probability >= threshold)
    return {
        "probability": probability,
        "probability_label": PROBABILITY_LABEL,
        "threshold": float(threshold),
        "classification": classification,
        "classification_label": CLASSIFICATION_LABEL,
        "classification_text": ("At or above threshold: model assigns the positive class (prediabetes or diabetes)"
                                if classification else "Below threshold: model assigns the negative class"),
        "disclaimer": DISCLAIMER,
    }