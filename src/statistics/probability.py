"""Member 2: probability functions.

Every result states its counts, target definition, and dataset variant.
"""
import pandas as pd

from src.common.config import TARGET_DEFINITION
from src.common.exceptions import InvalidParameterError
from src.data._checks import require_frame
from src.statistics.confidence_intervals import proportion_confidence_interval


def _condition_mask(data: pd.DataFrame, columns: list, values: list) -> pd.Series:
    """True for rows that meet EVERY (column == value) condition."""
    if not columns or len(columns) != len(values):
        raise InvalidParameterError("Supply equal-length, non-empty columns and values.")
    require_frame(data, columns)
    mask = pd.Series(True, index=data.index)
    for column, value in zip(columns, values):
        mask &= data[column] == value
    return mask


def calculate_probability(data: pd.DataFrame, column: str, value) -> float:
    """Calculate marginal P(column=value)."""
    require_frame(data, [column])
    probability = (data[column] == value).mean()
    return float(probability)


def calculate_conditional_probability(data: pd.DataFrame, target_column: str, target_value,
                                      condition_columns: list[str], condition_values: list) -> dict:
    """Calculate conditional probability with counts and CI; detect zero denominator."""
    require_frame(data, [target_column])
    mask = _condition_mask(data, condition_columns, condition_values)
    denominator = int(mask.sum())
    result = {
        "target": {target_column: target_value},
        "conditions": dict(zip(condition_columns, condition_values)),
        "denominator": denominator,
        "target_definition": TARGET_DEFINITION,
        "dataset_variant": data.attrs.get("dataset_variant", "unspecified"),
        "sample_size": len(data),
    }
    if denominator == 0:
        result.update(numerator=0, probability=None, confidence_interval=None,
                      warning="Zero denominator: no rows meet the conditions.")
        return result
    numerator = int((mask & (data[target_column] == target_value)).sum())
    probability = numerator / denominator   # numerator and denominator are already defined above
    result.update(numerator=numerator, probability=float(probability),
                  confidence_interval=proportion_confidence_interval(numerator, denominator),
                  warning=None)
    return result


def calculate_joint_probability(data: pd.DataFrame, conditions: dict) -> dict:
    """Calculate joint probability P(all conditions hold)."""
    mask = _condition_mask(data, list(conditions), list(conditions.values()))
    numerator, denominator = int(mask.sum()), len(data)
    return {
        "conditions": dict(conditions),
        "numerator": numerator,
        "denominator": denominator,
        "probability": numerator / denominator,
        "target_definition": TARGET_DEFINITION,
        "dataset_variant": data.attrs.get("dataset_variant", "unspecified"),
        "sample_size": denominator,
    }


def verify_bayes_theorem(data: pd.DataFrame, target_column: str, target_value,
                         condition_column: str, condition_value, tolerance: float = 1e-9) -> dict:
    """Compare direct conditional probability with Bayes RHS."""
    direct = calculate_conditional_probability(
        data, target_column, target_value, [condition_column], [condition_value])
    reverse = calculate_conditional_probability(
        data, condition_column, condition_value, [target_column], [target_value])
    prior = calculate_probability(data, target_column, target_value)
    evidence = calculate_probability(data, condition_column, condition_value)
    if direct["probability"] is None or reverse["probability"] is None:
        return {"matches": None, "warning": "Zero denominator: Bayes check undefined.",
                "dataset_variant": direct["dataset_variant"]}
    likelihood = reverse["probability"]          # P(condition | target)
    bayes_rhs = likelihood * prior / evidence                # built from likelihood, prior, evidence
    difference = abs(direct["probability"] - bayes_rhs)
    return {
        "direct_probability": direct["probability"],
        "likelihood": likelihood, "prior": prior, "evidence": evidence,
        "bayes_rhs": float(bayes_rhs), "absolute_difference": float(difference),
        "matches": bool(difference <= tolerance),
        "target_definition": TARGET_DEFINITION,
        "dataset_variant": direct["dataset_variant"], "sample_size": len(data),
        "warning": None,
    }


def calculate_risk_profile_probability(data: pd.DataFrame, target_column: str, target_value,
                                       profile: dict) -> dict:
    """Observed conditional probability for a profile; not an ML prediction."""
    result = calculate_conditional_probability(
        data, target_column, target_value, list(profile), list(profile.values()))
    result["note"] = "Observed share in this sample; not a model prediction."
    return result
