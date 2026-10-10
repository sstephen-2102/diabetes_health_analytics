"""Member 2: Bayes' theorem demonstrated and verified on the observed data (research question RQ4).

For each feature value, compare the directly counted P(category 1 | feature) with the same quantity
rebuilt through Bayes: P(feature | category 1) * P(category 1) / P(feature).
"""
import pandas as pd

from src.common.exceptions import InvalidParameterError
from src.data._checks import require_frame
from src.statistics.probability import verify_bayes_theorem

DEFAULT_BAYES_FEATURES = ["HighBP", "HighChol", "PhysActivity", "Smoker",
                          "Stroke", "HeartDiseaseorAttack", "DiffWalk"]


def run_bayes_suite(data: pd.DataFrame, target_column: str = "Diabetes_binary",
                    features: list[str] | None = None, feature_value=1) -> pd.DataFrame:
    """Return one row per feature with prior, likelihood, evidence, Bayes RHS, direct value, and match flag."""
    features = features or DEFAULT_BAYES_FEATURES
    if not features:
        raise InvalidParameterError("Supply at least one feature.")
    require_frame(data, [target_column] + list(features))
    rows = []
    for feature in features:
        check = verify_bayes_theorem(data, target_column, 1, feature, feature_value)
        if check["matches"] is None:
            raise InvalidParameterError(f"Bayes check undefined for {feature}={feature_value}: {check['warning']}")
        rows.append({
            "feature": feature, "feature_value": feature_value,
            "dataset_variant": check["dataset_variant"], "n": check["sample_size"],
            "prior_p_category1": check["prior"],
            "likelihood_p_feature_given_category1": check["likelihood"],
            "evidence_p_feature": check["evidence"],
            "bayes_rhs": check["bayes_rhs"],
            "direct_p_category1_given_feature": check["direct_probability"],
            "absolute_difference": check["absolute_difference"],
            "matches": check["matches"],
            "lift_over_prior": check["direct_probability"] / check["prior"],
        })
    return pd.DataFrame(rows)
