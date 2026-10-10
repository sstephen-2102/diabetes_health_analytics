"""Member 2: run the planned hypothesis family (H1-H7) and return one tidy results table.

Interpretation is deliberately not generated here; the 'summary' column only restates the data.
No multiplicity adjustment is applied (FDR is out of scope by team decision).
"""
import pandas as pd

from src.common.exceptions import InvalidParameterError
from src.data._checks import require_frame
from src.statistics.association import calculate_effect_size
from src.statistics.hypothesis_testing import run_chi_square_test, run_welch_t_test

HYPOTHESES = [
    ("H1", "chi_square", "HighBP"),
    ("H2", "chi_square", "HighChol"),
    ("H3", "welch", "BMI"),
    ("H4", "chi_square", "PhysActivity"),
    ("H5", "chi_square", "Smoker"),
    ("H6", "chi_square", "Age"),
    ("H7", "chi_square", "Income"),
]


def _p_label(p_value: float) -> str:
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def _rate_summary(table: pd.DataFrame) -> str:
    """Restate the category 1 share per feature value (table: feature values x target 0/1)."""
    shares = (table[1] / table.sum(axis=1) * 100).round(2)
    if len(shares) <= 2:
        return "; ".join(f"{label}: {share:.2f}%" for label, share in shares.items())
    low, high = shares.idxmin(), shares.idxmax()
    return f"category 1 share ranges from {shares[low]:.2f}% (value {low}) to {shares[high]:.2f}% (value {high})"


def run_hypothesis_suite(data: pd.DataFrame, target_column: str = "Diabetes_binary") -> pd.DataFrame:
    """Run H1-H7 on one sample and return one row per hypothesis."""
    require_frame(data, [target_column] + [feature for _, _, feature in HYPOTHESES])
    if set(data[target_column].unique()) != {0, 1}:
        raise InvalidParameterError("target_column must contain exactly the values 0 and 1.")
    variant = data.attrs.get("dataset_variant", "unspecified")
    rows = []
    for hypothesis, test, feature in HYPOTHESES:
        if test == "welch":
            w = run_welch_t_test(data, feature, target_column, 1, 0)
            ci = w["confidence_interval"]
            rows.append({
                "hypothesis": hypothesis, "feature": feature, "test": "Welch t-test",
                "dataset_variant": variant, "n": w["n_a"] + w["n_b"],
                "statistic": w["t_statistic"], "dof": w["degrees_of_freedom"],
                "p_value": w["p_value"], "p_label": _p_label(w["p_value"]),
                "effect_measure": "cohens_d", "effect_value": w["cohens_d"],
                "odds_ratio": None, "or_lower": None, "or_upper": None,
                "summary": (f"mean {feature} {w['mean_a']:.2f} (category 1) vs {w['mean_b']:.2f} (category 0); "
                            f"difference {w['mean_difference']:.2f} [{ci['lower']:.2f}, {ci['upper']:.2f}]"),
                "warnings": " | ".join(w["assumption_warnings"]),
            })
        else:
            chi = run_chi_square_test(data, feature, target_column)
            table = chi["contingency_table"]
            odds = None
            if table.shape == (2, 2):
                odds = calculate_effect_size("odds_ratio", data, feature=feature, target_column=target_column)
            rows.append({
                "hypothesis": hypothesis, "feature": feature, "test": "Chi-square",
                "dataset_variant": variant, "n": chi["n"],
                "statistic": chi["chi_square"], "dof": chi["degrees_of_freedom"],
                "p_value": chi["p_value"], "p_label": _p_label(chi["p_value"]),
                "effect_measure": "cramers_v", "effect_value": chi["cramers_v"],
                "odds_ratio": odds["value"] if odds else None,
                "or_lower": odds["lower"] if odds else None,
                "or_upper": odds["upper"] if odds else None,
                "summary": _rate_summary(table),
                "warnings": " | ".join(chi["assumption_warnings"]),
            })
    return pd.DataFrame(rows)
