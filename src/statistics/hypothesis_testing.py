"""Hypothesis tests (Member 2): chi-square and Welch t-test.

ANOVA, post-hoc analysis, and multiple-comparison adjustment remain stubs pending clarity from Sanjay.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.common.exceptions import InvalidParameterError
from src.data._checks import require_frame, require_numeric
from src.statistics.association import _crosstab, _two_group_values, calculate_effect_size, cramers_v_from_chi2
from src.statistics.confidence_intervals import difference_in_means_ci


def run_chi_square_test(data: pd.DataFrame, feature: str, target_column: str) -> dict:
    """Return contingency table, chi-square, df, p-value, Cramér V, n, and assumption warnings."""
    table = _crosstab(data, feature, target_column)
    chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
    n = int(table.to_numpy().sum())
    warnings = []
    small_cells = int((expected < 5).sum())
    if small_cells:
        warnings.append(f"{small_cells} cell(s) have expected count below 5; chi-square may be unreliable.")
    return {
        "test": "chi-square test of independence (no continuity correction)",
        "feature": feature, "target": target_column,
        "contingency_table": table,
        "expected_counts": pd.DataFrame(expected, index=table.index, columns=table.columns),
        "chi_square": float(chi2), "degrees_of_freedom": int(dof), "p_value": float(p_value),
        "cramers_v": cramers_v_from_chi2(chi2, n, *table.shape),
        "n": n, "assumption_warnings": warnings,
        "dataset_variant": data.attrs.get("dataset_variant", "unspecified"),
    }


def run_welch_t_test(data: pd.DataFrame, numeric_feature: str, target_column: str, group_a, group_b,
                     confidence_level: float = 0.95) -> dict:
    """Return group statistics, mean difference, t, df, p, CI, effect size (group_a minus group_b)."""
    require_numeric(data, [numeric_feature])
    a, b = _two_group_values(data, numeric_feature, target_column, group_a, group_b)
    t_statistic, p_value = stats.ttest_ind(a, b, equal_var=False)
    interval = difference_in_means_ci(a, b, confidence_level)
    warnings = []
    for label, values in (("group_a", a), ("group_b", b)):
        skew = float(stats.skew(values))
        if abs(skew) > 1:
            warnings.append(f"{label} is strongly skewed (skewness {skew:.2f}); interpret mean differences cautiously.")
    return {
        "test": "Welch two-sample t-test (unequal variances)",
        "feature": numeric_feature, "target": target_column,
        "group_a": group_a, "group_b": group_b,
        "n_a": len(a), "n_b": len(b),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "std_a": float(a.std(ddof=1)), "std_b": float(b.std(ddof=1)),
        "mean_difference": interval["difference"],
        "t_statistic": float(t_statistic), "degrees_of_freedom": interval["degrees_of_freedom"],
        "p_value": float(p_value),
        "confidence_interval": {"lower": interval["lower"], "upper": interval["upper"],
                                "confidence_level": confidence_level},
        "cohens_d": calculate_effect_size("cohens_d", data, numeric_feature=numeric_feature,
                                          target_column=target_column, group_a=group_a, group_b=group_b)["value"],
        "assumption_warnings": warnings,
        "dataset_variant": data.attrs.get("dataset_variant", "unspecified"),
    }


def run_anova(data: pd.DataFrame, numeric_feature: str, grouping_feature: str) -> dict:
    """Return group statistics, F, df, p, effect size.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    pass


def run_posthoc_analysis(data: pd.DataFrame, numeric_feature: str, grouping_feature: str, method: str = "tukey") -> pd.DataFrame:
    """Perform post-hoc comparisons with multiple-comparison adjustment.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    pass


def adjust_multiple_comparisons(p_values, method: str = "fdr_bh") -> pd.DataFrame:
    """Return original p, adjusted p, reject, method.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    pass


