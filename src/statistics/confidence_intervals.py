"""Confidence intervals (Member 2).

Methods are documented in each result so a reader can see how it was computed.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from src.common.exceptions import InvalidParameterError


def _check_confidence_level(confidence_level: float) -> None:
    if not 0 < confidence_level < 1:
        raise InvalidParameterError("confidence_level must be strictly between 0 and 1.")


def _clean_numeric(values, name: str) -> tuple[np.ndarray, int]:
    """Return (numeric values with missing dropped, number dropped)."""
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    n_missing = int(series.isna().sum())
    clean = series.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        raise InvalidParameterError(f"{name} needs at least 2 numeric values.")
    return clean, n_missing


def proportion_confidence_interval(successes: int, total: int, confidence_level: float = 0.95) -> dict:
    """Validate inputs and calculate a proportion CI using a documented method (Wilson)."""
    if total <= 0:
        raise InvalidParameterError("total must be positive.")
    if not 0 <= successes <= total:
        raise InvalidParameterError("successes must be between 0 and total.")
    _check_confidence_level(confidence_level)
    method = "wilson"
    lower, upper = proportion_confint(successes, total, alpha=1 - confidence_level, method=method)
    return {
        "estimate": successes / total, "lower": float(lower), "upper": float(upper),
        "confidence_level": confidence_level, "method": method,
        "successes": int(successes), "total": int(total),
    }


def mean_confidence_interval(values, confidence_level: float = 0.95) -> dict:
    """Calculate a t-based mean CI."""
    _check_confidence_level(confidence_level)
    clean, n_missing = _clean_numeric(values, "values")
    n = len(clean)
    mean, std = float(clean.mean()), float(clean.std(ddof=1))
    margin = float(stats.t.ppf(1 - (1 - confidence_level) / 2, df=n - 1)) * std / np.sqrt(n)
    return {
        "mean": mean, "lower": mean - margin, "upper": mean + margin,
        "std": std, "n": n, "n_missing_dropped": n_missing,
        "confidence_level": confidence_level, "method": "t-interval, df = n - 1",
    }


def difference_in_means_ci(group_a, group_b, confidence_level: float = 0.95) -> dict:
    """Calculate a Welch-consistent CI for mean difference (group_a minus group_b)."""
    _check_confidence_level(confidence_level)
    a, missing_a = _clean_numeric(group_a, "group_a")
    b, missing_b = _clean_numeric(group_b, "group_b")
    var_a, var_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(var_a + var_b))
    df = float((var_a + var_b) ** 2 / (var_a ** 2 / (len(a) - 1) + var_b ** 2 / (len(b) - 1)))
    difference = float(a.mean() - b.mean())
    margin = float(stats.t.ppf(1 - (1 - confidence_level) / 2, df=df)) * se
    return {
        "difference": difference, "lower": difference - margin, "upper": difference + margin,
        "standard_error": se, "degrees_of_freedom": df,
        "n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "n_missing_dropped": missing_a + missing_b,
        "confidence_level": confidence_level, "method": "Welch (Satterthwaite df), a minus b",
    }
