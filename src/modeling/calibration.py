"""Calibration utilities for comparing predicted probabilities with observed outcomes."""

import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from src.common.exceptions import DataValidationError, InvalidParameterError


def _validate_inputs(y_true, probabilities, n_bins) -> np.ndarray:
    if not isinstance(y_true, pd.Series):
        raise DataValidationError("y_true must be a pandas Series.")
    if y_true.empty:
        raise DataValidationError("y_true must not be empty.")
    if not all(y_true.isin([0, 1])):
        raise DataValidationError("y_true must be a binary series.")
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities):
        raise DataValidationError("y_true and probabilities must have the same length.")
    if np.isnan(probabilities).any() or (probabilities < 0).any() or (probabilities > 1).any():
        raise DataValidationError("probabilities must be between 0 and 1.")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool):
        raise InvalidParameterError("n_bins must be an integer.")
    if n_bins <= 0:
        raise InvalidParameterError("n_bins must be greater than 0.")
    return probabilities


def calculate_calibration_metrics(y_true, probabilities, n_bins: int = 10) -> dict:
    """Return Brier score, expected calibration error and the calibration table."""
    table = generate_calibration_data(y_true, probabilities, n_bins)

    n = int(table["count"].sum())

    ece = float((table["count"] / n * (table["observed_frequency"] - table["mean_predicted"]).abs()).sum())
    brier_score = float(brier_score_loss(y_true, np.asarray(probabilities, dtype=float)))
    return {
        "brier_score": brier_score,
        "expected_calibration_error": ece,
        "calibration_table": table,
        "n_bins": n_bins,
        "n": n,
    }


def generate_calibration_data(y_true, probabilities, n_bins: int = 10) -> pd.DataFrame:
    """Return bin, predicted probability, observed frequency, count.

    Bins are equal-width over [0, 1]; bins containing no observations are omitted.
    """
    probabilities = _validate_inputs(y_true, probabilities, n_bins)
    y = np.asarray(y_true)

    bin_ids = np.minimum(np.floor(probabilities * n_bins).astype(int), n_bins - 1)

    frame = pd.DataFrame({
        "bin": bin_ids,
        "predicted_probability": probabilities,
        "observed_frequency": y,
    })
    table = (
        frame.groupby("bin").agg(
            mean_predicted=("predicted_probability", "mean"),
            observed_frequency=("observed_frequency", "mean"),
            count=("observed_frequency", "count"),
        ).reset_index()
    )
    table["bin_lower"] = table["bin"] / n_bins
    table["bin_upper"] = (table["bin"] + 1) / n_bins
    return table[["bin", "bin_lower", "bin_upper", "mean_predicted", "observed_frequency", "count"]]
