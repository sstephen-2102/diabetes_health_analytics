"""Chart-ready data for model evaluation; no plotting here, so the report and the app share it."""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.common.exceptions import DataValidationError


def _validate_curve_inputs(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    try:
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"y_true and probabilities must be numeric: {exc}") from exc
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise DataValidationError("y_true and probabilities must be non-empty 1-D sequences of equal length.")
    if np.isnan(y).any() or np.isnan(p).any():
        raise DataValidationError("y_true and probabilities must not contain missing values.")
    if not np.isin(y, [0, 1]).all():
        raise DataValidationError("y_true must contain only 0 and 1.")
    if ((p < 0) | (p > 1)).any():
        raise DataValidationError("probabilities must be between 0 and 1.")
    if len(np.unique(y)) < 2:
        raise DataValidationError("y_true must contain both classes.")
    return y.astype(int), p


def prepare_roc_curve_data(y_true, probabilities) -> pd.DataFrame:
    """Return false-positive rate, true-positive rate, threshold.

    The first row (nothing predicted positive) has no threshold and is stored as NaN.
    """
    y, p = _validate_curve_inputs(y_true, probabilities)
    fpr, tpr, thresholds = roc_curve(y, p)
    thresholds = np.where(np.isinf(thresholds), np.nan, thresholds)
    return pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds})


def prepare_precision_recall_data(y_true, probabilities) -> pd.DataFrame:
    """Return precision, recall, threshold.

    The last row (recall 0, precision 1) has no threshold and is stored as NaN.
    """
    y, p = _validate_curve_inputs(y_true, probabilities)
    precision, recall, thresholds = precision_recall_curve(y, p, drop_intermediate=True)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.append(thresholds, np.nan)})