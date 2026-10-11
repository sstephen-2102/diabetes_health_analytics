"""Model interpretation utilities; results describe model reliance, not causal effects."""

import pandas as pd
from sklearn.inspection import permutation_importance
from src.common.exceptions import DataValidationError, InvalidParameterError


def calculate_feature_importance(model, X, y, method: str = "permutation",
    scoring: str = "roc_auc", n_repeats: int = 10,
    random_state: int = 42) -> pd.DataFrame:
    """Return feature, importance mean, importance SD; avoid causal claims.

    Evaluate on a held-out split: importance measured on training data reflects
    memorisation rather than predictive value.
    """
    if not hasattr(model, "predict_proba"):
        raise InvalidParameterError("model must be a fitted classifier with predict_proba.")
    if method != "permutation":
        raise InvalidParameterError(f"Unsupported method: {method}. Only 'permutation' is supported.")
    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 1:
        raise InvalidParameterError("n_repeats must be a positive integer.")

    if not isinstance(X, pd.DataFrame):
        raise DataValidationError("X must be a pandas DataFrame.")
    if X.empty:
        raise DataValidationError("X must not be empty.")
    if X.isnull().any().any():
        raise DataValidationError("X contains missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise DataValidationError("All features must be numeric.")
    if not isinstance(y, pd.Series):
        raise DataValidationError("y must be a pandas Series.")
    if len(X) != len(y):
        raise DataValidationError("X and y must have the same number of rows.")
    if not y.isin([0, 1]).all():
        raise DataValidationError("y must be a binary series.")

    result = permutation_importance(
        model, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    table = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
    table.attrs.update({"method": method, "scoring": scoring,
                        "n_repeats": n_repeats, "random_state": random_state})
    return table
