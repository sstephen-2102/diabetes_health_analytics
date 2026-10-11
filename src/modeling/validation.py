""" Validation utilities for Machine-learning experiments
"""

import pandas as pd
from src.common.exceptions import DataValidationError

def check_duplicate_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict:
    """Check for identical feature rows appearing in both datasets.
    This check is important for the Diabetes Health Analytics project
    because the source dataset contains exact duplicate rows.
    
    Parameters
    ----------
    X_train: Training feature dataset
    X_test: Testing feature dataset
    
    Returns
    -------
    dict leakage validation results
    """
    if not isinstance(X_train, pd.DataFrame):
        raise DataValidationError("X_train must be a pandas DataFrame.")
    if not isinstance(X_test, pd.DataFrame):
        raise DataValidationError("X_test must be a pandas DataFrame")
    if list(X_train.columns) != list(X_test.columns):
        raise DataValidationError("X_train and X_est must contain the same feature columns" \
        "in the same order")

    # Convert each row into a hashable representatin
    train_hashes = pd.util.hash_pandas_object(X_train, index=False)
    test_hashes = pd.util.hash_pandas_object(X_test, index=False)
    overlapping_hashes = set(train_hashes).intersection(set(test_hashes))
    leakage_detected = len(overlapping_hashes) > 0

    return {
        "leakage_detected": leakage_detected,
        "overlapping_unique_rows": len(overlapping_hashes),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "status": "FAIL" if leakage_detected else "PASS"
    }