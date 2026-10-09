"""Function signature skeleton module.

Implementation intentionally omitted; use TODOs as the contract.
"""

import pandas as pd
import numpy as np
from src.common.exceptions import DataValidationError, InvalidParameterError
from src.modeling.classification import train_logistic_classifier, train_random_forest

# GradientBoostingClassifier has no class_weight parameter — excluded intentionally
SUPPORTED_MODELS = {
    "logistic_regression": train_logistic_classifier,
    "random_forest":       train_random_forest,
}

def inspect_class_distribution(y) -> dict:
    """Return class counts, percentages, imbalance ratio, majority/minority class.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    if not isinstance(y, pd.Series):
        raise DataValidationError("y must be a pandas Series.")
    if not pd.api.types.is_numeric_dtype(y):
        raise DataValidationError("y must be a numeric series.")
    if not all(y.isin([0, 1])):
        raise DataValidationError("y must be a binary series.")
    
    y = pd.Series(y)
    counts = y.value_counts().sort_index()
    total = len(y)
    percentages = counts / total * 100
    majority_class = int(counts.idxmax())
    minority_class = int(counts.idxmin())
    imbalance_ratio = counts.max() / counts.min()
    return {
        "counts": counts.to_dict(),
        "percentages": percentages.to_dict(),
        "total": total,
        "imbalance_ratio": imbalance_ratio,
        "majority_class": majority_class,
        "minority_class": minority_class
    }


def balance_training_data(X_train, y_train, method: str = "undersample", random_state: int = 42) -> dict:
    """Balance training data only; report before/after distributions.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise DataValidationError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise DataValidationError("y_train must be a pandas Series.")
    if not pd.api.types.is_numeric_dtype(y_train):
        raise DataValidationError("y_train must be a numeric series.")
    if not all(y_train.isin([0, 1])):
        raise DataValidationError("y_train must be a binary series.")

    rng = np.random.default_rng(random_state)
    X = X_train.reset_index(drop=True)
    y = y_train.reset_index(drop=True)
    
    majority_class = int(y.value_counts().idxmax())
    minority_class = int(y.value_counts().idxmin())

    majority_indices = y[y == majority_class].index
    minority_indices = y[y == minority_class].index

    before_distribution = inspect_class_distribution(y)
    if method == "undersample":
        sampled_majority = rng.choice(majority_indices, size=len(minority_indices), replace=False)
        keep = np.concatenate([sampled_majority, minority_indices.to_numpy()])
        
    elif method == "oversample":
        extra = rng.choice(minority_indices, size=len(majority_indices) - len(minority_indices), replace=True)
        keep_original = np.arange(len(y))
        keep = np.concatenate([keep_original, extra])
    else:
        raise InvalidParameterError(f"Unsupported balancing method: {method}. Use 'undersample' or 'oversample'.")

    X_balanced = X.iloc[keep].reset_index(drop=True)
    y_balanced = y.iloc[keep].reset_index(drop=True)
    after_distribution = inspect_class_distribution(y_balanced)
    return {
        "X_train": X_balanced,
        "y_train": y_balanced,
        "method": method,
        "random_state": random_state,
        "before_distribution": before_distribution,
        "after_distribution": after_distribution
    }


def train_class_weighted_model(model_type: str, X_train, y_train, random_state: int = 42) -> object:
    """Train supported class-weighted models; record strategy.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    
    if model_type not in SUPPORTED_MODELS:
        raise InvalidParameterError(f"Unsupported model type: {model_type}")
    if not isinstance(X_train, pd.DataFrame):
        raise DataValidationError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise DataValidationError("y_train must be a pandas Series.")
    if not pd.api.types.is_numeric_dtype(y_train):
        raise DataValidationError("y_train must be a numeric series.")
    if not all(y_train.isin([0, 1])):
        raise DataValidationError("y_train must be a binary series.")
    if not isinstance(random_state, int):
        raise DataValidationError("random_state must be an integer.")

    train_fn = SUPPORTED_MODELS[model_type]
    return train_fn(X_train, y_train, class_weight="balanced", random_state=random_state)

