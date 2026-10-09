"""
Classification model utilities.

This module contains machine-learning classification functions for the Diabetes Health Analytics Project.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.common.exceptions import DataValidationError, InvalidParameterError, ModelTrainingError
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
from src.modeling.validation import check_duplicate_leakage

def split_data(data: pd.DataFrame,
                target_column: str, 
                test_size: float = 0.20, 
                random_state: int = 42, 
                stratify: bool = True
) -> dict:
    
    """
    Create a reproducible train/test split
    Parameters:
    -----------
    data:
        Modeling dataset supplied by the data-preparation layer.
        For the primary ML pipeline this should be the agreed 'unique_profile_v1' dataset.
    target_column:
        Name of the binary target column
    test_size:
        Proportion of observations assigned to the test 
    """
    # 1. Validate the input object
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("split_data() expected a pandas DataFrame.")
    if data.empty:
        raise DataValidationError("split_data() received an empty DataFrame.")

    # 2. Validate the target column
    if target_column not in data.columns:
        raise DataValidationError(f"Target column {target_column} was not found in the dataset")
    if data[target_column].isna().any():
        raise DataValidationError(f"Target column {target_column} contains missing values")

    # 3. Validate test size
    if not 0 < test_size < 1:
        raise InvalidParameterError("test_size must be greater than 0 and less than 1.")

    # 4. Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 5. Validate binary target
    unique_target_values = sorted(y.unique().tolist())
    if len(unique_target_values) != 2:
        raise DataValidationError(
            "The target column must contain exactly two classes."
            f"Found: {unique_target_values}"
        )

    # 6. Group identical feature vectors so they stay on one side of the split
    groups = pd.util.hash_pandas_object(X, index=False).values

    # 7. Create reproducible group-aware train/test split
    if stratify:
        n_splits = round(1/test_size)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    leakage = check_duplicate_leakage(X_train, X_test)
    if leakage["leakage_detected"]:
        raise DataValidationError(f"Duplicate leakage detected: {leakage}")

    # 8. Store metadata for reproducibility
    metadata = {
        "target_columns" : target_column,
        "test_size" : test_size,
        "train_size": len(X_train),
        "test_rows": len(X_test),
        "random_state": random_state,
        "stratified": stratify,
        "feature_count": X_train.shape[1],
        "target_classes": unique_target_values,
        "original_rows": len(data),
        "grouped_by_feature_vectors": True,
        "actual_test_fraction": len(X_test) / len(data),
        "leakage_check": leakage
    }

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metadata": metadata
    }

def train_logistic_classifier(X_train, y_train, class_weight=None, random_state: int = 42) -> object:
    """Train a reproducible machine-learning logistic regression model.
    
    Standardizes numberic features through a pipeline so that scaling
    is learned only from the training data.
    """
    # Validate feature data
    if not isinstance(X_train, pd.DataFrame):
        raise DataValidationError("X_train must be a pandas DataFrame.")
    if X_train.empty:
        raise DataValidationError("X_train must not be empty.")
    if X_train.columns.duplicated().any():
        raise DataValidationError("X_train contains duplocate column names")
    if X_train.isnull().any().any():
        raise DataValidationError("X_train contains missing values")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes):
        raise DataValidationError("All training features must be numeric.")
    
    # Validate target
    y = pd.Series(y_train)
    if len(X_train) != len(y):
        raise DataValidationError("X_train and y_train must have the same number of rows.")
    if y.isnull().any():
        raise DataValidationError("y_train contains missing values")
    
    classes = sorted(y.unique().tolist())
    if len(classes) != 2:
        raise DataValidationError("y_train must contain exactly two classes.")
    
    # Validate class weight
    if class_weight not in [None, "balanced", dict]:
        raise InvalidParameterError("Logistic Regression requires exactly two target classes"
        f"Found: {classes}")
    
    # Validate class weights
    if class_weight not in (None, "balanced") and not isinstance(class_weight, dict):
        raise InvalidParameterError("class_weight must be None, 'balanced' or a disctionary.")

    # Build the pipeline
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", 
            LogisticRegression(
                class_weight=class_weight,
                random_state=random_state,
                max_iter=1000
            )
            )
        ]
    )

    # Fit the model
    try:
        model.fit(X_train, y)
    except Exception as ex:
        raise ModelTrainingError(f"Logistic Regressiuon Training failed: {str(ex)}") from ex
    return model


def train_random_forest(X_train, y_train, class_weight=None, random_state: int = 42, **model_parameters) -> object:
    """Train Random Forest.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    # Validate feature data
    if not isinstance(X_train, pd.DataFrame):
        raise DataValidationError("X_train must be a pandas DataFrame.")
    if X_train.empty:
        raise DataValidationError("X_train must not be empty.")
    if X_train.columns.duplicated().any():
        raise DataValidationError("X_train contains duplocate column names")
    if X_train.isnull().any().any():
        raise DataValidationError("X_train contains missing values")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes):
        raise DataValidationError("All training features must be numeric.")
    
    # Validate target
    y = pd.Series(y_train)
    if len(X_train) != len(y):
        raise DataValidationError("X_train and y_train must have the same number of rows.")
    if y.isnull().any():
        raise DataValidationError("y_train contains missing values")
    
    classes = sorted(y.unique().tolist())
    if len(classes) != 2:
        raise DataValidationError("y_train must contain exactly two classes.")
    
    # Validate class weight
    if class_weight not in [None, "balanced", dict]:
        raise InvalidParameterError("Logistic Regression requires exactly two target classes"
        f"Found: {classes}")
    
    # Validate class weights
    if class_weight not in (None, "balanced") and not isinstance(class_weight, dict):
        raise InvalidParameterError("class_weight must be None, 'balanced' or a disctionary.")

    valid_cw = (None, "balanced", "balanced_subsample")
    if class_weight not in valid_cw and not isinstance(class_weight, dict):
        raise InvalidParameterError("class_weight must be None, 'balanced', 'balanced_subsample' or a disctionary.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            class_weight=class_weight,
            random_state=random_state,
            **model_parameters
        ))
    ])
    try:
        model.fit(X_train, y_train)
    except Exception as ex:
        raise ModelTrainingError(f"Random Forest training failed: {ex}") from ex
    return model


def train_gradient_boosting(X_train, y_train, random_state: int = 42, **model_parameters) -> object:
    """Train Gradient Boosting.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
     # Validate feature data
    if not isinstance(X_train, pd.DataFrame):
        raise DataValidationError("X_train must be a pandas DataFrame.")
    if X_train.empty:
        raise DataValidationError("X_train must not be empty.")
    if X_train.columns.duplicated().any():
        raise DataValidationError("X_train contains duplocate column names")
    if X_train.isnull().any().any():
        raise DataValidationError("X_train contains missing values")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes):
        raise DataValidationError("All training features must be numeric.")
    
    # Validate target
    y = pd.Series(y_train)
    if len(X_train) != len(y):
        raise DataValidationError("X_train and y_train must have the same number of rows.")
    if y.isnull().any():
        raise DataValidationError("y_train contains missing values")
    
    classes = sorted(y.unique().tolist())
    if len(classes) != 2:
        raise DataValidationError("y_train must contain exactly two classes.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(
            random_state=random_state,
            **model_parameters
        ))
    ])
    try:
        model.fit(X_train, y_train)
    except Exception as ex:
        raise ModelTrainingError(f"Gradient Boosting Training failed: {str(ex)}") from ex
    return model

def generate_predictions(model, X, threshold: float = 0.50) -> dict:
    """Return probabilities, predictions, threshold.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    if not isinstance(model, Pipeline):
        raise DataValidationError("model must be a sklearn pipeline.")
    if not isinstance(X, pd.DataFrame):
        raise DataValidationError("X must be a pandas DataFrame.")
    if X.empty:
        raise DataValidationError("X must not be empty.")
    if X.columns.duplicated().any():
        raise DataValidationError("X contains duplocate column names")
    if X.isnull().any().any():
        raise DataValidationError("X contains missing values")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise DataValidationError("All features must be numeric.")
    if not 0 <= threshold <= 1:
        raise InvalidParameterError("threshold must be between 0 and 1.")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "threshold": threshold
    }



def evaluate_classifier(y_true, y_pred, y_probability) -> dict:
    """Return accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, confusion matrix, Brier score.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    named = {"y_true": y_true, "y_pred": y_pred, "y_probability": y_probability}
    for name, values in named.items():
        if not isinstance(values, (pd.Series, np.ndarray, list)):
            raise DataValidationError(f"{name} must be a pandas Series, NumPy array or list.")
        values = pd.Series(np.asarray(values))
        if values.empty:
            raise DataValidationError(f"{name} must not be empty.")
        if values.isnull().any():
            raise DataValidationError(f"{name} contains missing values")
        if not pd.api.types.is_numeric_dtype(values):
            raise DataValidationError(f"{name} must be numeric.")
        named[name] = values
    y_true, y_pred, y_probability = named["y_true"], named["y_pred"], named["y_probability"]
    if not len(y_true) == len(y_pred) == len(y_probability):
        raise DataValidationError("y_true, y_pred, and y_probability must have the same length.")
    if not y_true.isin([0, 1]).all() or not y_pred.isin([0, 1]).all():
        raise DataValidationError("y_true and y_pred must contain only 0 and 1.")
    if not y_probability.between(0, 1).all():
        raise DataValidationError("y_probability must be between 0 and 1.")
    if y_true.nunique() < 2:
        raise DataValidationError("y_true must contain both classes to compute ROC-AUC.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "pr_auc": average_precision_score(y_true, y_probability),
        "brier_score": brier_score_loss(y_true, y_probability),
        "confusion_matrix": {"true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp) },
        "n": len(y_true)
    }
def compare_models(evaluation_results: dict) -> pd.DataFrame:
    """Create metric comparison table; do not embed universal best-model logic.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    if not isinstance(evaluation_results, dict):
        raise DataValidationError("evaluation_results must be a dictionary.")
    if not evaluation_results:
        raise DataValidationError("evaluation_results must not be empty.")
    if not all(isinstance(result, dict) for result in evaluation_results.values()):
        raise DataValidationError("evaluation_results must contain dictionaries.")
    required = {"accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier_score", "confusion_matrix"}
    for model_name, metrics in evaluation_results.items():
        missing = required - set(metrics)
        if missing:
            raise DataValidationError(f"{model_name} is missing metrics: {sorted(missing)}")
    rows = []
    for model_name, metrics in evaluation_results.items():
        row = {"model": model_name}
        row.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        row.update(metrics["confusion_matrix"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


