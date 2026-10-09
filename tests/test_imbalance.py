import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.modeling.imbalance import (
    balance_training_data,
    inspect_class_distribution,
    train_class_weighted_model,
)
from src.common.exceptions import DataValidationError, InvalidParameterError


# ---------------------------------------------------------------------------
# Shared fixture — imbalanced dataset (majority class 0, minority class 1)
# ---------------------------------------------------------------------------

@pytest.fixture
def imbalanced_data():
    rng = np.random.default_rng(42)
    n = 120
    X = pd.DataFrame({
        "BMI": rng.normal(28, 5, n),
        "Age": rng.integers(1, 14, n).astype(float),
        "HighBP": rng.integers(0, 2, n).astype(float),
    })
    # ~6:1 imbalance: 100 zeros, 20 ones
    y = pd.Series([0] * 100 + [1] * 20, name="Diabetes_binary")
    return X, y


# ---------------------------------------------------------------------------
# inspect_class_distribution
# ---------------------------------------------------------------------------

def test_inspect_returns_required_keys(imbalanced_data):
    _, y = imbalanced_data
    result = inspect_class_distribution(y)
    required = {"counts", "percentages", "total", "imbalance_ratio",
                "majority_class", "minority_class"}
    assert required.issubset(result.keys())


def test_inspect_total_is_correct(imbalanced_data):
    _, y = imbalanced_data
    result = inspect_class_distribution(y)
    assert result["total"] == len(y)


def test_inspect_counts_sum_to_total(imbalanced_data):
    _, y = imbalanced_data
    result = inspect_class_distribution(y)
    assert sum(result["counts"].values()) == result["total"]


def test_inspect_percentages_sum_to_100(imbalanced_data):
    _, y = imbalanced_data
    result = inspect_class_distribution(y)
    assert abs(sum(result["percentages"].values()) - 100.0) < 1e-6


def test_inspect_majority_minority_correct(imbalanced_data):
    _, y = imbalanced_data
    result = inspect_class_distribution(y)
    assert result["majority_class"] == 0
    assert result["minority_class"] == 1


def test_inspect_imbalance_ratio_correct(imbalanced_data):
    _, y = imbalanced_data
    result = inspect_class_distribution(y)
    assert abs(result["imbalance_ratio"] - 5.0) < 1e-6   # 100 / 20 = 5


def test_inspect_rejects_non_series():
    with pytest.raises(DataValidationError):
        inspect_class_distribution([0, 1, 0, 1])


def test_inspect_rejects_non_numeric():
    with pytest.raises(DataValidationError):
        inspect_class_distribution(pd.Series(["a", "b", "a"]))


def test_inspect_rejects_non_binary():
    with pytest.raises(DataValidationError):
        inspect_class_distribution(pd.Series([0, 1, 2, 0, 1]))


# ---------------------------------------------------------------------------
# balance_training_data — undersample
# ---------------------------------------------------------------------------

def test_undersample_returns_required_keys(imbalanced_data):
    X, y = imbalanced_data
    result = balance_training_data(X, y, method="undersample")
    assert {"X_train", "y_train", "method", "random_state",
            "before_distribution", "after_distribution"}.issubset(result.keys())


def test_undersample_balances_classes(imbalanced_data):
    X, y = imbalanced_data
    result = balance_training_data(X, y, method="undersample")
    counts = result["y_train"].value_counts()
    assert counts[0] == counts[1]


def test_undersample_reduces_total_rows(imbalanced_data):
    X, y = imbalanced_data
    result = balance_training_data(X, y, method="undersample")
    assert len(result["X_train"]) < len(X)


def test_undersample_X_y_same_length(imbalanced_data):
    X, y = imbalanced_data
    result = balance_training_data(X, y, method="undersample")
    assert len(result["X_train"]) == len(result["y_train"])


def test_undersample_is_reproducible(imbalanced_data):
    X, y = imbalanced_data
    r1 = balance_training_data(X, y, method="undersample", random_state=42)
    r2 = balance_training_data(X, y, method="undersample", random_state=42)
    pd.testing.assert_frame_equal(r1["X_train"], r2["X_train"])


def test_undersample_records_before_after(imbalanced_data):
    X, y = imbalanced_data
    result = balance_training_data(X, y, method="undersample")
    assert result["before_distribution"]["total"] == len(y)
    assert result["after_distribution"]["imbalance_ratio"] == 1.0


# ---------------------------------------------------------------------------
# balance_training_data — oversample
# ---------------------------------------------------------------------------

def test_oversample_balances_classes(imbalanced_data):
    X, y = imbalanced_data
    result = balance_training_data(X, y, method="oversample")
    counts = result["y_train"].value_counts()
    assert counts[0] == counts[1]


def test_oversample_increases_total_rows(imbalanced_data):
    X, y = imbalanced_data
    result = balance_training_data(X, y, method="oversample")
    assert len(result["X_train"]) > len(X)


def test_oversample_X_y_same_length(imbalanced_data):
    X, y = imbalanced_data
    result = balance_training_data(X, y, method="oversample")
    assert len(result["X_train"]) == len(result["y_train"])


# ---------------------------------------------------------------------------
# balance_training_data — validation errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["undersampling", "smote", ""])
def test_balance_rejects_unknown_method(imbalanced_data, method):
    X, y = imbalanced_data
    with pytest.raises(InvalidParameterError, match="Unsupported balancing method"):
        balance_training_data(X, y, method=method)


def test_balance_rejects_non_dataframe(imbalanced_data):
    _, y = imbalanced_data
    with pytest.raises(DataValidationError):
        balance_training_data("not_a_df", y)


def test_balance_rejects_non_series_y(imbalanced_data):
    X, _ = imbalanced_data
    with pytest.raises(DataValidationError):
        balance_training_data(X, [0, 1, 0])


# ---------------------------------------------------------------------------
# train_class_weighted_model
# ---------------------------------------------------------------------------

def test_weighted_logistic_returns_pipeline(imbalanced_data):
    X, y = imbalanced_data
    model = train_class_weighted_model("logistic_regression", X, y)
    assert isinstance(model, Pipeline)


def test_weighted_random_forest_returns_pipeline(imbalanced_data):
    X, y = imbalanced_data
    model = train_class_weighted_model("random_forest", X, y)
    assert isinstance(model, Pipeline)


def test_weighted_model_has_class_weight_balanced(imbalanced_data):
    X, y = imbalanced_data
    model = train_class_weighted_model("logistic_regression", X, y)
    assert model.named_steps["classifier"].class_weight == "balanced"


def test_weighted_model_rejects_invalid_type(imbalanced_data):
    X, y = imbalanced_data
    with pytest.raises(InvalidParameterError):
        train_class_weighted_model("gradient_boosting", X, y)


def test_weighted_model_rejects_non_dataframe(imbalanced_data):
    _, y = imbalanced_data
    with pytest.raises(DataValidationError):
        train_class_weighted_model("logistic_regression", "not_a_df", y)
