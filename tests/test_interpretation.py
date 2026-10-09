import numpy as np
import pandas as pd
import pytest

from src.modeling.classification import train_logistic_classifier
from src.modeling.interpretation import calculate_feature_importance
from src.common.exceptions import DataValidationError, InvalidParameterError


@pytest.fixture
def fitted_model_and_data():
    """Target depends strongly on Signal and not at all on Noise."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame({
        "Signal": rng.normal(0, 1, n),
        "Noise": rng.normal(0, 1, n),
    })
    probability = 1 / (1 + np.exp(-3 * X["Signal"]))
    y = pd.Series(rng.binomial(1, probability), name="Diabetes_binary")
    model = train_logistic_classifier(X, y)
    return model, X, y


def test_returns_required_columns(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    table = calculate_feature_importance(model, X, y, n_repeats=3)
    assert table.columns.tolist() == ["feature", "importance_mean", "importance_std"]


def test_one_row_per_feature(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    table = calculate_feature_importance(model, X, y, n_repeats=3)
    assert sorted(table["feature"]) == sorted(X.columns)


def test_sorted_by_importance_descending(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    table = calculate_feature_importance(model, X, y, n_repeats=3)
    assert table["importance_mean"].is_monotonic_decreasing


def test_informative_feature_ranks_first(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    table = calculate_feature_importance(model, X, y, n_repeats=5)
    assert table.iloc[0]["feature"] == "Signal"
    signal = table.set_index("feature").loc["Signal", "importance_mean"]
    noise = table.set_index("feature").loc["Noise", "importance_mean"]
    assert signal > 0.1
    assert abs(noise) < 0.05


def test_std_is_non_negative(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    table = calculate_feature_importance(model, X, y, n_repeats=3)
    assert (table["importance_std"] >= 0).all()


def test_is_reproducible(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    t1 = calculate_feature_importance(model, X, y, n_repeats=3, random_state=7)
    t2 = calculate_feature_importance(model, X, y, n_repeats=3, random_state=7)
    pd.testing.assert_frame_equal(t1, t2)


def test_records_settings_in_attrs(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    table = calculate_feature_importance(model, X, y, n_repeats=3)
    assert table.attrs == {"method": "permutation", "scoring": "roc_auc",
                           "n_repeats": 3, "random_state": 42}


def test_rejects_unsupported_method(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    with pytest.raises(InvalidParameterError):
        calculate_feature_importance(model, X, y, method="shap")


def test_rejects_model_without_predict_proba(fitted_model_and_data):
    _, X, y = fitted_model_and_data
    with pytest.raises(InvalidParameterError):
        calculate_feature_importance(object(), X, y)


@pytest.mark.parametrize("bad_repeats", [0, -1, 2.5])
def test_rejects_invalid_n_repeats(fitted_model_and_data, bad_repeats):
    model, X, y = fitted_model_and_data
    with pytest.raises(InvalidParameterError):
        calculate_feature_importance(model, X, y, n_repeats=bad_repeats)


def test_rejects_non_dataframe(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    with pytest.raises(DataValidationError):
        calculate_feature_importance(model, X.to_numpy(), y)


def test_rejects_missing_values(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    X = X.copy()
    X.iloc[0, 0] = np.nan
    with pytest.raises(DataValidationError):
        calculate_feature_importance(model, X, y)


def test_rejects_length_mismatch(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    with pytest.raises(DataValidationError):
        calculate_feature_importance(model, X, y.iloc[:-1])


def test_rejects_non_binary_target(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    y_bad = y.copy()
    y_bad.iloc[0] = 2
    with pytest.raises(DataValidationError):
        calculate_feature_importance(model, X, y_bad)
