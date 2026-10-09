import numpy as np 
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from src.modeling.classification import (
    train_logistic_classifier,
    train_random_forest,
    train_gradient_boosting,
)
from src.common.exceptions import DataValidationError, InvalidParameterError, ModelTrainingError

@pytest.fixture
def training_data():
    """Create a deterministic synthetic binary classificaiton dataset."""

    rng = np.random.default_rng(42) 
    X = pd.DataFrame({
        "BMI": rng.normal(loc=28, scale=5, size=120),
        "Age": rng.integers(low=1, high=14, size=120),
        "HighBP": rng.integers(low=0, high=2, size=120),
    })
    
    # Synthetic target with a learnable relationship
    score = (0.08 *(X["BMI"] - 28) + 0.25 * (X["Age"] - 7) + 0.8 * X["HighBP"])

    probability = 1 / (1 + np.exp(-score))
    y = pd.Series(rng.binomial(n=1, p=probability), name="Diabetes_binary")

    # Ensure both classes exist for this fixture
    y.iloc[0] = 0
    y.iloc[1] = 1

    return X, y

def test_logistic_classifier_trains_sucessfully(training_data):
    X, y = training_data
    model = train_logistic_classifier(X, y)
    assert isinstance(model, Pipeline)
    assert len(model.steps) == 2
    assert hasattr(model, "predict") 

def test_logistic_classifier_predicts_binary_classes(training_data):
    X, y = training_data
    model = train_logistic_classifier(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})

def test_logistic_classifer_returns_probabilities(training_data):
    X, y = training_data
    model = train_logistic_classifier(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape[1] == 2
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1)

def test_logistic_classifier_is_reproducible(training_data):
    X, y = training_data
    model_a = train_logistic_classifier(X, y, random_state=42)
    model_b = train_logistic_classifier(X, y, random_state=42)
    probabilities_a = model_a.predict_proba(X)
    probabilities_b = model_b.predict_proba(X)
    assert np.allclose(probabilities_a, probabilities_b)

def test_logistic_classifier_supports_class_weight(training_data):
    X, y = training_data
    model = train_logistic_classifier(X, y, class_weight="balanced")
    assert model.named_steps["classifier"].class_weight == "balanced"

def test_logistic_classifier_rejects_empty_data():
    X = pd.DataFrame(columns=["BMI", "Age"])
    y = pd.Series(dtype=int)
    with pytest.raises(DataValidationError):
        train_logistic_classifier(X, y)

def test_logistic_classifier_rejects_missing_values(training_data):
    X, y = training_data
    X = X.copy()
    X.loc[0, "BMI"] = np.nan

    with pytest.raises(DataValidationError):
        train_logistic_classifier(X, y)

def test_logistic_classifier_rejects_single_class(training_data):
    X, y = training_data
    y = pd.Series(np.zeros(len(y), dtype=int))

    with pytest.raises(DataValidationError):
        train_logistic_classifier(X, y)

def test_logistic_classifier_rejects_invalid_class_weight(training_data):
    X, y = training_data

    with pytest.raises(InvalidParameterError):
        train_logistic_classifier(X, y, class_weight="invalid")

def test_logistic_classifier_rejects_row_mismatch(training_data):
    X, y = training_data

    with pytest.raises(DataValidationError):
        train_logistic_classifier(X, y.loc[:-1])


@pytest.mark.parametrize("train_fn", [train_random_forest, train_gradient_boosting])
def test_tree_models_train_and_predict(training_data, train_fn):
    X, y = training_data
    model = train_fn(X, y, random_state=42, n_estimators=20)
    assert isinstance(model, Pipeline)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1)


@pytest.mark.parametrize("train_fn", [train_random_forest, train_gradient_boosting])
def test_tree_models_are_reproducible(training_data, train_fn):
    X, y = training_data
    model_a = train_fn(X, y, random_state=42, n_estimators=20)
    model_b = train_fn(X, y, random_state=42, n_estimators=20)
    assert np.allclose(model_a.predict_proba(X), model_b.predict_proba(X))


def test_random_forest_supports_class_weight(training_data):
    X, y = training_data
    model = train_random_forest(X, y, class_weight="balanced", n_estimators=20)
    assert model.named_steps["classifier"].class_weight == "balanced"