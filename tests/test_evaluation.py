"""Tests for evaluate_classifier() and compare_models() in src/modeling/classification.py."""
import numpy as np
import pandas as pd
import pytest

from src.common.exceptions import DataValidationError
from src.modeling.classification import (
    compare_models,
    evaluate_classifier,
    generate_predictions,
    train_logistic_classifier,
)

Y_TRUE = [0, 0, 0, 1, 1, 1]
Y_PRED = [0, 0, 1, 1, 1, 0]
Y_PROB = [0.1, 0.2, 0.7, 0.8, 0.9, 0.4]


def test_evaluate_classifier_returns_all_spec_metrics():
    result = evaluate_classifier(pd.Series(Y_TRUE), pd.Series(Y_PRED), pd.Series(Y_PROB))

    assert set(result) == {"accuracy", "precision", "recall", "specificity", "f1",
                           "roc_auc", "pr_auc", "brier_score", "confusion_matrix", "n"}


def test_evaluate_classifier_values_match_hand_calculation():
    result = evaluate_classifier(pd.Series(Y_TRUE), pd.Series(Y_PRED), pd.Series(Y_PROB))

    assert result["confusion_matrix"] == {"true_negatives": 2, "false_positives": 1,
                                          "false_negatives": 1, "true_positives": 2}
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["specificity"] == pytest.approx(2 / 3)
    expected_brier = np.mean((np.array(Y_PROB) - np.array(Y_TRUE)) ** 2)
    assert result["brier_score"] == pytest.approx(expected_brier)
    assert result["n"] == 6


@pytest.mark.parametrize("convert", [pd.Series, np.array, list])
def test_evaluate_classifier_accepts_series_arrays_and_lists(convert):
    result = evaluate_classifier(convert(Y_TRUE), convert(Y_PRED), convert(Y_PROB))

    assert result["accuracy"] == pytest.approx(4 / 6)


def test_evaluate_classifier_ignores_mismatched_indexes():
    y_true = pd.Series(Y_TRUE, index=[10, 11, 12, 13, 14, 15])

    result = evaluate_classifier(y_true, np.array(Y_PRED), np.array(Y_PROB))

    assert result["confusion_matrix"]["true_positives"] == 2


def test_evaluate_classifier_chains_with_generate_predictions():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=200) > 0).astype(int))
    model = train_logistic_classifier(X, y)
    predictions = generate_predictions(model, X)

    result = evaluate_classifier(y, predictions["predictions"], predictions["probabilities"])

    assert 0.5 < result["roc_auc"] <= 1


def test_evaluate_classifier_handles_all_one_class_predictions():
    result = evaluate_classifier(Y_TRUE, [0] * 6, Y_PROB)

    assert result["confusion_matrix"]["true_positives"] == 0
    assert result["precision"] == 0
    assert result["specificity"] == 1


@pytest.mark.parametrize("y_true, y_pred, y_prob, message", [
    ([0, 1, 1], [0, 1], [0.1, 0.9, 0.8], "same length"),
    ([0, 1, None], [0, 1, 1], [0.1, 0.9, 0.8], "missing"),
    ([0, 1, 2], [0, 1, 1], [0.1, 0.9, 0.8], "only 0 and 1"),
    ([0, 1, 1], [0, 1, 1], [0.1, 1.2, 0.8], "between 0 and 1"),
    ([1, 1, 1], [0, 1, 1], [0.1, 0.9, 0.8], "both classes"),
    ([], [], [], "empty"),
])
def test_evaluate_classifier_rejects_bad_inputs(y_true, y_pred, y_prob, message):
    with pytest.raises(DataValidationError, match=message):
        evaluate_classifier(y_true, y_pred, y_prob)


def test_evaluate_classifier_rejects_non_sequence_input():
    with pytest.raises(DataValidationError, match="Series, NumPy array or list"):
        evaluate_classifier("010", Y_PRED, Y_PROB)


def test_compare_models_builds_one_row_per_model():
    good = evaluate_classifier(Y_TRUE, Y_PRED, Y_PROB)
    perfect = evaluate_classifier(Y_TRUE, Y_TRUE, [0, 0, 0, 1, 1, 1])

    table = compare_models({"logistic": good, "random_forest": perfect})

    assert list(table.index) == ["logistic", "random_forest"]
    assert table.loc["random_forest", "accuracy"] == 1
    assert table.loc["logistic", "true_negatives"] == 2
    assert "confusion_matrix" not in table.columns


def test_compare_models_does_not_sort_or_rank():
    good = evaluate_classifier(Y_TRUE, Y_PRED, Y_PROB)
    perfect = evaluate_classifier(Y_TRUE, Y_TRUE, [0, 0, 0, 1, 1, 1])

    table = compare_models({"weaker": good, "stronger": perfect})

    assert list(table.index) == ["weaker", "stronger"]
    assert not {"rank", "best", "winner"} & set(table.columns)


def test_compare_models_rejects_missing_metrics():
    incomplete = evaluate_classifier(Y_TRUE, Y_PRED, Y_PROB)
    del incomplete["brier_score"]

    with pytest.raises(DataValidationError, match="brier_score"):
        compare_models({"logistic": incomplete})


@pytest.mark.parametrize("bad", [None, {}, {"logistic": [0.5]}])
def test_compare_models_rejects_bad_container(bad):
    with pytest.raises(DataValidationError):
        compare_models(bad)
