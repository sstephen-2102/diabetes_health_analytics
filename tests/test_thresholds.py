import numpy as np
import pandas as pd
import pytest

from src.modeling.thresholds import evaluate_thresholds, generate_threshold_curve
from src.common.exceptions import DataValidationError


@pytest.fixture
def labels_and_probabilities():
    y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.40, 0.60, 0.35, 0.55, 0.80, 0.95])
    return y_true, probabilities


@pytest.fixture
def default_grid():
    return [float(t) for t in np.arange(0.10, 0.91, 0.05).round(2)]


# ---------------------------------------------------------------------------
# evaluate_thresholds — happy path
# ---------------------------------------------------------------------------

def test_evaluate_returns_one_row_per_threshold(labels_and_probabilities, default_grid):
    y_true, probabilities = labels_and_probabilities
    result = evaluate_thresholds(y_true, probabilities, default_grid)
    assert len(result) == len(default_grid)


def test_evaluate_has_required_columns(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    result = evaluate_thresholds(y_true, probabilities, [0.5])
    expected = {"threshold", "precision", "recall", "specificity", "f1",
                "true_negatives", "false_positives", "false_negatives", "true_positives"}
    assert expected.issubset(result.columns)


def test_evaluate_known_values_at_half(labels_and_probabilities):
    # At 0.5: predicted positive = 0.60, 0.55, 0.80, 0.95 -> TP=3, FP=1, TN=3, FN=1
    y_true, probabilities = labels_and_probabilities
    row = evaluate_thresholds(y_true, probabilities, [0.5]).iloc[0]
    assert row["true_positives"] == 3
    assert row["false_positives"] == 1
    assert row["true_negatives"] == 3
    assert row["false_negatives"] == 1
    assert row["precision"] == pytest.approx(0.75)
    assert row["recall"] == pytest.approx(0.75)
    assert row["specificity"] == pytest.approx(0.75)
    assert row["f1"] == pytest.approx(0.75)


def test_evaluate_confusion_counts_sum_to_n(labels_and_probabilities, default_grid):
    y_true, probabilities = labels_and_probabilities
    result = evaluate_thresholds(y_true, probabilities, default_grid)
    totals = result[["true_negatives", "false_positives",
                     "false_negatives", "true_positives"]].sum(axis=1)
    assert (totals == len(y_true)).all()


def test_evaluate_output_sorted_by_threshold(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    result = evaluate_thresholds(y_true, probabilities, [0.7, 0.2, 0.5])
    assert result["threshold"].tolist() == [0.2, 0.5, 0.7]


def test_evaluate_recall_non_increasing(labels_and_probabilities, default_grid):
    y_true, probabilities = labels_and_probabilities
    recall = evaluate_thresholds(y_true, probabilities, default_grid)["recall"]
    assert recall.is_monotonic_decreasing


def test_evaluate_high_threshold_predicts_nothing_positive(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    row = evaluate_thresholds(y_true, probabilities, [0.99]).iloc[0]
    assert row["true_positives"] == 0
    assert row["false_positives"] == 0
    assert row["precision"] == 0.0
    assert row["specificity"] == 1.0


def test_evaluate_accepts_series_probabilities(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    result = evaluate_thresholds(y_true, pd.Series(probabilities), [0.5])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# evaluate_thresholds — validation errors
# ---------------------------------------------------------------------------

def test_evaluate_rejects_non_series_labels(labels_and_probabilities):
    _, probabilities = labels_and_probabilities
    with pytest.raises(DataValidationError):
        evaluate_thresholds([0, 1, 0, 1, 0, 1, 0, 1], probabilities, [0.5])


def test_evaluate_rejects_empty_thresholds(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    with pytest.raises(DataValidationError):
        evaluate_thresholds(y_true, probabilities, [])


def test_evaluate_rejects_threshold_out_of_range(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    with pytest.raises(DataValidationError):
        evaluate_thresholds(y_true, probabilities, [1.5])


def test_evaluate_rejects_length_mismatch(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    with pytest.raises(DataValidationError):
        evaluate_thresholds(y_true, probabilities[:-1], [0.5])


def test_evaluate_rejects_probability_out_of_range(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    bad = probabilities.copy()
    bad[0] = 1.2
    with pytest.raises(DataValidationError):
        evaluate_thresholds(y_true, bad, [0.5])


def test_evaluate_rejects_non_binary_labels(labels_and_probabilities):
    _, probabilities = labels_and_probabilities
    y_bad = pd.Series([0, 1, 2, 0, 1, 1, 0, 1])
    with pytest.raises(DataValidationError):
        evaluate_thresholds(y_bad, probabilities, [0.5])


# ---------------------------------------------------------------------------
# generate_threshold_curve
# ---------------------------------------------------------------------------

def test_curve_returns_plotting_columns(labels_and_probabilities, default_grid):
    y_true, probabilities = labels_and_probabilities
    results = evaluate_thresholds(y_true, probabilities, default_grid)
    curve = generate_threshold_curve(results)
    assert curve.columns.tolist() == ["threshold", "precision", "recall", "specificity", "f1"]


def test_curve_sorted_by_threshold(labels_and_probabilities, default_grid):
    y_true, probabilities = labels_and_probabilities
    results = evaluate_thresholds(y_true, probabilities, default_grid)
    shuffled = results.sample(frac=1, random_state=0)
    curve = generate_threshold_curve(shuffled)
    assert curve["threshold"].is_monotonic_increasing


def test_curve_rejects_non_dataframe():
    with pytest.raises(DataValidationError):
        generate_threshold_curve({"threshold": [0.5]})


def test_curve_rejects_missing_columns():
    with pytest.raises(DataValidationError):
        generate_threshold_curve(pd.DataFrame({"threshold": [0.5], "precision": [0.7]}))
