"""Tests for chart-data helpers in src/analysis/visualization.py."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.visualization import prepare_precision_recall_data, prepare_roc_curve_data
from src.common.exceptions import DataValidationError

Y = [0, 0, 1, 1, 0, 1, 0, 1]
P = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.7]


# ---------------------------------------------------------------- ROC

def test_roc_returns_expected_columns():
    table = prepare_roc_curve_data(Y, P)

    assert list(table.columns) == ["false_positive_rate", "true_positive_rate", "threshold"]


def test_roc_runs_from_origin_to_top_right():
    table = prepare_roc_curve_data(Y, P)

    assert (table.iloc[0][["false_positive_rate", "true_positive_rate"]] == 0).all()
    assert (table.iloc[-1][["false_positive_rate", "true_positive_rate"]] == 1).all()
    assert table["false_positive_rate"].is_monotonic_increasing
    assert table["true_positive_rate"].is_monotonic_increasing


def test_roc_first_threshold_is_nan_not_infinite():
    table = prepare_roc_curve_data(Y, P)

    assert np.isnan(table["threshold"].iloc[0])
    assert np.isfinite(table["threshold"].iloc[1:]).all()


def test_roc_area_matches_sklearn():
    table = prepare_roc_curve_data(Y, P)

    area = np.trapezoid(table["true_positive_rate"], table["false_positive_rate"])
    assert area == pytest.approx(roc_auc_score(Y, P))


# ---------------------------------------------------------------- precision-recall

def test_pr_returns_expected_columns_with_aligned_lengths():
    table = prepare_precision_recall_data(Y, P)

    assert list(table.columns) == ["precision", "recall", "threshold"]
    assert table["threshold"].isna().sum() == 1
    assert np.isnan(table["threshold"].iloc[-1])


def test_pr_ends_at_recall_zero_precision_one():
    table = prepare_precision_recall_data(Y, P)

    assert table["recall"].iloc[-1] == 0
    assert table["precision"].iloc[-1] == 1
    assert table["recall"].iloc[0] == 1


def test_pr_average_precision_matches_sklearn():
    table = prepare_precision_recall_data(Y, P)

    recall_steps = -np.diff(table["recall"].to_numpy())
    assert (recall_steps * table["precision"].to_numpy()[:-1]).sum() == pytest.approx(average_precision_score(Y, P))


# ---------------------------------------------------------------- shared input handling

@pytest.mark.parametrize("prepare", [prepare_roc_curve_data, prepare_precision_recall_data])
@pytest.mark.parametrize("convert", [list, np.array, pd.Series])
def test_accepts_lists_arrays_and_series(prepare, convert):
    table = prepare(convert(Y), convert(P))

    assert len(table) > 1


@pytest.mark.parametrize("prepare", [prepare_roc_curve_data, prepare_precision_recall_data])
@pytest.mark.parametrize("y_true, probabilities, message", [
    ([0, 1, 1], [0.1, 0.9], "equal length"),
    ([], [], "non-empty"),
    ([0, 1, None], [0.1, 0.9, 0.5], "missing"),
    ([0, 1, 1], [0.1, np.nan, 0.5], "missing"),
    ([0, 1, 2], [0.1, 0.9, 0.5], "only 0 and 1"),
    ([0, 1, 1], [0.1, 1.5, 0.5], "between 0 and 1"),
    ([1, 1, 1], [0.1, 0.9, 0.5], "both classes"),
    ([0, 1], ["a", "b"], "numeric"),
    ([[0, 1]], [[0.1, 0.9]], "1-D"),
])
def test_rejects_bad_inputs(prepare, y_true, probabilities, message):
    with pytest.raises(DataValidationError, match=message):
        prepare(y_true, probabilities)
