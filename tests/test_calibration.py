import numpy as np
import pandas as pd
import pytest

from src.modeling.calibration import calculate_calibration_metrics, generate_calibration_data
from src.common.exceptions import DataValidationError, InvalidParameterError


@pytest.fixture
def labels_and_probabilities():
    # Two populated bins with n_bins=2: [0, 0.5) and [0.5, 1.0]
    y_true = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    return y_true, probabilities


# ---------------------------------------------------------------------------
# generate_calibration_data
# ---------------------------------------------------------------------------

def test_data_has_required_columns(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    table = generate_calibration_data(y_true, probabilities, n_bins=2)
    assert table.columns.tolist() == ["bin", "bin_lower", "bin_upper",
                                      "mean_predicted", "observed_frequency", "count"]


def test_data_known_values(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    table = generate_calibration_data(y_true, probabilities, n_bins=2)
    low, high = table.iloc[0], table.iloc[1]
    assert low["mean_predicted"] == pytest.approx(0.25)
    assert low["observed_frequency"] == pytest.approx(0.25)
    assert low["count"] == 4
    assert high["mean_predicted"] == pytest.approx(0.75)
    assert high["observed_frequency"] == pytest.approx(0.75)
    assert high["count"] == 4


def test_data_counts_sum_to_n(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    table = generate_calibration_data(y_true, probabilities, n_bins=10)
    assert table["count"].sum() == len(y_true)


def test_data_omits_empty_bins():
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.05, 0.06, 0.95, 0.96])
    table = generate_calibration_data(y_true, probabilities, n_bins=10)
    assert table["bin"].tolist() == [0, 9]


def test_data_probability_of_one_lands_in_last_bin():
    y_true = pd.Series([0, 1])
    probabilities = np.array([0.0, 1.0])
    table = generate_calibration_data(y_true, probabilities, n_bins=10)
    assert table["bin"].tolist() == [0, 9]
    assert table.iloc[-1]["bin_upper"] == pytest.approx(1.0)


def test_data_bin_edges_match_bin_index(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    table = generate_calibration_data(y_true, probabilities, n_bins=4)
    assert (table["bin_lower"] == table["bin"] / 4).all()
    assert (table["bin_upper"] == (table["bin"] + 1) / 4).all()


def test_data_accepts_series_probabilities(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    table = generate_calibration_data(y_true, pd.Series(probabilities), n_bins=2)
    assert len(table) == 2


# ---------------------------------------------------------------------------
# calculate_calibration_metrics
# ---------------------------------------------------------------------------

def test_metrics_returns_required_keys(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    result = calculate_calibration_metrics(y_true, probabilities, n_bins=2)
    assert {"brier_score", "expected_calibration_error", "calibration_table",
            "n_bins", "n"}.issubset(result.keys())


def test_metrics_brier_score_known_value(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    result = calculate_calibration_metrics(y_true, probabilities, n_bins=2)
    expected = float(np.mean((probabilities - y_true.to_numpy()) ** 2))
    assert result["brier_score"] == pytest.approx(expected)


def test_metrics_ece_zero_when_perfectly_calibrated(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    result = calculate_calibration_metrics(y_true, probabilities, n_bins=2)
    assert result["expected_calibration_error"] == pytest.approx(0.0)


def test_metrics_ece_detects_overconfidence():
    # Model says 0.9 for everyone, but only half are positive
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.9, 0.9, 0.9, 0.9])
    result = calculate_calibration_metrics(y_true, probabilities, n_bins=10)
    assert result["expected_calibration_error"] == pytest.approx(0.4)


def test_metrics_n_matches_input(labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    result = calculate_calibration_metrics(y_true, probabilities)
    assert result["n"] == len(y_true)
    assert result["n_bins"] == 10


# ---------------------------------------------------------------------------
# Validation errors (shared by both functions)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", [generate_calibration_data, calculate_calibration_metrics])
def test_rejects_non_series_labels(func, labels_and_probabilities):
    _, probabilities = labels_and_probabilities
    with pytest.raises(DataValidationError):
        func([0, 0, 0, 1, 0, 1, 1, 1], probabilities)


@pytest.mark.parametrize("func", [generate_calibration_data, calculate_calibration_metrics])
def test_rejects_length_mismatch(func, labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    with pytest.raises(DataValidationError):
        func(y_true, probabilities[:-1])


@pytest.mark.parametrize("func", [generate_calibration_data, calculate_calibration_metrics])
def test_rejects_probability_out_of_range(func, labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    bad = probabilities.copy()
    bad[0] = -0.1
    with pytest.raises(DataValidationError):
        func(y_true, bad)


@pytest.mark.parametrize("func", [generate_calibration_data, calculate_calibration_metrics])
def test_rejects_non_binary_labels(func, labels_and_probabilities):
    _, probabilities = labels_and_probabilities
    with pytest.raises(DataValidationError):
        func(pd.Series([0, 1, 2, 0, 1, 1, 0, 1]), probabilities)


@pytest.mark.parametrize("func", [generate_calibration_data, calculate_calibration_metrics])
@pytest.mark.parametrize("bad_bins", [0, -3, 2.5])
def test_rejects_invalid_n_bins(func, bad_bins, labels_and_probabilities):
    y_true, probabilities = labels_and_probabilities
    with pytest.raises(InvalidParameterError):
        func(y_true, probabilities, n_bins=bad_bins)
