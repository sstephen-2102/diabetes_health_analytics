"""Tests for src/analysis/model_figures.py: structure of each figure, not pixels."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.analysis.model_figures import (MIN_CALIBRATION_COUNT, bmi_by_age_figure, calibration_figure,
                                        importance_figure, odds_ratio_figure, precision_recall_figure,
                                        roc_figure, threshold_figure)
from src.analysis.visualization import prepare_precision_recall_data, prepare_roc_curve_data
from src.modeling.thresholds import evaluate_thresholds

MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]
RNG = np.random.default_rng(0)
Y = RNG.integers(0, 2, 400)
PROBS = {m: np.clip(Y * 0.3 + RNG.random(400) * 0.7, 0, 1) for m in MODELS}


def line_labels(ax):
    return [line.get_label() for line in ax.get_lines()]


def test_roc_figure_has_one_line_per_model_plus_reference():
    curves = {m: prepare_roc_curve_data(Y, p) for m, p in PROBS.items()}

    fig = roc_figure(curves, {m: 0.8 for m in MODELS})

    labels = line_labels(fig.axes[0])
    assert isinstance(fig, Figure)
    assert len(labels) == 4
    assert labels[0] == "Logistic Regression (AUC 0.800)"
    assert labels[-1] == "No skill"


def test_precision_recall_figure_shows_prevalence_baseline():
    curves = {m: prepare_precision_recall_data(Y, p) for m, p in PROBS.items()}

    fig = precision_recall_figure(curves, {m: 0.4 for m in MODELS}, prevalence=0.153)

    assert "No skill (0.153)" in line_labels(fig.axes[0])
    assert fig.axes[0].get_ylim() == (0, 1)


def test_calibration_figure_hides_sparse_bins():
    table = pd.DataFrame({"mean_predicted": [0.1, 0.5, 0.9], "observed_frequency": [0.1, 0.5, 1.0],
                          "count": [500, MIN_CALIBRATION_COUNT, MIN_CALIBRATION_COUNT - 1]})

    fig = calibration_figure({"gradient_boosting": table})

    plotted = fig.axes[0].get_lines()[0].get_xdata()
    assert list(plotted) == [0.1, 0.5]
    assert str(MIN_CALIBRATION_COUNT) in fig.axes[0].get_title()


def test_threshold_figure_has_one_panel_per_model_and_shared_legend():
    tables = {m: evaluate_thresholds(pd.Series(Y), p, [0.1, 0.3, 0.5, 0.7, 0.9]) for m, p in PROBS.items()}

    fig = threshold_figure(tables, default_threshold=0.5)

    assert len(fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes] == ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Precision", "Recall", "Specificity", "F1"]


def test_threshold_figure_leaves_precision_blank_when_nothing_predicted_positive():
    table = evaluate_thresholds(pd.Series(Y), PROBS["gradient_boosting"], [0.1, 0.5, 1.0])
    assert table.iloc[-1]["true_positives"] + table.iloc[-1]["false_positives"] == 0

    fig = threshold_figure({"gradient_boosting": table}, default_threshold=0.5)

    precision = fig.axes[0].get_lines()[0].get_ydata()
    assert np.isnan(precision[-1])
    assert not np.isnan(precision[:-1]).any()


def test_importance_figure_shows_top_n_features():
    table = pd.DataFrame({"feature": [f"f{i}" for i in range(15)],
                          "importance_mean": np.linspace(0.05, 0.0, 15), "importance_std": 0.001})

    fig = importance_figure({m: table for m in MODELS}, top_n=5)

    assert len(fig.axes) == 3
    shown = [label.get_text() for label in fig.axes[0].get_yticklabels()]
    assert shown == ["f4", "f3", "f2", "f1", "f0"]


ODDS = pd.DataFrame({"feature": ["HighBP", "Income", "HvyAlcoholConsump"], "odds_ratio": [2.1, 0.95, 0.46],
                     "ci_lower": [2.07, 0.94, 0.43], "ci_upper": [2.2, 0.96, 0.5], "p_value": [0, 0, 0]})


def test_odds_ratio_figure_is_sorted_with_plain_log_ticks():
    fig = odds_ratio_figure(ODDS)
    ax = fig.axes[0]

    assert [label.get_text() for label in ax.get_yticklabels()] == ["HvyAlcoholConsump", "Income", "HighBP"]
    assert ax.get_xscale() == "log"
    tick_text = [label.get_text() for label in ax.get_xticklabels()]
    assert "1" in tick_text and "2" in tick_text
    assert not any("10^" in text or "times" in text for text in tick_text)


def test_bmi_by_age_figure_has_one_point_per_age_band():
    table = pd.DataFrame({"age_group": ["18-24", "25-29", "80+"], "bmi_odds_ratio": [1.03, 1.04, 1.09],
                          "ci_lower": [1.02, 1.03, 1.08], "ci_upper": [1.04, 1.05, 1.10]})

    fig = bmi_by_age_figure(table)

    assert len(fig.axes[0].get_lines()[0].get_xdata()) == 3


@pytest.mark.parametrize("build", [
    lambda: roc_figure({m: prepare_roc_curve_data(Y, p) for m, p in PROBS.items()}, {m: 0.8 for m in MODELS}),
    lambda: odds_ratio_figure(ODDS),
])
def test_figures_save_to_png(tmp_path, build):
    path = tmp_path / "figure.png"

    build().savefig(path, dpi=60)

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
