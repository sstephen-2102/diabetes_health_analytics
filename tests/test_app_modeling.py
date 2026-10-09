"""Tests for the modeling pages of the Streamlit app: service loaders, charts and page smoke tests.

Exports are generated from small synthetic CSVs, so the tests do not need the real outputs/ folder.
"""
import shutil

import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from app import artifacts, services
from app.components import charts
from src.analysis.export_duplicate_sensitivity import SCENARIOS, export_duplicate_sensitivity
from src.analysis.export_modeling import export_modeling
from src.analysis.export_regression import export_regression
from src.common.exceptions import DataLoadError, InvalidParameterError
from tests.test_export_modeling import MODELS
from tests.test_export_modeling import create_survey_csv as create_modeling_csv
from tests.test_export_regression import create_survey_csv as create_regression_csv

PAGES = ["machine_learning", "imbalance_threshold", "calibration", "interpretation", "regression_lab", "prediction"]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("app")
    output = folder / "outputs"
    modeling_csv = create_modeling_csv(folder / "modeling.csv")
    export_modeling(str(modeling_csv), str(output), check_reference=False, importance_sample=200)
    export_duplicate_sensitivity(str(modeling_csv), str(output), check_reference=False)
    export_regression(str(create_regression_csv(folder / "regression.csv")), str(output), check_reference=False)
    return output


LOADERS = [artifacts.model_artifacts, artifacts.modeling_outputs, artifacts.regression_outputs,
           artifacts.duplicate_sensitivity]


@pytest.fixture
def app_outputs(outputs, monkeypatch):
    """Point the app's cached loaders at the generated exports, with empty caches before and after."""
    monkeypatch.setattr(artifacts, "OUTPUTS", outputs)
    for loader in LOADERS:
        loader.clear()
    yield outputs
    for loader in LOADERS:
        loader.clear()


def run_page(page: str) -> AppTest:
    return AppTest.from_string(f"from app.pages import {page}\n{page}.render()", default_timeout=60).run()


def page_text(at: AppTest) -> str:
    parts = [e.value for kind in ("title", "subheader", "markdown", "caption", "info", "warning", "error")
             for e in getattr(at, kind)]
    return " ".join(str(p) for p in parts + [m.label + " " + m.value for m in at.metric])


# ---------------------------------------------------------------- services

def test_load_modeling_outputs_contract(outputs):
    loaded = services.load_modeling_outputs(str(outputs))

    assert loaded["status"] == "success"
    assert loaded["dataset_variant"] == "unique_profile_v1"
    assert loaded["metadata"]["models"] == MODELS
    assert loaded["metadata"]["leakage_check"] == "PASS"
    assert 0 < loaded["metadata"]["test_prevalence"] < 1
    results = loaded["results"]
    assert list(results["comparison"].index) == MODELS
    for kind in ["thresholds", "calibration", "importance", "roc", "precision_recall"]:
        assert list(results[kind]) == MODELS
        assert all(isinstance(t, pd.DataFrame) and not t.empty for t in results[kind].values())
    assert {"brier_score", "expected_calibration_error"} <= set(results["calibration_summary"].columns)


def test_test_prevalence_matches_confusion_counts(outputs):
    loaded = services.load_modeling_outputs(str(outputs))
    row = loaded["results"]["comparison"].iloc[0]

    assert loaded["metadata"]["test_prevalence"] == pytest.approx(
        (row["false_negatives"] + row["true_positives"]) / row["n"])


def test_load_regression_outputs_contract(outputs):
    loaded = services.load_regression_outputs(str(outputs))

    assert loaded["dataset_variant"] == "full_clean_v1"
    assert set(loaded["results"]) >= {"baseline_coefficients", "interaction_coefficients", "baseline_odds_ratios",
                                      "model_comparison", "bmi_odds_by_age", "interaction_term",
                                      "model_comparison_test"}
    assert loaded["warnings"]
    assert loaded["metadata"]["converged"] == {"baseline": True, "interaction": True}


def test_load_duplicate_sensitivity_contract(outputs):
    loaded = services.load_duplicate_sensitivity(str(outputs))
    results = loaded["results"]

    assert loaded["metadata"]["primary_scenario"] == "unique_grouped"
    assert list(results["scenarios"].index) == list(SCENARIOS)
    assert (results["all_test_rows"]["test_subset"] == "all").all()
    assert len(results["all_test_rows"]) == len(SCENARIOS) * len(MODELS)
    assert loaded["warnings"]


@pytest.mark.parametrize("loader, command", [(services.load_modeling_outputs, "export_modeling"),
                                             (services.load_regression_outputs, "export_regression"),
                                             (services.load_duplicate_sensitivity, "export_duplicate_sensitivity"),
                                             (services.load_model_artifacts, "export_modeling")])
def test_loaders_explain_missing_exports(tmp_path, loader, command):
    with pytest.raises(DataLoadError, match=f"{command} --csv data/raw/cdc_diabetes.csv"):
        loader(str(tmp_path))


def test_load_modeling_outputs_missing_table(outputs, tmp_path):
    copy = tmp_path / "outputs"
    shutil.copytree(outputs, copy)
    (copy / "tables" / "roc_curve_random_forest.csv").unlink()

    with pytest.raises(DataLoadError, match="roc_curve_random_forest"):
        services.load_modeling_outputs(str(copy))


def test_threshold_comparison_selects_grid_row_for_every_model(outputs):
    tables = services.load_modeling_outputs(str(outputs))["results"]["thresholds"]

    table = services.get_threshold_comparison(tables, 0.3)

    assert list(table.index) == MODELS
    assert table["threshold"].tolist() == pytest.approx([0.3] * len(MODELS))
    expected = tables["random_forest"].set_index("threshold").loc[0.3, "recall"]
    assert table.loc["random_forest", "recall"] == pytest.approx(expected)
    assert table["true_positives"].dtype.kind == "i"


def test_threshold_comparison_tolerates_float_rounding(outputs):
    tables = services.load_modeling_outputs(str(outputs))["results"]["thresholds"]

    assert len(services.get_threshold_comparison(tables, 0.1 + 0.05 * 3)) == len(MODELS)


def test_threshold_comparison_rejects_threshold_off_grid(outputs):
    tables = services.load_modeling_outputs(str(outputs))["results"]["thresholds"]

    with pytest.raises(InvalidParameterError, match="not in the exported grid"):
        services.get_threshold_comparison(tables, 0.33)


# ---------------------------------------------------------------- charts

def test_every_chart_builds_from_exported_tables(outputs):
    modeling = services.load_modeling_outputs(str(outputs))
    results, comparison = modeling["results"], modeling["results"]["comparison"]
    regression = services.load_regression_outputs(str(outputs))["results"]

    figures = [
        charts.roc_chart(results["roc"], comparison["roc_auc"].to_dict()),
        charts.precision_recall_chart(results["precision_recall"], comparison["pr_auc"].to_dict(), 0.2),
        charts.confusion_matrix_chart(comparison.loc["logistic_regression"]),
        charts.threshold_chart(results["thresholds"]["logistic_regression"], 0.5),
        charts.imbalance_chart(results["imbalance"], ["precision", "recall"]),
        charts.calibration_chart(results["calibration"]),
        charts.importance_chart(results["importance"]["gradient_boosting"], top_n=5),
        charts.odds_ratio_chart(regression["baseline_odds_ratios"], "Odds ratios"),
        charts.bmi_by_age_chart(regression["bmi_odds_by_age"]),
    ]

    assert all(isinstance(f, go.Figure) and f.data for f in figures)


def test_roc_chart_has_one_line_per_model_plus_no_skill(outputs):
    results = services.load_modeling_outputs(str(outputs))["results"]
    fig = charts.roc_chart(results["roc"], results["comparison"]["roc_auc"].to_dict())

    assert [t.name.split(" (")[0] for t in fig.data] == ["Logistic regression", "Random forest",
                                                         "Gradient boosting", "No skill"]


def test_confusion_matrix_chart_places_counts():
    fig = charts.confusion_matrix_chart({"true_negatives": 50, "false_positives": 5,
                                         "false_negatives": 10, "true_positives": 35})

    assert [list(row) for row in fig.data[0].z] == [[50, 5], [10, 35]]


def test_calibration_chart_hides_sparse_bins():
    table = pd.DataFrame({"mean_predicted": [0.1, 0.5, 0.9], "observed_frequency": [0.1, 0.4, 0.7],
                          "count": [500, 60, 10]})

    fig = charts.calibration_chart({"logistic_regression": table}, min_count=50)

    assert list(fig.data[0].x) == [0.1, 0.5]


def test_importance_chart_shows_top_n_largest_first_at_top():
    table = pd.DataFrame({"feature": list("abcdef"), "importance_mean": [0.1, 0.5, 0.3, 0.0, 0.2, 0.4],
                          "importance_std": [0.01] * 6})

    fig = charts.importance_chart(table, top_n=3)

    assert list(fig.data[0].y) == ["c", "f", "b"]


def test_odds_ratio_chart_uses_asymmetric_confidence_intervals():
    table = pd.DataFrame({"feature": ["x"], "odds_ratio": [2.0], "ci_lower": [1.5], "ci_upper": [3.0]})

    error = charts.odds_ratio_chart(table, "t").data[0].error_x

    assert list(error.array) == [1.0]
    assert list(error.arrayminus) == [0.5]


# ---------------------------------------------------------------- pages

@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors_and_names_dataset_variant(app_outputs, page):
    at = run_page(page)

    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert "Dataset variant" in page_text(at)


@pytest.mark.parametrize("page", PAGES)
def test_page_never_uses_diagnostic_language(app_outputs, page):
    text = page_text(run_page(page)).lower()

    for phrase in ["you have diabetes", "you are diagnosed", "diagnosed you"]:
        assert phrase not in text


@pytest.mark.parametrize("page, command", [("machine_learning", "export_modeling"),
                                           ("regression_lab", "export_regression"),
                                           ("prediction", "export_modeling")])
def test_page_explains_missing_exports(tmp_path, monkeypatch, page, command):
    monkeypatch.setattr(artifacts, "OUTPUTS", tmp_path)
    for loader in LOADERS:
        loader.clear()

    at = run_page(page)

    assert not at.exception
    assert command in at.error[0].value


def test_machine_learning_page_shows_duplicate_sensitivity(app_outputs):
    at = run_page("machine_learning")
    text = page_text(at)

    assert "Duplicate-handling sensitivity" in text
    assert "full_clean_v1 and unique_profile_v1" in text
    metric = next(s for s in at.selectbox if s.label == "Metric (all test rows)")
    metric.set_value("Brier score").run()
    assert not at.exception
    assert len(at.dataframe) == 3


def test_machine_learning_page_works_without_sensitivity_export(outputs, tmp_path, monkeypatch):
    copy = tmp_path / "outputs"
    shutil.copytree(outputs, copy)
    (copy / "duplicate_sensitivity_manifest.json").unlink()
    monkeypatch.setattr(artifacts, "OUTPUTS", copy)
    for loader in LOADERS:
        loader.clear()

    at = run_page("machine_learning")

    assert not at.exception
    assert not at.error
    assert any("export_duplicate_sensitivity" in i.value for i in at.info)
    assert "Confusion matrix" in page_text(at)


def test_threshold_slider_updates_threshold_metrics(app_outputs):
    at = run_page("imbalance_threshold")
    recall_at = {}
    for threshold in (0.1, 0.9):
        at.select_slider[0].set_value(threshold).run()
        recall_at[threshold] = float(next(m.value for m in at.metric if m.label == "Recall"))

    assert recall_at[0.1] >= recall_at[0.9]


def test_prediction_page_shows_spec_wording_and_disclaimer(app_outputs):
    at = run_page("prediction")
    labels = [m.label for m in at.metric]

    assert "Model-estimated probability of the positive Diabetes_binary class" in labels
    assert any(label.startswith("Model classification at selected threshold") for label in labels)
    assert "not a medical diagnosis" in at.warning[0].value


def test_prediction_page_responds_to_inputs_and_threshold(app_outputs):
    at = run_page("prediction")

    def shown(prefix):
        return next(m.value for m in at.metric if m.label.startswith(prefix))

    before = shown("Model-estimated")
    high_bp = next(s for s in at.selectbox if s.label == "High blood pressure")
    high_bp.set_value(1 - high_bp.value).run()
    assert shown("Model-estimated") != before

    for threshold in (0.05, 0.95):
        at.slider[0].set_value(threshold).run()
        probability = float(shown("Model-estimated").rstrip("%")) / 100
        assert shown("Model classification") == str(int(probability >= threshold))


def test_prediction_page_lists_defaulted_inputs(app_outputs):
    text = page_text(run_page("prediction"))

    assert "typical training-set values" in text
    assert "Income" in text


def test_app_navigation_loads(app_outputs):
    at = AppTest.from_file("../streamlit_app.py", default_timeout=60).run()

    assert not at.exception, [e.message for e in at.exception]
