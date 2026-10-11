"""Tests for the survey-data pages of the Streamlit app: the Member 1 pages, plus shared checks for the
Member 2 pages (page-specific Member 2 tests are in test_app_statistics.py).

A small synthetic CSV with some repeated rows stands in for the survey data, so no download is needed.
"""
import hashlib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import artifacts, services
from app.components import charts
from src.common.config import FEATURE_COLUMNS, TARGET_DEFINITION
from src.common.exceptions import DataLoadError
from src.data import loader
from tests.test_export_modeling import create_survey_csv

PAGES = ["dashboard", "data_explorer", "exploratory_analysis", "probability_explorer", "statistical_testing"]
REPEATED_ROWS = 30


def create_csv_with_repeats(path):
    create_survey_csv(path)
    data = pd.read_csv(path)
    pd.concat([data, data.head(REPEATED_ROWS)]).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def survey_csv(tmp_path_factory):
    return create_csv_with_repeats(tmp_path_factory.mktemp("survey") / "survey.csv")


@pytest.fixture
def app_data(survey_csv, tmp_path, monkeypatch):
    """Point the app at the synthetic CSV, with no modeling exports, and empty caches before and after."""
    monkeypatch.setattr(artifacts, "RAW_CSV", survey_csv)
    monkeypatch.setattr(artifacts, "OUTPUTS", tmp_path)
    for cached in (artifacts.eda_data, artifacts.modeling_outputs):
        cached.clear()
    yield survey_csv
    for cached in (artifacts.eda_data, artifacts.modeling_outputs):
        cached.clear()


def run_page(page: str) -> AppTest:
    return AppTest.from_string(f"from app.pages import {page}\n{page}.render()", default_timeout=60).run()


def page_text(at: AppTest) -> str:
    parts = [e.value for kind in ("title", "subheader", "markdown", "caption", "info", "warning", "error")
             for e in getattr(at, kind)]
    return " ".join(str(p) for p in parts + [m.label + " " + m.value for m in at.metric])


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


# ---------------------------------------------------------------- services

def test_load_eda_data_returns_both_variants_with_context(survey_csv):
    result = services.load_eda_data(str(survey_csv))

    assert result["status"] == "success"
    assert set(result["results"]["variants"]) == {"full_clean_v1", "unique_profile_v1"}
    assert result["target_definition"] == TARGET_DEFINITION
    assert result["dataset_variant"] == "full_clean_v1 and unique_profile_v1"
    assert result["metadata"]["dataset_version_sha256"] == hashlib.sha256(survey_csv.read_bytes()).hexdigest()
    assert result["results"]["raw_validation"]["duplicate_rows"] == REPEATED_ROWS
    full, unique = (result["results"]["variants"][name]["data"] for name in ("full_clean_v1", "unique_profile_v1"))
    assert len(full) - len(unique) == REPEATED_ROWS
    assert "change_unique_minus_full" in result["results"]["sensitivity"]
    assert any("survey weights" in w for w in result["warnings"])


def test_load_eda_data_downloads_a_missing_csv(survey_csv, tmp_path, monkeypatch):
    target = tmp_path / "raw" / "cdc_diabetes.csv"
    calls = []

    def fake_download(path):
        calls.append(path)
        target.parent.mkdir(parents=True)
        target.write_bytes(survey_csv.read_bytes())
        return path

    monkeypatch.setattr(loader, "download_uci_dataset", fake_download)

    result = services.load_eda_data(str(target))

    assert calls == [str(target)]
    assert result["metadata"]["source_file"] == "cdc_diabetes.csv"


def test_load_eda_data_explains_failed_download(tmp_path, monkeypatch):
    def failing_download(path):
        raise DataLoadError("UCI acquisition failed: offline")

    monkeypatch.setattr(loader, "download_uci_dataset", failing_download)

    with pytest.raises(DataLoadError, match="Cannot prepare the survey dataset.*offline"):
        services.load_eda_data(str(tmp_path / "missing.csv"))


def test_load_eda_data_rejects_invalid_schema(survey_csv, tmp_path):
    path = tmp_path / "invalid.csv"
    pd.read_csv(survey_csv).drop(columns="Age").to_csv(path, index=False)

    with pytest.raises(DataLoadError, match="Cannot prepare the survey dataset"):
        services.load_eda_data(str(path))


def test_value_distribution_gives_percentages_within_each_target_group():
    data = pd.DataFrame({"BMI": [20, 20, 30, 30, 30, 40], "Diabetes_binary": [0, 0, 0, 1, 1, 1]})

    table = services.get_value_distribution(data, "BMI")

    assert list(table.index) == [20, 30, 40]
    assert table[0].tolist() == pytest.approx([200 / 3, 100 / 3, 0])
    assert table[1].tolist() == pytest.approx([0, 200 / 3, 100 / 3])


def test_value_distribution_rejects_unknown_feature():
    with pytest.raises(Exception, match="Weight"):
        services.get_value_distribution(pd.DataFrame({"Diabetes_binary": [0, 1]}), "Weight")


# ---------------------------------------------------------------- charts

def test_prevalence_chart_marks_overall_share():
    prevalence = pd.DataFrame({"category": [0, 1], "label": ["No", "Yes"], "total_observations": [80, 20],
                               "target_count": [8, 10], "target_percentage": [10.0, 50.0]})

    fig = charts.prevalence_chart(prevalence, 18.0, "title")

    assert list(fig.data[0].x) == [10.0, 50.0]
    assert fig.layout.shapes[0].x0 == 18.0


def test_correlation_heatmap_uses_readable_labels_and_fixed_scale():
    matrix = pd.DataFrame([[1, 0.5], [0.5, 1]], index=["BMI", "Age"], columns=["BMI", "Age"])

    fig = charts.correlation_heatmap(matrix, {"BMI": "Body mass index", "Age": "Age group"})

    assert list(fig.data[0].x) == ["Body mass index", "Age group"]
    assert (fig.data[0].zmin, fig.data[0].zmax) == (-1, 1)


def test_value_distribution_chart_has_one_line_per_target_group():
    table = pd.DataFrame({0: [50.0, 50.0], 1: [25.0, 75.0]}, index=[20, 30])

    fig = charts.value_distribution_chart(table, {0: "Neither", 1: "Either"}, "title", "BMI")

    assert [trace.name for trace in fig.data] == ["Neither", "Either"]


# ---------------------------------------------------------------- pages

@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors_and_names_dataset_variant(app_data, page):
    at = run_page(page)

    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert "Dataset variant" in page_text(at)


@pytest.mark.parametrize("page", PAGES)
def test_page_never_uses_diagnostic_or_causal_language(app_data, page):
    text = page_text(run_page(page)).lower()

    for phrase in ["you have diabetes", "you are diagnosed", "diagnosed you", " causes ", " prevents "]:
        assert phrase not in text


@pytest.mark.parametrize("page", PAGES)
def test_page_explains_missing_data(tmp_path, monkeypatch, page):
    def failing_download(path):
        raise DataLoadError("UCI acquisition failed: offline")

    monkeypatch.setattr(loader, "download_uci_dataset", failing_download)
    monkeypatch.setattr(artifacts, "RAW_CSV", tmp_path / "missing.csv")
    artifacts.eda_data.clear()

    at = run_page(page)
    artifacts.eda_data.clear()

    assert not at.exception
    assert "Cannot prepare the survey dataset" in at.error[0].value


def test_dashboard_lists_research_questions_and_methodology_notes(app_data):
    text = page_text(run_page("dashboard"))

    for number in range(1, 14):
        assert f"**RQ{number}**" in text
    assert "not U.S. population prevalence" in text
    assert "not a medical diagnosis" in text
    assert "Selected findings" in text


def test_dashboard_skips_model_finding_without_modeling_exports(app_data):
    text = page_text(run_page("dashboard"))

    assert "ROC AUC" not in text
    assert "Mean BMI" in text


def test_data_explorer_switches_variant(app_data):
    at = run_page("data_explorer")
    full_rows = int(metric(at, "Rows").replace(",", ""))
    assert metric(at, "Repeated rows") == f"{REPEATED_ROWS:,}"

    at.radio[0].set_value("unique_profile_v1").run()

    assert not at.exception
    assert int(metric(at, "Rows").replace(",", "")) == full_rows - REPEATED_ROWS
    assert metric(at, "Repeated rows") == "0"
    assert f"{REPEATED_ROWS:,} repeated rows were removed" in page_text(at)


def test_data_explorer_shows_category_mapping_for_selected_variable(app_data):
    at = run_page("data_explorer")
    at.selectbox[0].set_value("Education").run()

    mapping = at.dataframe[1].value
    assert not at.exception
    assert mapping["label"].iloc[0] == "Never attended school / kindergarten only"
    assert mapping["percentage"].sum() == pytest.approx(100)


@pytest.mark.parametrize("variant", ["full_clean_v1", "unique_profile_v1"])
def test_exploratory_analysis_renders_every_feature(app_data, variant):
    at = run_page("exploratory_analysis")
    at.radio[0].set_value(variant).run()

    for feature in FEATURE_COLUMNS:
        at.selectbox[0].set_value(feature).run()
        assert not at.exception, (feature, [e.message for e in at.exception])


def test_exploratory_analysis_numeric_feature_shows_summary_metrics(app_data):
    at = run_page("exploratory_analysis")
    at.selectbox[0].set_value("BMI").run()

    labels = [m.label for m in at.metric]
    assert {"Mean", "Median", "IQR"} <= set(labels)
