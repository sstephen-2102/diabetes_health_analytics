"""Tests for the one-command export pipeline."""
import pytest

from src.analysis import pipeline
from src.common.exceptions import InvalidParameterError
from tests.test_export_modeling import create_survey_csv


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "STEPS", {name: (lambda csv, output, check, name=name:
                                                   calls.append((name, csv, output, check)) or {"step": name})
                                            for name in pipeline.STEPS})
    return calls


def test_pipeline_runs_every_step_in_order(tmp_path, recorded):
    csv = create_survey_csv(tmp_path / "survey.csv")

    results = pipeline.run_analysis_pipeline(str(csv), str(tmp_path / "out"), check_reference=False)

    assert [call[0] for call in recorded] == ["eda", "statistics", "regression", "modeling", "duplicate_sensitivity"]
    assert all(call[1:] == (str(csv), str(tmp_path / "out"), False) for call in recorded)
    assert list(results) == [call[0] for call in recorded]


def test_pipeline_runs_selected_steps_only(tmp_path, recorded):
    csv = create_survey_csv(tmp_path / "survey.csv")

    pipeline.run_analysis_pipeline(str(csv), steps=["modeling", "eda"])

    assert [call[0] for call in recorded] == ["modeling", "eda"]


@pytest.mark.parametrize("steps", [[], ["anova"]])
def test_pipeline_rejects_unknown_or_empty_steps(tmp_path, steps):
    with pytest.raises(InvalidParameterError, match="steps"):
        pipeline.run_analysis_pipeline(str(tmp_path / "survey.csv"), steps=steps)


def test_pipeline_statistics_step_writes_tables(tmp_path):
    csv = create_survey_csv(tmp_path / "survey.csv")

    results = pipeline.run_analysis_pipeline(str(csv), str(tmp_path / "out"), check_reference=False,
                                             steps=["statistics"])

    assert (tmp_path / "out" / "tables" / "member2_hypothesis_results.csv").exists()
    assert set(results["statistics"]) == {"hypotheses", "bayes"}
