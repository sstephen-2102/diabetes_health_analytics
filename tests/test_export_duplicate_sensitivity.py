"""Tests for src/analysis/export_duplicate_sensitivity.py on a small generated dataset."""
import json

import pandas as pd
import pytest

from src.analysis.export_duplicate_sensitivity import SCENARIOS, export_duplicate_sensitivity
from src.analysis.export_modeling import export_modeling
from src.common.exceptions import DataValidationError
from tests.test_export_modeling import MODELS, create_survey_csv

UNIQUE_ROWS, REPEATED_ROWS = 600, 200


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("duplicates")
    csv = create_survey_csv(folder / "survey.csv", n_rows=UNIQUE_ROWS)
    survey = pd.read_csv(csv)
    pd.concat([survey, survey.head(REPEATED_ROWS)]).to_csv(csv, index=False)
    output = folder / "outputs"
    manifest = export_duplicate_sensitivity(str(csv), str(output), check_reference=False)
    table = pd.read_csv(output / "tables" / "duplicate_sensitivity.csv")
    return {"csv": csv, "folder": folder, "output": output, "manifest": manifest, "table": table}


def all_rows(table):
    return table[table["test_subset"] == "all"].set_index(["scenario", "model"])


def test_every_scenario_and_model_has_a_full_test_row(exported):
    rows = all_rows(exported["table"])

    assert set(rows.index) == {(s, m) for s in SCENARIOS for m in MODELS}


def test_scenarios_use_the_right_variant_sizes(exported):
    scenarios = exported["manifest"]["scenarios"]
    total = {"full_clean_v1": UNIQUE_ROWS + REPEATED_ROWS, "unique_profile_v1": UNIQUE_ROWS}

    for name, (variant, split_type) in SCENARIOS.items():
        details = scenarios[name]
        assert (details["dataset_variant"], details["split"]) == (variant, split_type)
        assert details["train_rows"] + details["test_rows"] == total[variant]


def test_grouped_splits_never_share_feature_vectors(exported):
    scenarios = exported["manifest"]["scenarios"]

    assert scenarios["unique_grouped"]["seen_test_rows"] == 0
    assert scenarios["full_grouped"]["seen_test_rows"] == 0


def test_random_split_on_full_data_leaks_repeated_rows(exported):
    details = exported["manifest"]["scenarios"]["full_random"]

    assert details["seen_test_rows"] > 0
    assert details["seen_test_share"] == pytest.approx(details["seen_test_rows"] / details["test_rows"])


def test_seen_and_unseen_subsets_partition_the_test_set(exported):
    table = exported["table"].set_index(["scenario", "model", "test_subset"])

    for model in MODELS:
        parts = table.loc[("full_random", model, "seen_in_training"), "n"] + \
                table.loc[("full_random", model, "unseen_in_training"), "n"]
        assert parts == table.loc[("full_random", model, "all"), "n"]


def test_primary_scenario_matches_modeling_export(exported):
    output = exported["folder"] / "modeling_outputs"
    export_modeling(str(exported["csv"]), str(output), check_reference=False, importance_sample=200)
    primary = pd.read_csv(output / "tables" / "model_comparison.csv", index_col="model")
    rows = all_rows(exported["table"]).loc["unique_grouped"]

    for metric in ["roc_auc", "pr_auc", "brier_score", "recall", "precision"]:
        assert rows[metric].to_dict() == pytest.approx(primary[metric].to_dict())


def test_manifest_file_matches_return_value(exported):
    saved = json.loads((exported["output"] / "duplicate_sensitivity_manifest.json").read_text())

    assert saved == exported["manifest"]
    assert all((exported["output"] / f).is_file() for f in saved["files"])
    assert saved["primary_scenario"] == "unique_grouped"


def test_export_is_reproducible(exported, tmp_path):
    export_duplicate_sensitivity(str(exported["csv"]), str(tmp_path), check_reference=False)
    again = pd.read_csv(tmp_path / "tables" / "duplicate_sensitivity.csv")

    pd.testing.assert_frame_equal(again, exported["table"])


def test_reference_check_rejects_small_dataset_before_writing(tmp_path):
    csv = create_survey_csv(tmp_path / "survey.csv")

    with pytest.raises(DataValidationError, match="reference"):
        export_duplicate_sensitivity(str(csv), str(tmp_path / "outputs"))
    assert not (tmp_path / "outputs").exists()
