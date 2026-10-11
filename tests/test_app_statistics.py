"""Tests for the Member 2 service functions and pages (Probability Explorer, Statistical Testing Lab)."""
import pandas as pd
import pytest
from scipy import stats

from app import services
from src.common.config import FEATURE_COLUMNS, TARGET_DEFINITION, VARIABLE_METADATA
from src.common.exceptions import InvalidParameterError
from tests.test_app_data import app_data, page_text, run_page, survey_csv  # noqa: F401 (fixtures)

TARGET = "Diabetes_binary"
CATEGORICAL = [c for c in FEATURE_COLUMNS if VARIABLE_METADATA[c].get("category_labels")]


@pytest.fixture
def small():
    """10 rows: HighBP = 1 for 4 rows (3 in category 1), HighBP = 0 for 6 rows (1 in category 1)."""
    frame = pd.DataFrame({
        "HighBP": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "Age": [9, 9, 10, 10, 9, 9, 10, 10, 11, 11],
        "BMI": [30, 34, 31, 35, 22, 25, 24, 27, 23, 26],
        TARGET: [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    })
    frame.attrs["dataset_variant"] = "full_clean_v1"
    return frame


# ---------------------------------------------------------------- probability service

def test_probability_analysis_reports_marginal_joint_and_conditional(small):
    result = services.get_probability_analysis(small, TARGET, 1, {"HighBP": 1})
    results = result["results"]

    assert result["dataset_variant"] == "full_clean_v1"
    assert result["target_definition"] == TARGET_DEFINITION
    assert results["marginal"]["probability"] == pytest.approx(0.4)
    assert (results["marginal"]["numerator"], results["marginal"]["denominator"]) == (4, 10)
    assert results["joint_conditions"]["probability"] == pytest.approx(0.4)
    assert results["joint_with_target"]["probability"] == pytest.approx(0.3)
    conditional = results["conditional"]
    assert (conditional["numerator"], conditional["denominator"]) == (3, 4)
    assert conditional["probability"] == pytest.approx(
        results["joint_with_target"]["probability"] / results["joint_conditions"]["probability"])
    interval = conditional["confidence_interval"]
    assert interval["lower"] < 0.75 < interval["upper"]


def test_probability_analysis_checks_bayes_only_for_a_single_condition(small):
    one = services.get_probability_analysis(small, TARGET, 1, {"HighBP": 1})["results"]
    two = services.get_probability_analysis(small, TARGET, 1, {"HighBP": 1, "Age": 9})["results"]

    assert one["bayes"]["matches"] is True
    assert one["bayes"]["direct_probability"] == pytest.approx(0.75)
    assert "bayes" not in two
    assert (two["conditional"]["numerator"], two["conditional"]["denominator"]) == (2, 2)


def test_probability_analysis_without_conditions_returns_marginal_only(small):
    results = services.get_probability_analysis(small, TARGET, 1, {})["results"]

    assert set(results) == {"marginal"}


def test_probability_analysis_handles_zero_denominator(small):
    result = services.get_probability_analysis(small, TARGET, 1, {"HighBP": 1, "Age": 11})

    assert result["results"]["conditional"]["probability"] is None
    assert any("Zero denominator" in w for w in result["warnings"])


def test_probability_analysis_flags_small_denominators(small):
    result = services.get_probability_analysis(small, TARGET, 1, {"HighBP": 1})

    assert any("Only 4 respondents" in w for w in result["warnings"])


def test_probability_analysis_rejects_target_as_condition(small):
    with pytest.raises(InvalidParameterError):
        services.get_probability_analysis(small, TARGET, 1, {TARGET: 1})


# ---------------------------------------------------------------- statistical test service

def test_chi_square_result_matches_scipy_and_adds_odds_ratio(small):
    chi = services.get_statistical_test_result(small, "chi_square", "HighBP", TARGET, {})["results"]

    expected = stats.chi2_contingency(pd.crosstab(small["HighBP"], small[TARGET]), correction=False)
    assert chi["chi_square"] == pytest.approx(expected[0])
    assert chi["row_percentages"].sum(axis=1).tolist() == pytest.approx([100, 100])
    assert chi["row_percentages"].loc[1, 1] == pytest.approx(75)
    assert chi["odds_ratio"]["value"] == pytest.approx((3 * 5) / (1 * 1))


def test_chi_square_omits_odds_ratio_beyond_two_categories(small):
    chi = services.get_statistical_test_result(small, "chi_square", "Age", TARGET, {})["results"]

    assert "odds_ratio" not in chi
    assert chi["degrees_of_freedom"] == 2


def test_welch_defaults_to_category_1_minus_category_0(small):
    welch = services.get_statistical_test_result(small, "welch", "BMI", TARGET, {})["results"]

    ones, zeros = small.loc[small[TARGET] == 1, "BMI"], small.loc[small[TARGET] == 0, "BMI"]
    assert welch["mean_difference"] == pytest.approx(ones.mean() - zeros.mean())
    assert welch["p_value"] == pytest.approx(stats.ttest_ind(ones, zeros, equal_var=False).pvalue)


def test_welch_warnings_name_the_target_groups():
    skewed = pd.DataFrame({"BMI": [20] * 30 + [90] + [25] * 30 + [95], TARGET: [1] * 31 + [0] * 31})

    result = services.get_statistical_test_result(skewed, "welch", "BMI", TARGET, {})

    assert result["warnings"][0].startswith("BMI in category 1 is strongly skewed")
    assert not any("group_" in w for w in result["warnings"])


def test_unknown_test_type_is_rejected(small):
    with pytest.raises(InvalidParameterError, match="chi_square"):
        services.get_statistical_test_result(small, "anova", "BMI", TARGET, {})


# ---------------------------------------------------------------- confidence intervals and suites

def test_proportion_intervals_cover_each_category_and_overall(small):
    result = services.get_confidence_intervals(small, "HighBP", TARGET)["results"]
    table = result["table"]

    assert result["kind"] == "proportion"
    assert table["group"].tolist() == ["All respondents", "0: No", "1: Yes"]
    assert table["estimate"].tolist() == pytest.approx([40, 100 / 6, 75])
    assert ((table["lower"] <= table["estimate"]) & (table["estimate"] <= table["upper"])).all()


def test_mean_intervals_widen_with_confidence_level(small):
    narrow = services.get_confidence_intervals(small, "BMI", TARGET, 0.90)["results"]
    wide = services.get_confidence_intervals(small, "BMI", TARGET, 0.99)["results"]

    assert narrow["kind"] == "mean"
    assert narrow["table"]["group"].str[0].tolist() == ["1", "0"]
    widths = [(r["table"]["upper"] - r["table"]["lower"]).tolist() for r in (narrow, wide)]
    assert all(w99 > w90 for w90, w99 in zip(*widths))
    assert wide["difference"]["upper"] - wide["difference"]["lower"] > (
        narrow["difference"]["upper"] - narrow["difference"]["lower"])


def test_hypothesis_summary_states_multiplicity_decision(survey_csv):
    data = pd.read_csv(survey_csv)

    result = services.get_hypothesis_summary(data, TARGET)

    assert result["results"]["hypotheses"]["hypothesis"].tolist() == [f"H{i}" for i in range(1, 8)]
    assert "No multiple-comparison adjustment" in result["warnings"][0]
    assert "Bonferroni" in result["warnings"][0]


def test_bayes_summary_agrees_with_direct_counts(survey_csv):
    table = services.get_bayes_summary(pd.read_csv(survey_csv), TARGET)["results"]["bayes"]

    assert len(table) == 7
    assert table["matches"].all()


# ---------------------------------------------------------------- pages

def test_probability_explorer_shows_conditional_share_and_decomposition(app_data):
    at = run_page("probability_explorer")

    text = page_text(at)
    assert "respondents with High blood pressure = Yes" in text
    assert "95% CI" in text
    assert at.dataframe[0].value["Probability"].tolist() == [
        "P(category 1)", "P(conditions)", "P(conditions and category 1)", "P(category 1 | conditions)"]
    assert "not a model prediction" in text


def test_probability_explorer_explains_empty_and_impossible_conditions(app_data):
    at = run_page("probability_explorer")
    at.multiselect[0].set_value([]).run()
    assert "Choose one or more conditions" in page_text(at)

    at.multiselect[0].set_value(["HighBP", "Age", "Income", "Education"]).run()
    for feature, code in {"Age": 1, "Income": 1, "Education": 1}.items():
        next(s for s in at.selectbox if s.key == f"condition_{feature}").set_value(code).run()

    assert not at.exception
    text = page_text(at)
    assert "No respondents have" in text or "Respondents meeting conditions" in text


def test_probability_explorer_bayes_tab_handles_every_feature(app_data):
    at = run_page("probability_explorer")

    for feature in CATEGORICAL:
        next(s for s in at.selectbox if s.key == "bayes_feature").set_value(feature).run()
        assert not at.exception, (feature, [e.message for e in at.exception])
        assert "Bayes' theorem and the direct count agree" in page_text(at)


def test_statistical_testing_lists_h1_to_h7(app_data):
    at = run_page("statistical_testing")

    assert at.dataframe[0].value["Hypothesis"].tolist() == [f"H{i}" for i in range(1, 8)]
    assert "Bonferroni" in page_text(at)
    assert "ANOVA and post-hoc comparisons are not included" in page_text(at)


@pytest.mark.parametrize("test", ["chi_square", "welch"])
def test_statistical_testing_runs_every_feature(app_data, test):
    at = run_page("statistical_testing")
    at.radio[1].set_value(test).run()

    options = next(s for s in at.selectbox if s.key == f"test_feature_{test}").options
    for feature in options:
        next(s for s in at.selectbox if s.key == f"test_feature_{test}").set_value(feature).run()
        assert not at.exception, (test, feature, [e.message for e in at.exception])
    assert {"p"} <= {m.label for m in at.metric}


@pytest.mark.parametrize("variant", ["full_clean_v1", "unique_profile_v1"])
def test_statistical_testing_intervals_for_every_variable(app_data, variant):
    at = run_page("statistical_testing")
    at.radio[0].set_value(variant).run()

    for feature in FEATURE_COLUMNS:
        next(s for s in at.selectbox if s.key == "interval_feature").set_value(feature).run()
        assert not at.exception, (variant, feature, [e.message for e in at.exception])
