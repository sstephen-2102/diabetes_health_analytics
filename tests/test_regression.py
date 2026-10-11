import numpy as np
import pandas as pd
import pytest

from src.modeling.regression import (
    compare_regression_models,
    extract_odds_ratios,
    fit_logistic_regression,
)
from src.common.exceptions import DataValidationError


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data():
    """Small deterministic dataset with a learnable logistic relationship."""
    rng = np.random.default_rng(42)
    n = 200
    bmi = rng.normal(28, 5, n)
    age = rng.integers(1, 14, n).astype(float)
    high_bp = rng.integers(0, 2, n).astype(float)
    score = 0.08 * (bmi - 28) + 0.25 * (age - 7) + 0.8 * high_bp
    prob = 1 / (1 + np.exp(-score))
    target = rng.binomial(1, prob).astype(float)
    # Ensure both classes present
    target[0] = 0
    target[1] = 1
    df = pd.DataFrame({"BMI": bmi, "Age": age, "HighBP": high_bp,
                       "Diabetes_binary": target})
    df.attrs["dataset_variant"] = "full_clean_v1"
    return df


@pytest.fixture
def baseline_model(regression_data):
    return fit_logistic_regression(
        regression_data, "Diabetes_binary", ["BMI", "Age", "HighBP"]
    )


@pytest.fixture
def interaction_model(regression_data):
    return fit_logistic_regression(
        regression_data, "Diabetes_binary", ["BMI", "Age", "HighBP"],
        interaction_terms=[("BMI", "Age")]
    )


# ---------------------------------------------------------------------------
# fit_logistic_regression — happy path
# ---------------------------------------------------------------------------

def test_fit_returns_dict_with_required_keys(baseline_model):
    required = {"results", "n", "log_likelihood", "aic", "bic",
                "pseudo_r2", "feature_columns", "interaction_terms",
                "target_column", "dataset_variant"}
    assert required.issubset(baseline_model.keys())


def test_fit_results_has_required_columns(baseline_model):
    expected_cols = {"feature", "coefficient", "std_error", "p_value",
                     "odds_ratio", "ci_lower", "ci_upper"}
    assert expected_cols.issubset(baseline_model["results"].columns)


def test_fit_row_count_matches_data(regression_data, baseline_model):
    assert baseline_model["n"] == len(regression_data)


def test_fit_odds_ratios_are_positive(baseline_model):
    assert (baseline_model["results"]["odds_ratio"] > 0).all()


def test_fit_ci_lower_less_than_upper(baseline_model):
    df = baseline_model["results"]
    assert (df["ci_lower"] < df["ci_upper"]).all()


def test_fit_p_values_in_range(baseline_model):
    pv = baseline_model["results"]["p_value"]
    assert (pv >= 0).all() and (pv <= 1).all()


def test_fit_dataset_variant_preserved(baseline_model):
    assert baseline_model["dataset_variant"] == "full_clean_v1"


def test_fit_features_in_results(baseline_model):
    # Results should include a row for the constant and each feature
    features_in_results = baseline_model["results"]["feature"].tolist()
    assert "BMI" in features_in_results
    assert "Age" in features_in_results
    assert "HighBP" in features_in_results


# ---------------------------------------------------------------------------
# fit_logistic_regression — interaction terms
# ---------------------------------------------------------------------------

def test_fit_with_interaction_adds_column(interaction_model):
    features = interaction_model["results"]["feature"].tolist()
    assert any("BMI" in f and "Age" in f for f in features)


def test_fit_interaction_has_more_rows_than_baseline(baseline_model, interaction_model):
    assert len(interaction_model["results"]) > len(baseline_model["results"])


def test_fit_without_interaction_terms_is_empty_list(baseline_model):
    assert baseline_model["interaction_terms"] == []


# ---------------------------------------------------------------------------
# fit_logistic_regression — validation errors
# ---------------------------------------------------------------------------

def test_fit_rejects_non_dataframe():
    with pytest.raises(DataValidationError):
        fit_logistic_regression("not_a_df", "Diabetes_binary", ["BMI"])


def test_fit_rejects_missing_target_column(regression_data):
    with pytest.raises(DataValidationError):
        fit_logistic_regression(regression_data, "NonExistent", ["BMI"])


def test_fit_rejects_missing_feature_column(regression_data):
    with pytest.raises(DataValidationError):
        fit_logistic_regression(regression_data, "Diabetes_binary", ["BMI", "Missing"])


def test_fit_rejects_invalid_interaction_column(regression_data):
    with pytest.raises(DataValidationError):
        fit_logistic_regression(
            regression_data, "Diabetes_binary", ["BMI", "Age"],
            interaction_terms=[("BMI", "DoesNotExist")]
        )


def test_fit_rejects_malformed_interaction_terms(regression_data):
    with pytest.raises(DataValidationError):
        fit_logistic_regression(
            regression_data, "Diabetes_binary", ["BMI"],
            interaction_terms=[("BMI",)]   # tuple of length 1, not 2
        )


# ---------------------------------------------------------------------------
# extract_odds_ratios
# ---------------------------------------------------------------------------

def test_extract_returns_dataframe(baseline_model):
    result = extract_odds_ratios(baseline_model)
    assert isinstance(result, pd.DataFrame)


def test_extract_has_required_columns(baseline_model):
    result = extract_odds_ratios(baseline_model)
    assert {"feature", "odds_ratio", "ci_lower", "ci_upper", "p_value"}.issubset(result.columns)


def test_extract_sorted_descending(baseline_model):
    result = extract_odds_ratios(baseline_model)
    ors = result["odds_ratio"].tolist()
    assert ors == sorted(ors, reverse=True)


def test_extract_rejects_non_dict():
    with pytest.raises(DataValidationError):
        extract_odds_ratios("not_a_dict")


def test_extract_rejects_missing_results_key():
    with pytest.raises(DataValidationError):
        extract_odds_ratios({"no_results_key": True})


# ---------------------------------------------------------------------------
# compare_regression_models
# ---------------------------------------------------------------------------

def test_compare_returns_dict_with_required_keys(baseline_model, interaction_model):
    result = compare_regression_models(baseline_model, interaction_model)
    assert {"baseline", "interaction", "delta_aic", "delta_bic"}.issubset(result.keys())


def test_compare_delta_aic_is_numeric(baseline_model, interaction_model):
    result = compare_regression_models(baseline_model, interaction_model)
    assert isinstance(result["delta_aic"], float)


def test_compare_interaction_has_more_features(baseline_model, interaction_model):
    result = compare_regression_models(baseline_model, interaction_model)
    assert result["interaction"]["interaction_terms"] == [("BMI", "Age")]


def test_compare_rejects_non_dict_baseline(interaction_model):
    with pytest.raises(DataValidationError):
        compare_regression_models("not_a_dict", interaction_model)


def test_compare_rejects_mismatched_target(regression_data):
    m1 = fit_logistic_regression(regression_data, "Diabetes_binary", ["BMI"])
    # Create a second model dict with a different target name
    m2 = dict(m1)
    m2["target_column"] = "SomeOtherTarget"
    with pytest.raises(DataValidationError):
        compare_regression_models(m1, m2)


# ---------------------------------------------------------------------------
# convergence, covariance and likelihood-ratio test
# ---------------------------------------------------------------------------

def test_fit_reports_convergence(baseline_model, interaction_model):
    assert baseline_model["converged"] is True
    assert interaction_model["converged"] is True


def test_fit_covariance_is_labelled_and_symmetric(interaction_model):
    covariance = interaction_model["covariance"]
    features = interaction_model["results"]["feature"].tolist()

    assert list(covariance.index) == features
    assert list(covariance.columns) == features
    np.testing.assert_allclose(covariance.values, covariance.values.T)


def test_fit_covariance_diagonal_matches_standard_errors(interaction_model):
    variances = np.diag(interaction_model["covariance"].values)

    np.testing.assert_allclose(np.sqrt(variances), interaction_model["results"]["std_error"].values)


def test_compare_likelihood_ratio_statistic(baseline_model, interaction_model):
    result = compare_regression_models(baseline_model, interaction_model)

    expected = 2 * (interaction_model["log_likelihood"] - baseline_model["log_likelihood"])
    assert result["lr_statistic"] == pytest.approx(expected)
    assert result["lr_statistic"] >= 0
    assert result["lr_df"] == 1


def test_compare_likelihood_ratio_p_value_matches_chi_square(baseline_model, interaction_model):
    from scipy import stats

    result = compare_regression_models(baseline_model, interaction_model)

    assert result["lr_p_value"] == pytest.approx(stats.chi2.sf(result["lr_statistic"], 1))
    assert 0 <= result["lr_p_value"] <= 1


def test_compare_lr_test_is_skipped_without_interaction_terms(baseline_model):
    result = compare_regression_models(baseline_model, baseline_model)

    assert result["lr_df"] == 0
    assert result["lr_p_value"] is None
    assert result["lr_statistic"] == pytest.approx(0)
