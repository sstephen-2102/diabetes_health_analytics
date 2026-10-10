"""Tests for confidence intervals, hypothesis tests, and effect sizes (Member 2)."""
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from src.common.exceptions import InvalidParameterError, MissingColumnError
from src.statistics.association import calculate_effect_size
from src.statistics.confidence_intervals import (
    difference_in_means_ci, mean_confidence_interval, proportion_confidence_interval,
)
from src.statistics.hypothesis_testing import run_chi_square_test, run_welch_t_test


def highbp_frame() -> pd.DataFrame:
    """Counts copied from Member 1's full_clean_v1 HighBP prevalence table."""
    rows = ([(0, 1)] * 8742 + [(0, 0)] * (144851 - 8742)
            + [(1, 1)] * 26604 + [(1, 0)] * (108829 - 26604))
    df = pd.DataFrame(rows, columns=["HighBP", "Diabetes_binary"])
    df.attrs["dataset_variant"] = "test_variant"
    return df


class ProportionCITests(unittest.TestCase):
    def test_normal_case_brackets_estimate(self):
        result = proportion_confidence_interval(26604, 108829)
        self.assertLess(result["lower"], result["estimate"])
        self.assertGreater(result["upper"], result["estimate"])
        self.assertEqual(result["method"], "wilson")

    def test_successes_above_total_raises(self):
        with self.assertRaises(InvalidParameterError):
            proportion_confidence_interval(6, 5)

    def test_zero_total_and_bad_level_raise(self):
        with self.assertRaises(InvalidParameterError):
            proportion_confidence_interval(0, 0)
        with self.assertRaises(InvalidParameterError):
            proportion_confidence_interval(1, 5, confidence_level=1.5)

    def test_zero_successes_stays_in_bounds(self):
        result = proportion_confidence_interval(0, 50)
        self.assertGreaterEqual(result["lower"], 0.0)
        self.assertGreater(result["upper"], 0.0)


class MeanCITests(unittest.TestCase):
    def test_matches_scipy(self):
        values = np.random.default_rng(3).normal(10, 2, 200)
        low, high = stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=stats.sem(values))
        result = mean_confidence_interval(values)
        self.assertAlmostEqual(result["lower"], low)
        self.assertAlmostEqual(result["upper"], high)

    def test_single_value_raises(self):
        with self.assertRaises(InvalidParameterError):
            mean_confidence_interval([1])

    def test_missing_values_are_dropped_and_counted(self):
        self.assertEqual(mean_confidence_interval([1, 2, np.nan, 4])["n_missing_dropped"], 1)


class DifferenceInMeansCITests(unittest.TestCase):
    def test_matches_scipy_welch(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(32, 7, 500), rng.normal(28, 6, 800)
        ref = stats.ttest_ind(a, b, equal_var=False)
        interval = ref.confidence_interval(0.95)
        result = difference_in_means_ci(a, b)
        self.assertAlmostEqual(result["lower"], interval.low)
        self.assertAlmostEqual(result["upper"], interval.high)
        self.assertAlmostEqual(result["degrees_of_freedom"], ref.df)

    def test_group_too_small_raises(self):
        with self.assertRaises(InvalidParameterError):
            difference_in_means_ci([1, 2], [3])

    def test_sign_follows_group_order(self):
        self.assertGreater(difference_in_means_ci([5, 6, 7], [1, 2, 3])["difference"], 0)


class ChiSquareTests(unittest.TestCase):
    def test_matches_manual_calculation(self):
        result = run_chi_square_test(highbp_frame(), "HighBP", "Diabetes_binary")
        observed = result["contingency_table"].to_numpy(float)
        expected = observed.sum(1, keepdims=True) * observed.sum(0, keepdims=True) / observed.sum()
        self.assertAlmostEqual(result["chi_square"], ((observed - expected) ** 2 / expected).sum(), places=6)
        self.assertEqual(result["degrees_of_freedom"], 1)
        self.assertEqual(result["n"], 253680)
        self.assertTrue(0 < result["cramers_v"] < 1)

    def test_missing_column_raises(self):
        with self.assertRaises(MissingColumnError):
            run_chi_square_test(highbp_frame(), "Nope", "Diabetes_binary")

    def test_constant_feature_raises(self):
        with self.assertRaises(InvalidParameterError):
            run_chi_square_test(pd.DataFrame({"a": [1, 1, 1], "b": [0, 1, 0]}), "a", "b")

    def test_small_expected_counts_warn(self):
        df = pd.DataFrame({"a": [0, 0, 1, 1, 0], "b": [0, 1, 0, 1, 0]})
        self.assertTrue(run_chi_square_test(df, "a", "b")["assumption_warnings"])


class WelchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.df = pd.DataFrame({"BMI": np.r_[rng.normal(32, 7, 3000), rng.normal(28, 6, 9000)],
                                "target": np.r_[np.ones(3000), np.zeros(9000)].astype(int)})

    def test_matches_scipy(self):
        result = run_welch_t_test(self.df, "BMI", "target", 1, 0)
        ref = stats.ttest_ind(self.df.BMI[self.df.target == 1], self.df.BMI[self.df.target == 0], equal_var=False)
        self.assertAlmostEqual(result["t_statistic"], ref.statistic)
        self.assertAlmostEqual(result["degrees_of_freedom"], ref.df)
        self.assertAlmostEqual(result["p_value"], ref.pvalue)
        self.assertGreater(result["cohens_d"], 0)

    def test_unknown_group_raises(self):
        with self.assertRaises(InvalidParameterError):
            run_welch_t_test(self.df, "BMI", "target", 1, 9)

    def test_skewed_group_warns(self):
        df = pd.DataFrame({"x": np.r_[np.zeros(900), np.full(100, 30.0), np.zeros(500)],
                           "g": np.r_[np.ones(1000), np.zeros(500)].astype(int)})
        self.assertTrue(run_welch_t_test(df, "x", "g", 1, 0)["assumption_warnings"])


class EffectSizeTests(unittest.TestCase):
    def test_odds_ratio_matches_scipy(self):
        from scipy.stats.contingency import odds_ratio
        df = highbp_frame()
        table = pd.crosstab(df.HighBP, df.Diabetes_binary).to_numpy()
        result = calculate_effect_size("odds_ratio", df, feature="HighBP", target_column="Diabetes_binary")
        self.assertAlmostEqual(result["value"], odds_ratio(table, kind="sample").statistic, places=6)
        self.assertLess(result["lower"], result["value"])
        self.assertLess(result["value"], result["upper"])

    def test_cohens_d_sign_and_zero_variance(self):
        df = pd.DataFrame({"x": [5, 6, 7, 1, 2, 3], "g": [1, 1, 1, 0, 0, 0]})
        self.assertGreater(calculate_effect_size("cohens_d", df, numeric_feature="x", target_column="g",
                                                 group_a=1, group_b=0)["value"], 0)
        flat = pd.DataFrame({"x": [2, 2, 2, 2], "g": [1, 1, 0, 0]})
        with self.assertRaises(InvalidParameterError):
            calculate_effect_size("cohens_d", flat, numeric_feature="x", target_column="g", group_a=1, group_b=0)

    def test_unknown_type_raises_and_zero_cell_warns(self):
        with self.assertRaises(InvalidParameterError):
            calculate_effect_size("bogus", highbp_frame())
        df = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 0]})
        self.assertIsNotNone(calculate_effect_size("odds_ratio", df, feature="a", target_column="b")["warning"])


if __name__ == "__main__":
    unittest.main()
