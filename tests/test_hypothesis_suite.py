"""Tests for src/statistics/hypothesis_suite.py."""
import unittest

import numpy as np
import pandas as pd

from src.common.exceptions import InvalidParameterError, MissingColumnError
from src.statistics.hypothesis_suite import run_hypothesis_suite


def suite_frame(n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    target = (rng.random(n) < 0.14).astype(int)
    df = pd.DataFrame({
        "Diabetes_binary": target,
        "HighBP": np.where(target == 1, rng.random(n) < 0.7, rng.random(n) < 0.4).astype(int),
        "HighChol": rng.integers(0, 2, n), "PhysActivity": rng.integers(0, 2, n),
        "Smoker": rng.integers(0, 2, n), "Age": rng.integers(1, 14, n),
        "Income": rng.integers(1, 9, n), "BMI": rng.normal(28, 6, n) + 4 * target,
    })
    df.attrs["dataset_variant"] = "test_variant"
    return df


class HypothesisSuiteTests(unittest.TestCase):
    def test_returns_seven_labelled_rows(self):
        result = run_hypothesis_suite(suite_frame())
        self.assertEqual(list(result["hypothesis"]), ["H1", "H2", "H3", "H4", "H5", "H6", "H7"])
        self.assertTrue((result["dataset_variant"] == "test_variant").all())
        self.assertTrue(result.loc[result.feature == "HighBP", "p_value"].iloc[0] < 0.001)

    def test_odds_ratio_only_for_two_by_two_features(self):
        result = run_hypothesis_suite(suite_frame()).set_index("feature")
        self.assertIsNotNone(result.loc["HighBP", "odds_ratio"])
        self.assertTrue(pd.isna(result.loc["Age", "odds_ratio"]))
        self.assertTrue(pd.isna(result.loc["BMI", "odds_ratio"]))

    def test_missing_column_raises(self):
        with self.assertRaises(MissingColumnError):
            run_hypothesis_suite(suite_frame().drop(columns=["Smoker"]))

    def test_non_binary_target_raises(self):
        df = suite_frame()
        df.loc[0, "Diabetes_binary"] = 2
        with self.assertRaises(InvalidParameterError):
            run_hypothesis_suite(df)


if __name__ == "__main__":
    unittest.main()
