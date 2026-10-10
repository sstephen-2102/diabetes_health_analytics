"""Tests for src/statistics/bayes_suite.py."""
import unittest

import numpy as np
import pandas as pd

from src.common.exceptions import InvalidParameterError, MissingColumnError
from src.statistics.bayes_suite import run_bayes_suite


def bayes_frame(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    target = (rng.random(n) < 0.14).astype(int)
    df = pd.DataFrame({"Diabetes_binary": target,
                       "HighBP": np.where(target == 1, rng.random(n) < 0.7, rng.random(n) < 0.4).astype(int),
                       "Smoker": rng.integers(0, 2, n)})
    df.attrs["dataset_variant"] = "test_variant"
    return df


class BayesSuiteTests(unittest.TestCase):
    def test_both_sides_match_for_each_feature(self):
        result = run_bayes_suite(bayes_frame(), features=["HighBP", "Smoker"])
        self.assertEqual(len(result), 2)
        self.assertTrue(result["matches"].all())
        self.assertTrue((result["dataset_variant"] == "test_variant").all())

    def test_lift_is_direct_over_prior(self):
        row = run_bayes_suite(bayes_frame(), features=["HighBP"]).iloc[0]
        self.assertAlmostEqual(row["lift_over_prior"],
                               row["direct_p_category1_given_feature"] / row["prior_p_category1"])
        self.assertGreater(row["lift_over_prior"], 1)

    def test_missing_feature_raises(self):
        with self.assertRaises(MissingColumnError):
            run_bayes_suite(bayes_frame(), features=["Nope"])

    def test_absent_feature_value_raises(self):
        with self.assertRaises(InvalidParameterError):
            run_bayes_suite(bayes_frame(), features=["HighBP"], feature_value=7)


if __name__ == "__main__":
    unittest.main()
