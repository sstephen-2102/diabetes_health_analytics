"""Tests for src/statistics/probability.py (normal, invalid-input, and edge cases)."""
import unittest

import pandas as pd

from src.common.exceptions import DataValidationError, InvalidParameterError, MissingColumnError
from src.statistics.probability import (
    calculate_conditional_probability, calculate_joint_probability, calculate_probability,
    calculate_risk_profile_probability, verify_bayes_theorem,
)


def small_frame() -> pd.DataFrame:
    df = pd.DataFrame({"HighBP": [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                       "Diabetes_binary": [1, 0, 1, 0, 0, 0, 0, 0, 1, 1]})
    df.attrs["dataset_variant"] = "test_variant"
    return df


class MarginalProbabilityTests(unittest.TestCase):
    def test_normal_case(self):
        self.assertAlmostEqual(calculate_probability(small_frame(), "HighBP", 1), 0.4)

    def test_missing_column_raises(self):
        with self.assertRaises(MissingColumnError):
            calculate_probability(small_frame(), "Nope", 1)

    def test_empty_data_raises(self):
        with self.assertRaises(DataValidationError):
            calculate_probability(small_frame().iloc[0:0], "HighBP", 1)

    def test_value_absent_gives_zero(self):
        self.assertEqual(calculate_probability(small_frame(), "HighBP", 7), 0.0)


class ConditionalProbabilityTests(unittest.TestCase):
    def test_normal_case_counts_and_probability(self):
        result = calculate_conditional_probability(small_frame(), "Diabetes_binary", 1, ["HighBP"], [1])
        self.assertEqual((result["numerator"], result["denominator"]), (3, 4))
        self.assertAlmostEqual(result["probability"], 0.75)
        self.assertEqual(result["dataset_variant"], "test_variant")
        self.assertIn("lower", result["confidence_interval"])

    def test_mismatched_condition_lists_raise(self):
        with self.assertRaises(InvalidParameterError):
            calculate_conditional_probability(small_frame(), "Diabetes_binary", 1, ["HighBP"], [1, 0])

    def test_zero_denominator_is_flagged_not_crashed(self):
        result = calculate_conditional_probability(small_frame(), "Diabetes_binary", 1, ["HighBP"], [7])
        self.assertIsNone(result["probability"])
        self.assertIn("Zero denominator", result["warning"])

    def test_two_conditions(self):
        df = small_frame().assign(Sex=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        result = calculate_conditional_probability(df, "Diabetes_binary", 1, ["HighBP", "Sex"], [1, 0])
        self.assertEqual(result["denominator"], 3)


class JointAndProfileTests(unittest.TestCase):
    def test_joint_probability(self):
        result = calculate_joint_probability(small_frame(), {"HighBP": 1, "Diabetes_binary": 1})
        self.assertEqual(result["numerator"], 3)
        self.assertAlmostEqual(result["probability"], 0.3)

    def test_joint_empty_conditions_raise(self):
        with self.assertRaises(InvalidParameterError):
            calculate_joint_probability(small_frame(), {})

    def test_profile_is_labelled_not_a_prediction(self):
        result = calculate_risk_profile_probability(small_frame(), "Diabetes_binary", 1, {"HighBP": 0})
        self.assertIn("not a model prediction", result["note"])
        self.assertAlmostEqual(result["probability"], 1 / 6)


class BayesVerificationTests(unittest.TestCase):
    def test_both_sides_match(self):
        result = verify_bayes_theorem(small_frame(), "Diabetes_binary", 1, "HighBP", 1)
        self.assertTrue(result["matches"])
        self.assertAlmostEqual(result["direct_probability"], result["bayes_rhs"])

    def test_missing_column_raises(self):
        with self.assertRaises(MissingColumnError):
            verify_bayes_theorem(small_frame(), "Diabetes_binary", 1, "Nope", 1)

    def test_undefined_when_condition_absent(self):
        result = verify_bayes_theorem(small_frame(), "Diabetes_binary", 1, "HighBP", 7)
        self.assertIsNone(result["matches"])


if __name__ == "__main__":
    unittest.main()
