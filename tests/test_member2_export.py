"""Test for src/analysis/export_member2.py using a small synthetic CSV."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.export_member2 import export_member2
from src.common.config import FEATURE_COLUMNS


def synthetic_csv(path: Path, n: int = 5000) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({column: rng.integers(0, 2, n) for column in FEATURE_COLUMNS})
    df["BMI"] = rng.integers(15, 60, n)
    df["GenHlth"] = rng.integers(1, 6, n)
    df["MentHlth"] = rng.integers(0, 31, n)
    df["PhysHlth"] = rng.integers(0, 31, n)
    df["Age"] = rng.integers(1, 14, n)
    df["Education"] = rng.integers(1, 7, n)
    df["Income"] = rng.integers(1, 9, n)
    df["Diabetes_binary"] = (rng.random(n) < 0.14).astype(int)
    df.to_csv(path, index=False)


class ExportMember2Tests(unittest.TestCase):
    def test_writes_both_tables_for_both_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            csv_path = Path(temp) / "sample.csv"
            synthetic_csv(csv_path)
            paths = export_member2(str(csv_path), str(Path(temp) / "out"))
            hypotheses = pd.read_csv(paths["hypotheses"])
            bayes = pd.read_csv(paths["bayes"])
        self.assertEqual(len(hypotheses), 14)
        self.assertEqual(len(bayes), 14)
        self.assertEqual(set(hypotheses["dataset_variant"]), {"full_clean_v1", "unique_profile_v1"})
        self.assertTrue(bayes["matches"].all())

    def test_missing_csv_raises(self):
        with self.assertRaises(Exception):
            export_member2("does_not_exist.csv")


if __name__ == "__main__":
    unittest.main()
