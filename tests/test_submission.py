from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.submission import save_and_recheck_submission, validate_submission


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = pd.DataFrame(
            {
                "row_id": ["TEST_000001", "TEST_000017", "TEST_000213"],
                "control_success": [0.5, 0.5, 0.5],
            }
        )
        self.keys = ["row_id"]

    def _valid_submission(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_id": ["TEST_000001", "TEST_000017", "TEST_000213"],
                "control_success": [0.1, 0.2, 0.9],
            }
        )

    def test_row_count_must_match_sample(self) -> None:
        bad = self._valid_submission().iloc[:2].copy()
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            validate_submission(self.sample, bad, self.keys)

    def test_column_names_and_order(self) -> None:
        bad = self._valid_submission()[["control_success", "row_id"]]
        with self.assertRaisesRegex(ValueError, "columns do not match"):
            validate_submission(self.sample, bad, self.keys)

    def test_row_id_set_and_order(self) -> None:
        reordered = self._valid_submission().iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "key order"):
            validate_submission(self.sample, reordered, self.keys)
        missing = self._valid_submission().copy()
        missing.loc[0, "row_id"] = "TEST_MISSING"
        with self.assertRaisesRegex(ValueError, "Submission key mismatch"):
            validate_submission(self.sample, missing, self.keys)

    def test_nan_and_inf_rejected(self) -> None:
        nan_sub = self._valid_submission()
        nan_sub.loc[1, "control_success"] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_submission(self.sample, nan_sub, self.keys)
        inf_sub = self._valid_submission()
        inf_sub.loc[1, "control_success"] = np.inf
        with self.assertRaisesRegex(ValueError, "inf"):
            validate_submission(self.sample, inf_sub, self.keys)

    def test_probability_range(self) -> None:
        bad = self._valid_submission()
        bad.loc[2, "control_success"] = 1.01
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_submission(self.sample, bad, self.keys)

    def test_index_not_saved_and_reread_validation(self) -> None:
        submission = self._valid_submission()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            saved = save_and_recheck_submission(
                self.sample, submission, self.keys, path, probability=True
            )
            reloaded = pd.read_csv(saved)
            self.assertListEqual(list(reloaded.columns), ["row_id", "control_success"])
            # No unnamed index column from to_csv(index=True)
            self.assertFalse(any(col.startswith("Unnamed") for col in reloaded.columns))
            self.assertEqual(len(reloaded), len(self.sample))
            self.assertListEqual(
                reloaded["row_id"].tolist(), self.sample["row_id"].tolist()
            )
            validate_submission(self.sample, reloaded, self.keys)

    def test_official_sample_shape_contract(self) -> None:
        sample_path = Path("data/raw/sample_submission.csv")
        if not sample_path.exists():
            self.skipTest("official sample_submission.csv not present under data/raw")
        sample = pd.read_csv(sample_path, encoding="utf-8-sig")
        self.assertListEqual(list(sample.columns), ["row_id", "control_success"])
        self.assertEqual(len(sample), 5)
        submission = sample.copy()
        submission["control_success"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        validate_submission(sample, submission, ["row_id"])


if __name__ == "__main__":
    unittest.main()
