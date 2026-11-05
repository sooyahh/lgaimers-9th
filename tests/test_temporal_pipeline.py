from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data import sort_temporal_frame
from src.evaluate import build_temporal_folds, safe_group_scores, score_metric
from src.submission import align_predictions_to_sample, validate_submission
from src.train import _validate_chronology


class TemporalSplitTests(unittest.TestCase):
    def test_expanding_and_holdout_have_no_future_rows(self) -> None:
        frame = pd.DataFrame({"season": np.repeat(np.arange(2019, 2025), 3)})
        folds = build_temporal_folds(frame, "season", 2, 1, 1, 0)
        self.assertEqual(folds[-1].name, "recent_holdout")
        self.assertEqual(folds[-1].valid_seasons, (2024,))
        for fold in folds:
            self.assertLess(max(fold.train_seasons), min(fold.valid_seasons))
            self.assertTrue(set(fold.train_idx).isdisjoint(set(fold.valid_idx)))

    def test_insufficient_seasons_fails_closed(self) -> None:
        frame = pd.DataFrame({"season": [2023, 2024]})
        with self.assertRaises(ValueError):
            build_temporal_folds(frame, "season", 2, 1, 1, 0)

    def test_overlapping_event_times_are_rejected(self) -> None:
        frame = pd.DataFrame({"season": [2019, 2019, 2020, 2020, 2021, 2021],
                              "event_time": ["2019-01-01", "2022-01-01", "2020-01-01",
                                             "2020-02-01", "2021-01-01", "2021-02-01"]})
        folds = build_temporal_folds(frame, "season", 1, 1, 1, 0)
        with self.assertRaisesRegex(ValueError, "Time leakage"):
            _validate_chronology(frame, folds, "event_time")

    def test_composite_order_keys_stably_sort_pitches(self) -> None:
        frame = pd.DataFrame({
            "season": [2020, 2019, 2019], "game": [1, 2, 1], "pitch": [1, 2, 3]
        })
        ordered, source_positions = sort_temporal_frame(
            frame, ["season", "game", "pitch"], "season"
        )
        self.assertListEqual(ordered["game"].tolist(), [1, 2, 1])
        self.assertListEqual(source_positions.tolist(), [2, 1, 0])

    def test_non_unique_order_keys_are_rejected(self) -> None:
        frame = pd.DataFrame({"season": [2019, 2019], "game": [1, 1]})
        with self.assertRaisesRegex(ValueError, "cannot recover the actual time order"):
            sort_temporal_frame(frame, ["season", "game"], "season")

    def test_missing_value_in_any_order_key_is_rejected(self) -> None:
        frame = pd.DataFrame({"season": [2019, 2019], "game": [1, np.nan], "pitch": [1, 2]})
        with self.assertRaisesRegex(ValueError, "order_cols contain missing values"):
            sort_temporal_frame(frame, ["season", "game", "pitch"], "season")


class SubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = pd.DataFrame({"id": [3, 1, 2], "prob": [0.0, 0.0, 0.0]})

    def test_reordered_ids_are_rejected(self) -> None:
        bad = pd.DataFrame({"id": [1, 2, 3], "prob": [0.2, 0.3, 0.4]})
        with self.assertRaisesRegex(ValueError, "key order"):
            validate_submission(self.sample, bad, "id")

    def test_out_of_range_probability_is_rejected(self) -> None:
        bad = self.sample.copy()
        bad["prob"] = [0.2, 1.1, 0.4]
        with self.assertRaisesRegex(ValueError, "\[0, 1\]"):
            validate_submission(self.sample, bad, "id")

    def test_test_predictions_are_aligned_to_sample_ids(self) -> None:
        predictions = pd.DataFrame({"id": [1, 2, 3], "prob": [0.1, 0.2, 0.3]})
        aligned = align_predictions_to_sample(predictions, self.sample, "id")
        self.assertListEqual(aligned["id"].tolist(), [3, 1, 2])
        self.assertListEqual(aligned["prob"].tolist(), [0.3, 0.1, 0.2])

    def test_duplicate_submission_keys_are_rejected(self) -> None:
        predictions = pd.DataFrame({"id": [1, 1, 2], "prob": [0.1, 0.2, 0.3]})
        with self.assertRaisesRegex(ValueError, "duplicate submission keys"):
            align_predictions_to_sample(predictions, self.sample, ["id"])

    def test_missing_submission_key_value_is_rejected(self) -> None:
        predictions = pd.DataFrame({"id": [1, np.nan, 3], "prob": [0.1, 0.2, 0.3]})
        with self.assertRaisesRegex(ValueError, "missing submission key values"):
            align_predictions_to_sample(predictions, self.sample, ["id"])

    def test_missing_prediction_key_is_rejected(self) -> None:
        predictions = pd.DataFrame({"id": [1, 2], "prob": [0.1, 0.2]})
        with self.assertRaisesRegex(ValueError, "missing_prediction_keys=1"):
            align_predictions_to_sample(predictions, self.sample, ["id"])

    def test_unexpected_prediction_key_is_rejected(self) -> None:
        predictions = pd.DataFrame({"id": [1, 2, 3, 4], "prob": [0.1, 0.2, 0.3, 0.4]})
        with self.assertRaisesRegex(ValueError, "unexpected_prediction_keys=1"):
            align_predictions_to_sample(predictions, self.sample, ["id"])

    def test_composite_submission_keys_align_one_to_one(self) -> None:
        sample = pd.DataFrame({"game": [2, 1], "pitch": [1, 2], "prob": [0.0, 0.0]})
        predictions = pd.DataFrame({"game": [1, 2], "pitch": [2, 1], "prob": [0.2, 0.8]})
        aligned = align_predictions_to_sample(predictions, sample, ["game", "pitch"])
        self.assertListEqual(aligned["prob"].tolist(), [0.8, 0.2])


class MetricAndGroupTests(unittest.TestCase):
    def test_custom_metric_module_function_hook(self) -> None:
        score = score_metric(
            [0, 1], [0.2, 0.8], "official_custom", "binary_classification",
            "sklearn.metrics:brier_score_loss",
        )
        self.assertAlmostEqual(score, 0.04)

    def test_small_groups_are_reported_but_not_scored(self) -> None:
        frame = pd.DataFrame({"pitcher": ["P1", "P1"], "target": [0, 1], "prediction": [0.2, 0.8]})
        result = safe_group_scores(frame, ["pitcher"], "target", "prediction", "log_loss", min_rows=3)
        self.assertFalse(bool(result.loc[0, "eligible_for_score"]))
        self.assertTrue(pd.isna(result.loc[0, "score"]))
        self.assertEqual(result.loc[0, "score_reason"], "below_min_rows:3")


if __name__ == "__main__":
    unittest.main()
