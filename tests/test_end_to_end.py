from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from scripts.generate_synthetic_data import build_frames
from src.predict import predict
from src.train import train
from src.models import build_model as real_build_model


class EndToEndTest(unittest.TestCase):
    def test_synthetic_train_diagnostics_and_submission(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            train_df, test_df, sample_df = build_frames()
            train_df = train_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
            sample_df = sample_df.iloc[::-1].reset_index(drop=True)
            train_path, test_path, sample_path = root / "train.csv", root / "test.csv", root / "sample.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            sample_df.to_csv(sample_path, index=False)
            config = {
                "project": {"purpose": "test", "data_version": "synthetic"},
                "data": {
                    "train_path": str(train_path), "test_path": str(test_path),
                    "sample_submission_path": str(sample_path), "target_col": "target",
                    "id_col": "row_id", "season_col": "season", "time_col": "event_time",
                    "submission_key_cols": ["row_id"],
                    "order_cols": ["season", "event_time", "row_id"],
                    "entity_id_col": "pitcher_id",
                },
                "task": {"problem_type": "binary_classification", "metric": "log_loss",
                         "metric_callable": None, "metric_direction": "minimize"},
                "cv": {"strategy": "expanding_window", "min_train_seasons": 2,
                       "validation_seasons": 1, "holdout_seasons": 1, "gap_seasons": 0},
                "models": {"enabled": ["global_mean", "logistic_regression", "catboost"],
                           "missing_dependency": "skip",
                           "params": {"global_mean": {}, "logistic_regression": {"solver": "liblinear"},
                                      "catboost": {}}},
                "features": {"parse_datetime": True, "add_missing_indicators": True},
                "diagnostics": {"calibration_bins": 5, "min_group_rows": 10},
                "submission": {"models": None},
                "runtime": {"seed": 42},
                "output": {"models_dir": str(root / "models"), "predictions_dir": str(root / "predictions"),
                           "submissions_dir": str(root / "submissions"), "experiments_path": str(root / "experiments.csv")},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            def controlled_build_model(name, params, seed):
                if name == "catboost":
                    raise ImportError("simulated missing catboost")
                return real_build_model(name, params, seed)

            with patch("src.train.build_model", side_effect=controlled_build_model):
                experiment_id = train(config_path)
            pred_dir = root / "predictions" / experiment_id
            for filename in ["oof_predictions.csv", "holdout_predictions.csv",
                             "all_validation_predictions.csv", "fold_metrics.csv", "season_metrics.csv",
                             "entity_metrics.csv", "entity_status_metrics.csv",
                             "probability_distribution.csv", "calibration_bins.csv"]:
                self.assertTrue((pred_dir / filename).exists(), filename)
            submission_path = predict(config_path, experiment_id)
            submission = pd.read_csv(submission_path)
            self.assertListEqual(submission["row_id"].tolist(), sample_df["row_id"].tolist())
            self.assertTrue(submission["success_probability"].between(0, 1).all())
            oof = pd.read_csv(pred_dir / "oof_predictions.csv")
            holdout = pd.read_csv(pred_dir / "holdout_predictions.csv")
            self.assertEqual(set(oof["split_type"]), {"expanding_window"})
            self.assertEqual(set(holdout["split_type"]), {"recent_holdout"})
            experiments = pd.read_csv(root / "experiments.csv")
            self.assertEqual(len(experiments), 3)
            self.assertEqual(
                experiments.set_index("model").loc["catboost", "status"], "skipped"
            )
            self.assertEqual(
                set(experiments.loc[experiments["status"] == "success", "model"]),
                {"global_mean", "logistic_regression"},
            )


if __name__ == "__main__":
    unittest.main()
