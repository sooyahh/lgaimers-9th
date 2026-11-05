"""TM-002A feature-set delta: add only tm_expected_rel_speed, no training."""

from __future__ import annotations

import unittest

from scripts.tree_exp_common import build_feature_lists
from src.trackman_profiles import TM_EXPECTED_REL_SPEED
from src.utils import load_config


class Tm002AFeatureDeltaTests(unittest.TestCase):
    def test_config_matches_baseline_except_project_notes(self) -> None:
        baseline = load_config("configs/baseline.yaml")
        tm = load_config("configs/tm002a.yaml")
        self.assertEqual(baseline["runtime"]["seed"], tm["runtime"]["seed"])
        self.assertEqual(baseline["cv"], tm["cv"])
        self.assertEqual(
            baseline["models"]["params"]["catboost"],
            tm["models"]["params"]["catboost"],
        )
        self.assertEqual(baseline["task"], tm["task"])
        self.assertEqual(baseline["features"]["exclude_cols"], tm["features"]["exclude_cols"])
        self.assertEqual(
            baseline["features"]["categorical_cols"],
            tm["features"]["categorical_cols"],
        )
        self.assertEqual(baseline["runtime"]["seed"], 42)
        self.assertEqual(tm["models"]["params"]["catboost"]["iterations"], 300)
        self.assertEqual(tm["models"]["params"]["catboost"]["learning_rate"], 0.05)
        self.assertEqual(tm["models"]["params"]["catboost"]["depth"], 6)
        self.assertEqual(tm["models"]["params"]["catboost"]["loss_function"], "Logloss")

    def test_build_feature_lists_delta_is_exactly_one_added_column(self) -> None:
        columns = [
            "row_id",
            "control_success",
            "season",
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
            "pitcher_id",
        ]
        exclude = ["row_id", "home_win_expectancy", "away_win_expectancy", "li"]
        used_b, _, _ = build_feature_lists(
            columns,
            target_col="control_success",
            exclude_cols=exclude,
            categorical_cols=["pitcher_id"],
        )
        used_tm, nums, _ = build_feature_lists(
            columns + [TM_EXPECTED_REL_SPEED],
            target_col="control_success",
            exclude_cols=exclude,
            categorical_cols=["pitcher_id"],
        )
        self.assertEqual(set(used_tm) - set(used_b), {TM_EXPECTED_REL_SPEED})
        self.assertEqual(set(used_b) - set(used_tm), set())
        self.assertEqual(len(used_tm) - len(used_b), 1)
        self.assertIn(TM_EXPECTED_REL_SPEED, nums)

    def test_verdict_rejects_mean_and_worst_fold_drop(self) -> None:
        from scripts.run_tm002a_expected_velocity import decide_verdict

        verdict, _ = decide_verdict(
            {
                "d2022": -10.0,
                "d2023": -20.0,
                "d2024": -5.0,
                "dmean": -11.7,
                "dstd": 1.0,
                "dworst": -20.0,
            },
            leakage_ok=True,
            near_clone=False,
        )
        self.assertEqual(verdict, "REJECT")

    def test_verdict_rejects_leakage(self) -> None:
        from scripts.run_tm002a_expected_velocity import decide_verdict

        verdict, _ = decide_verdict(
            {
                "d2022": 10.0,
                "d2023": 10.0,
                "d2024": 10.0,
                "dmean": 10.0,
                "dstd": 0.0,
                "dworst": 10.0,
            },
            leakage_ok=False,
            near_clone=False,
        )
        self.assertEqual(verdict, "REJECT")


if __name__ == "__main__":
    unittest.main()
