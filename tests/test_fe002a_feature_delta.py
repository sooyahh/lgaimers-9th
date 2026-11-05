"""FE-002A feature-set delta: drop only prev1 success rate, no training."""

from __future__ import annotations

import unittest

from scripts.tree_exp_common import build_feature_lists
from src.utils import load_config


REMOVED = "asof_pitcher_prev1_game_success_rate"


class Fe002AFeatureDeltaTests(unittest.TestCase):
    def test_config_delta_is_exactly_prev1_success(self) -> None:
        baseline = load_config("configs/baseline.yaml")
        fe = load_config("configs/fe002a.yaml")
        self.assertEqual(baseline["runtime"]["seed"], fe["runtime"]["seed"])
        self.assertEqual(baseline["cv"], fe["cv"])
        self.assertEqual(
            baseline["models"]["params"]["catboost"],
            fe["models"]["params"]["catboost"],
        )
        self.assertEqual(baseline["task"], fe["task"])
        b_ex = set(baseline["features"]["exclude_cols"])
        f_ex = set(fe["features"]["exclude_cols"])
        self.assertEqual(f_ex - b_ex, {REMOVED})
        self.assertEqual(b_ex - f_ex, set())
        self.assertEqual(
            baseline["features"]["categorical_cols"],
            fe["features"]["categorical_cols"],
        )
        self.assertNotIn("asof_pitcher_ball_rate", f_ex)

    def test_build_feature_lists_drops_only_prev1(self) -> None:
        columns = [
            "row_id",
            "control_success",
            "season",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "pitcher_id",
        ]
        used_b, _, _ = build_feature_lists(
            columns,
            target_col="control_success",
            exclude_cols=["row_id", "home_win_expectancy", "away_win_expectancy", "li"],
            categorical_cols=["pitcher_id"],
        )
        used_f, _, _ = build_feature_lists(
            columns,
            target_col="control_success",
            exclude_cols=[
                "row_id",
                "home_win_expectancy",
                "away_win_expectancy",
                "li",
                REMOVED,
            ],
            categorical_cols=["pitcher_id"],
        )
        self.assertEqual(set(used_b) - set(used_f), {REMOVED})
        self.assertEqual(set(used_f) - set(used_b), set())
        self.assertEqual(len(used_b) - len(used_f), 1)
        self.assertIn("asof_pitcher_prev3_game_success_rate", used_f)
        self.assertIn("asof_pitcher_prev5_game_success_rate", used_f)

    def test_case_b_verdict_when_prev1_low_worsens(self) -> None:
        from scripts.run_fe002a_prev1_ablation import decide_verdict

        verdict, _, conclusion = decide_verdict(
            {
                "d2022": -17.5,
                "d2023": 5.6,
                "d2024": -9.2,
                "dmean": -7.0,
                "dstd": -9.5,
                "dworst": 5.6,
            },
            -91.8,
        )
        self.assertEqual(verdict, "REJECT")
        self.assertEqual(conclusion, "B")
