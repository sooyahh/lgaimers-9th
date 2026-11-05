"""FE-001A feature-set delta: exactly one removed column, no training."""

from __future__ import annotations

import unittest

from scripts.tree_exp_common import build_feature_lists
from src.utils import load_config


REMOVED = "asof_pitcher_ball_rate"


class Fe001AFeatureDeltaTests(unittest.TestCase):
    def test_config_delta_is_exactly_ball_rate(self) -> None:
        baseline = load_config("configs/baseline.yaml")
        fe = load_config("configs/fe001a.yaml")
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

    def test_build_feature_lists_drops_only_ball_rate(self) -> None:
        columns = [
            "row_id",
            "control_success",
            "season",
            "asof_pitcher_ball_rate",
            "asof_pitcher_strike_rate",
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


if __name__ == "__main__":
    unittest.main()
