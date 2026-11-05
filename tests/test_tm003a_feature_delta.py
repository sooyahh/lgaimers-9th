"""TM-003A feature-set delta: add only tm_expected_ivb_by_count, no training."""

from __future__ import annotations

import unittest

from scripts.tree_exp_common import build_feature_lists
from src.trackman_profiles import MIN_CELL_N, TM_EXPECTED_IVB_BY_COUNT, TM_EXPECTED_IVB_GLOBAL
from src.utils import load_config


class Tm003AFeatureDeltaTests(unittest.TestCase):
    def test_config_matches_baseline_except_project_notes(self) -> None:
        baseline = load_config("configs/baseline.yaml")
        tm = load_config("configs/tm003a.yaml")
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
        self.assertEqual(MIN_CELL_N, 1000)
        from scripts.run_tm003a_expected_ivb import assert_frozen_tm003a_metadata

        assert_frozen_tm003a_metadata(tm)
        self.assertEqual(tm["tm003a"]["model_feature"], TM_EXPECTED_IVB_BY_COUNT)
        self.assertFalse(tm["tm003a"]["min_n_is_tunable"])
        self.assertTrue(tm["tm003a"]["canonical_requires_explicit_flag"])

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
            columns + [TM_EXPECTED_IVB_BY_COUNT],
            target_col="control_success",
            exclude_cols=exclude,
            categorical_cols=["pitcher_id"],
        )
        used_with_diag, _, _ = build_feature_lists(
            columns + [TM_EXPECTED_IVB_BY_COUNT, TM_EXPECTED_IVB_GLOBAL],
            target_col="control_success",
            exclude_cols=exclude,
            categorical_cols=["pitcher_id"],
        )
        self.assertEqual(set(used_tm) - set(used_b), {TM_EXPECTED_IVB_BY_COUNT})
        self.assertEqual(set(used_b) - set(used_tm), set())
        self.assertEqual(len(used_tm) - len(used_b), 1)
        self.assertIn(TM_EXPECTED_IVB_BY_COUNT, nums)
        self.assertIn(TM_EXPECTED_IVB_GLOBAL, used_with_diag)
        self.assertNotEqual(TM_EXPECTED_IVB_BY_COUNT, TM_EXPECTED_IVB_GLOBAL)

    def test_min_n_is_frozen(self) -> None:
        from src.trackman_profiles import build_season_ivb_profile
        import pandas as pd

        tm = pd.DataFrame(
            {
                "season": [2019],
                "pitch_type_group": ["fastball"],
                "balls_before": [0],
                "strikes_before": [0],
                "induced_vert_break": [1.0],
            }
        )
        with self.assertRaises(ValueError):
            build_season_ivb_profile(tm, 2020, min_n=500)


if __name__ == "__main__":
    unittest.main()
