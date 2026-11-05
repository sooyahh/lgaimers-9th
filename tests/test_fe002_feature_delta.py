"""FE-002 feature-set delta: same names as EXP-004, prev1 values shrunk, no training."""

from __future__ import annotations

import inspect
import unittest

from scripts.fe002_prev1_shrinkage import PREV1_COL, SHRINKAGE_K
from scripts.run_fe002_prev1_shrinkage import (
    COLLAPSE_BSS_DELTA,
    EXPERIMENT_ID,
    PREV1_TRANSFORM,
    REPLACED_FEATURE,
    assert_configs_match_except_project,
    decide_verdict,
)
from scripts.tree_exp_common import build_feature_lists
from src.utils import load_config


class Fe002FeatureDeltaTests(unittest.TestCase):
    def test_config_matches_baseline_except_project_notes(self) -> None:
        baseline = load_config("configs/baseline.yaml")
        fe = load_config("configs/fe002.yaml")
        assert_configs_match_except_project(baseline, fe)
        self.assertEqual(baseline["runtime"]["seed"], 42)
        self.assertEqual(fe["runtime"]["seed"], 42)
        self.assertEqual(baseline["cv"], fe["cv"])
        self.assertEqual(baseline["task"], fe["task"])
        self.assertEqual(
            baseline["models"]["params"]["catboost"],
            fe["models"]["params"]["catboost"],
        )
        self.assertEqual(fe["models"]["params"]["catboost"]["iterations"], 300)
        self.assertEqual(fe["models"]["params"]["catboost"]["learning_rate"], 0.05)
        self.assertEqual(fe["models"]["params"]["catboost"]["depth"], 6)
        self.assertEqual(fe["models"]["params"]["catboost"]["loss_function"], "Logloss")
        self.assertEqual(baseline["features"]["exclude_cols"], fe["features"]["exclude_cols"])
        self.assertEqual(baseline["features"]["categorical_cols"], fe["features"]["categorical_cols"])
        self.assertNotIn(PREV1_COL, fe["features"]["exclude_cols"])
        self.assertNotIn("count_code", fe["features"]["categorical_cols"])
        self.assertIn("k=500", fe["project"]["notes"])
        self.assertIn("not raw prev1", fe["project"]["notes"])
        self.assertEqual(EXPERIMENT_ID, "FE-002")
        self.assertEqual(REPLACED_FEATURE, PREV1_COL)
        self.assertEqual(PREV1_TRANSFORM, "row_local_shrinkage_k500")
        self.assertEqual(SHRINKAGE_K, 500)
        self.assertEqual(COLLAPSE_BSS_DELTA, 50.0)

    def test_build_feature_lists_names_match_and_keep_prev3_prev5(self) -> None:
        columns = [
            "row_id",
            "control_success",
            "season",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "balls_before",
            "strikes_before",
            "pitcher_id",
        ]
        exclude = ["row_id", "home_win_expectancy", "away_win_expectancy", "li"]
        cats = ["pitcher_id"]
        used_b, nums_b, _ = build_feature_lists(
            columns,
            target_col="control_success",
            exclude_cols=exclude,
            categorical_cols=cats,
        )
        used_f, nums_f, cats_f = build_feature_lists(
            columns,
            target_col="control_success",
            exclude_cols=exclude,
            categorical_cols=cats,
        )
        self.assertEqual(set(used_b), set(used_f))
        self.assertEqual(set(nums_b), set(nums_f))
        self.assertIn(PREV1_COL, used_f)
        self.assertIn("asof_pitcher_prev3_game_success_rate", used_f)
        self.assertIn("asof_pitcher_prev5_game_success_rate", used_f)
        self.assertIn("balls_before", used_f)
        self.assertIn("strikes_before", used_f)
        self.assertNotIn("count_code", used_f)
        self.assertNotIn("tm_expected_rel_speed", used_f)
        self.assertNotIn("count_code", cats_f)

    def test_fe002_is_not_fe002a_drop(self) -> None:
        fe002 = load_config("configs/fe002.yaml")
        fe002a = load_config("configs/fe002a.yaml")
        self.assertNotIn(PREV1_COL, fe002["features"]["exclude_cols"])
        self.assertIn(PREV1_COL, fe002a["features"]["exclude_cols"])

    def test_tree_exp_common_has_no_fe002_formula(self) -> None:
        from scripts.tree_exp_common import run_tree_experiment

        source = inspect.getsource(run_tree_experiment)
        self.assertNotIn("shrinkage", source)
        self.assertNotIn("count_code", source)

    def test_adopt_when_slice_and_mean_improve_without_collapse(self) -> None:
        verdict, _ = decide_verdict(
            {
                "d2022": 5.0,
                "d2023": 10.0,
                "d2024": 8.0,
                "dmean": 7.666,
                "dstd": -1.0,
                "dworst": 5.0,
            },
            20.0,
        )
        self.assertEqual(verdict, "ADOPT_CANDIDATE")

    def test_reject_when_prev1_low_slice_does_not_improve(self) -> None:
        verdict, _ = decide_verdict(
            {
                "d2022": 10.0,
                "d2023": 10.0,
                "d2024": 10.0,
                "dmean": 10.0,
                "dstd": 0.0,
                "dworst": 10.0,
            },
            0.0,
        )
        self.assertEqual(verdict, "REJECT")

    def test_reject_when_mean_worsens_without_stability_gain(self) -> None:
        verdict, _ = decide_verdict(
            {
                "d2022": -20.0,
                "d2023": -5.0,
                "d2024": -5.0,
                "dmean": -10.0,
                "dstd": 5.0,
                "dworst": -20.0,
            },
            15.0,
        )
        self.assertEqual(verdict, "REJECT")

    def test_hold_when_slice_and_mean_improve_but_one_fold_collapses(self) -> None:
        verdict, _ = decide_verdict(
            {
                "d2022": -60.0,
                "d2023": 40.0,
                "d2024": 50.0,
                "dmean": 10.0,
                "dstd": 10.0,
                "dworst": -60.0,
            },
            25.0,
        )
        self.assertEqual(verdict, "HOLD")

    def test_hold_when_slice_improves_but_mean_drops_with_stability_gain(self) -> None:
        verdict, _ = decide_verdict(
            {
                "d2022": -20.0,
                "d2023": 5.0,
                "d2024": -8.0,
                "dmean": -7.666,
                "dstd": -2.0,
                "dworst": 4.0,
            },
            12.0,
        )
        self.assertEqual(verdict, "HOLD")

    def test_verdict_does_not_take_2023_low_n_slice(self) -> None:
        params = inspect.signature(decide_verdict).parameters
        self.assertEqual(list(params), ["deltas", "prev1_low_delta"])


if __name__ == "__main__":
    unittest.main()
