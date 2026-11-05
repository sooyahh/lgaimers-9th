"""Unit / smoke tests for LB-PROBE-L42-2019-2023-001. No full 2024 training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_full_train_exp005 import ID_DROP
from scripts.run_l42_2019_2023_lb_probe import (
    EXPERIMENT_ID,
    SCRIPT_TEMPLATE,
    TRAIN_SEASONS,
    VALID_SEASONS,
    WEIGHTS,
    assert_l42_contract,
    assert_no_2024_train_leakage,
    build_submit_zip,
    feature_list_hash,
    l42_feature_lists,
    run_primary_holdout,
)
from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from src.utils import load_config


BASELINE_44 = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


class L42ContractTests(unittest.TestCase):
    def test_experiment_id_and_window(self) -> None:
        self.assertEqual(EXPERIMENT_ID, "LB-PROBE-L42-2019-2023-001")
        self.assertEqual(TRAIN_SEASONS, [2019, 2020, 2021, 2022, 2023])
        self.assertEqual(VALID_SEASONS, [2024])
        self.assertEqual(max(TRAIN_SEASONS), 2023)
        self.assertNotIn(2024, TRAIN_SEASONS)
        self.assertEqual(WEIGHTS, {int(k): float(v) for k, v in SEASON_WEIGHTS.items()})
        self.assertNotIn(2024, WEIGHTS)

    def test_l42_drops_ids_keeps_game_type(self) -> None:
        cfg = load_config("configs/baseline.yaml")
        columns = ["row_id", "control_success", *BASELINE_44]
        used, _, cats = l42_feature_lists(
            columns,
            target_col="control_success",
            exclude_cols=list(cfg["features"]["exclude_cols"]),
            categorical_cols=list(cfg["features"]["categorical_cols"]),
        )
        assert_l42_contract(used, cats)
        self.assertEqual(len(used), 42)
        self.assertEqual(set(BASELINE_44) - set(used), set(ID_DROP))
        self.assertIn("game_type", used)
        self.assertNotIn("pitcher_id", used)
        self.assertNotIn("batter_id", used)
        self.assertEqual(
            used,
            [c for c in BASELINE_44 if c not in ID_DROP],
        )
        self.assertEqual(len(feature_list_hash(used)), 64)

    def test_catboost_params_match_baseline(self) -> None:
        cfg = load_config("configs/baseline.yaml")
        self.assertEqual(cfg["runtime"]["seed"], 42)
        self.assertEqual(
            cfg["models"]["params"]["catboost"],
            {
                "iterations": 300,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "Logloss",
            },
        )
        self.assertEqual(cfg["data"]["target_col"], "control_success")
        self.assertEqual(cfg["cv"]["primary"]["train_seasons"], TRAIN_SEASONS)
        self.assertEqual(cfg["cv"]["primary"]["valid_seasons"], VALID_SEASONS)

    def test_2024_train_leakage_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            assert_no_2024_train_leakage([2019, 2020, 2021, 2022, 2023, 2024], np.array([2019]))
        with self.assertRaises(AssertionError):
            assert_no_2024_train_leakage(TRAIN_SEASONS, np.array([2019, 2024]))
        assert_no_2024_train_leakage(TRAIN_SEASONS, np.array([2019, 2020, 2021, 2022, 2023]))

    def test_script_is_row_local(self) -> None:
        body = SCRIPT_TEMPLATE.split('if __name__ == "__main__":', 1)[0]
        code = body.split('"""', 2)[-1]
        lowered = code.lower()
        for banned in ("groupby(", ".rolling(", "target_encode", "value_counts("):
            self.assertNotIn(banned, lowered)
        self.assertIn("numeric_medians_primary_train_2019_2023", SCRIPT_TEMPLATE)
        self.assertNotIn("n_test = ", code)
        self.assertNotIn("assert len(test)", code)


class TinyHoldoutSmokeTests(unittest.TestCase):
    def test_tiny_synthetic_holdout_and_zip(self) -> None:
        rng = np.random.default_rng(42)
        n = 120
        seasons = np.array([2019, 2020, 2021, 2022, 2023, 2024] * (n // 6), dtype=int)
        frame = pd.DataFrame({"season": seasons, "row_id": [f"r{i}" for i in range(n)]})
        for col in BASELINE_44:
            if col == "season":
                continue
            if col in {
                "top_bottom",
                "game_type",
                "base_state",
                "pitcher_id",
                "batter_id",
                "pitcher_hand",
                "batter_hand",
                "pitcher_team_id",
                "batter_team_id",
            }:
                frame[col] = rng.choice(["A", "B"], size=n)
            else:
                frame[col] = rng.normal(size=n)
        frame["control_success"] = (rng.random(n) > 0.4).astype(int)
        used, nums, cats = l42_feature_lists(
            list(frame.columns),
            target_col="control_success",
            exclude_cols=["row_id", "home_win_expectancy", "away_win_expectancy", "li"],
            categorical_cols=[
                "top_bottom",
                "game_type",
                "base_state",
                "pitcher_id",
                "batter_id",
                "pitcher_hand",
                "batter_hand",
                "pitcher_team_id",
                "batter_team_id",
            ],
        )
        holdout = run_primary_holdout(
            train_df=frame,
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            target_col="control_success",
            season_col="season",
            id_col="row_id",
            params={"iterations": 10, "depth": 2, "loss_function": "Logloss", "verbose": False},
            seed=42,
            scale=100000.0,
            weight_map=WEIGHTS,
        )
        self.assertEqual(holdout["n_features"] if "n_features" in holdout else len(holdout["used"]), 42)
        self.assertEqual(holdout["train_season_max"], 2023)
        self.assertEqual(holdout["n_train"] + holdout["n_valid"], n)
        self.assertTrue(np.isfinite(holdout["pred_frame"]["prediction"]).all())
        self.assertGreaterEqual(holdout["pred_frame"]["prediction"].min(), 0.0)
        self.assertLessEqual(holdout["pred_frame"]["prediction"].max(), 1.0)
        self.assertEqual(holdout["pred_frame"]["season"].unique().tolist(), [2024])

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model_dir = root / "model"
            model_dir.mkdir()
            import joblib

            joblib.dump(holdout["model"], model_dir / "catboost_primary.joblib")
            (model_dir / "feature_meta.json").write_text(
                json.dumps(
                    {
                        "features_used": used,
                        "numeric_cols": nums,
                        "categorical_cols": cats,
                        "numeric_medians_primary_train_2019_2023": holdout["medians"],
                    }
                ),
                encoding="utf-8",
            )
            req = root / "requirements.txt"
            req.write_text("pandas\n", encoding="utf-8")
            names = build_submit_zip(
                model_dir=model_dir,
                requirements_src=req,
                zip_path=root / "submit.zip",
                bundle_dir=root / "bundle",
            )
            self.assertEqual(
                set(names),
                {
                    "script.py",
                    "requirements.txt",
                    "model/catboost_primary.joblib",
                    "model/feature_meta.json",
                },
            )


if __name__ == "__main__":
    unittest.main()
