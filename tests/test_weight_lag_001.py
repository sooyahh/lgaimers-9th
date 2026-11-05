"""Unit / contract tests for WEIGHT-LAG-001. No full CatBoost training."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_exp004_catboost_weighted import fit_catboost
from scripts.run_full_train_exp005 import ID_DROP
from scripts.run_l42_2019_2023_lb_probe import assert_l42_contract, l42_feature_lists
from scripts.run_recency_audit_001 import (
    EXPECTED_CATBOOST_PARAMS,
    EXPECTED_SEED,
    LOCKED_WINDOWS,
)
from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from scripts.run_weight_lag_001 import (
    CONTRAST_ATOL,
    CONTROL_M,
    EXPECTED_FIT_COUNT,
    FULL_HISTORY_WINDOWS,
    LAG_DEFINITION,
    MULTIPLIERS,
    VALID_SEASONS,
    WEIGHT_POLICY,
    assert_canonical_controls,
    assert_full_history_matches_recency,
    assert_rowcount_and_feature_lock,
    assert_weight_endpoints,
    compute_contrasts,
    fold_max_lag,
    iter_cell_specs,
    lag_weight,
    sample_weights_for_seasons,
    season_lag,
    season_weight_map,
)
from src.evaluate import build_season_holdout, evaluate_predictions as canonical_evaluate
from src.metrics import reference_brier_score as canonical_reference_brier
from src.utils import load_config
from tests.test_l42_2019_2023_lb_probe import BASELINE_44


class WindowContractTests(unittest.TestCase):
    def test_fifteen_cells_control_first(self) -> None:
        cells = iter_cell_specs()
        self.assertEqual(len(cells), EXPECTED_FIT_COUNT)
        self.assertEqual(len(cells), 5 * 3)
        self.assertTrue(all(c.is_control for c in cells[:3]))
        self.assertEqual([c.valid_season for c in cells[:3]], [2022, 2023, 2024])
        seen = {(c.valid_season, c.multiplier) for c in cells}
        expected = {(y, float(m)) for y in VALID_SEASONS for m in MULTIPLIERS}
        self.assertEqual(seen, expected)

    def test_full_history_windows_match_recency_b_expanding(self) -> None:
        assert_full_history_matches_recency()
        self.assertEqual(list(FULL_HISTORY_WINDOWS[2022]), [2019, 2020, 2021])
        self.assertEqual(list(FULL_HISTORY_WINDOWS[2023]), [2019, 2020, 2021, 2022])
        self.assertEqual(list(FULL_HISTORY_WINDOWS[2024]), [2019, 2020, 2021, 2022, 2023])
        for valid, seasons in FULL_HISTORY_WINDOWS.items():
            self.assertEqual(list(seasons), LOCKED_WINDOWS[valid]["B_expanding"])
            self.assertTrue(max(seasons) < valid)
            self.assertEqual(max(seasons), valid - 1)

    def test_valid_season_never_in_train(self) -> None:
        for cell in iter_cell_specs():
            self.assertNotIn(cell.valid_season, cell.train_seasons)
            self.assertTrue(max(cell.train_seasons) < cell.valid_season)

    def test_holdout_builder_rejects_valid_in_train(self) -> None:
        frame = pd.DataFrame({"season": [2019, 2020, 2021], "x": [1, 2, 3]})
        with self.assertRaises(ValueError):
            build_season_holdout(frame, "season", [2019, 2020, 2021], [2021], name="leak")

    def test_season_lag_rejects_future_or_same_season(self) -> None:
        with self.assertRaises(AssertionError):
            season_lag(2024, 2024)
        with self.assertRaises(AssertionError):
            season_lag(2025, 2024)
        self.assertEqual(season_lag(2023, 2024), 1)


class WeightEndpointTests(unittest.TestCase):
    def test_endpoints_for_every_canonical_fold(self) -> None:
        for m in MULTIPLIERS:
            for valid, seasons in FULL_HISTORY_WINDOWS.items():
                mapping = season_weight_map(seasons, valid, m)
                newest = max(seasons)
                oldest = min(seasons)
                self.assertAlmostEqual(mapping[newest], m, delta=CONTRAST_ATOL)
                self.assertAlmostEqual(mapping[oldest], 1.0, delta=CONTRAST_ATOL)
                assert_weight_endpoints(mapping, seasons, valid, m)

    def test_monotonic_decrease_toward_older_seasons(self) -> None:
        for m in (1.5, 2.0, 3.0, 4.0):
            for valid, seasons in FULL_HISTORY_WINDOWS.items():
                mapping = season_weight_map(seasons, valid, m)
                ordered = sorted(seasons, reverse=True)
                weights = [mapping[s] for s in ordered]
                for earlier, later in zip(weights, weights[1:]):
                    self.assertGreater(earlier, later)

    def test_formula_examples(self) -> None:
        self.assertEqual(lag_weight(1, 3, 4.0), 4.0)
        self.assertEqual(lag_weight(3, 3, 4.0), 1.0)
        self.assertAlmostEqual(lag_weight(2, 3, 4.0), 2.0, delta=CONTRAST_ATOL)
        self.assertEqual(lag_weight(1, 5, 4.0), 4.0)
        self.assertEqual(lag_weight(5, 5, 4.0), 1.0)
        self.assertAlmostEqual(lag_weight(3, 5, 4.0), 2.0, delta=CONTRAST_ATOL)

    def test_l_equals_one_returns_m(self) -> None:
        self.assertEqual(lag_weight(1, 1, 4.0), 4.0)
        self.assertEqual(lag_weight(1, 1, 1.0), 1.0)
        mapping = season_weight_map([2023], 2024, 3.0)
        self.assertEqual(mapping, {2023: 3.0})

    def test_max_lag_is_fold_local(self) -> None:
        self.assertEqual(fold_max_lag([2019, 2020, 2021], 2022), 3)
        self.assertEqual(fold_max_lag([2019, 2020, 2021, 2022, 2023], 2024), 5)


class IdentityAndRelativeLagTests(unittest.TestCase):
    def test_m1_is_one_for_every_lag(self) -> None:
        for lag, L in ((1, 1), (1, 3), (2, 3), (3, 3), (5, 5), (1, 5)):
            self.assertEqual(lag_weight(lag, L, 1.0), 1.0)
        for valid, seasons in FULL_HISTORY_WINDOWS.items():
            mapping = season_weight_map(seasons, valid, 1.0)
            self.assertTrue(all(v == 1.0 for v in mapping.values()))
            weights, _ = sample_weights_for_seasons(
                np.repeat(list(seasons), 4), valid, 1.0
            )
            self.assertTrue(np.all(weights == 1.0))

    def test_weights_are_validation_relative_not_calendar_fixed(self) -> None:
        w_2022 = season_weight_map([2019, 2020, 2021], 2022, 2.0)
        w_2024 = season_weight_map([2019, 2020, 2021, 2022, 2023], 2024, 2.0)
        self.assertEqual(w_2022[2021], 2.0)
        self.assertNotEqual(w_2022[2021], w_2024[2021])
        self.assertAlmostEqual(w_2024[2021], 2.0 ** 0.5, delta=CONTRAST_ATOL)
        self.assertNotEqual(w_2022, {2019: 1.0, 2020: 1.5, 2021: 2.0})

    def test_does_not_use_canonical_calendar_season_weights(self) -> None:
        mapping = season_weight_map([2019, 2020, 2021, 2022, 2023], 2024, 2.0)
        calendar = {int(k): float(v) for k, v in SEASON_WEIGHTS.items()}
        self.assertNotEqual(mapping, calendar)
        # Newest/oldest can coincide with calendar values by chance; the interior must not copy the table.
        for year in (2020, 2021, 2022):
            self.assertGreater(abs(mapping[year] - calendar[year]), 1e-3)


class NoTargetDependencyTests(unittest.TestCase):
    def test_weight_helpers_do_not_reference_targets_or_stats(self) -> None:
        sources = "\n".join(
            [
                inspect.getsource(lag_weight),
                inspect.getsource(season_weight_map),
                inspect.getsource(sample_weights_for_seasons),
                inspect.getsource(season_lag),
                inspect.getsource(fold_max_lag),
            ]
        )
        forbidden = (
            "control_success",
            "target_rate",
            "y_tr",
            "y_va",
            "test.csv",
            "SEASON_WEIGHTS",
        )
        for token in forbidden:
            self.assertNotIn(token, sources)
        self.assertNotIn("valid target", sources.lower())


class FeatureAndModelLockTests(unittest.TestCase):
    def _l42(self) -> tuple[list[str], list[str], list[str]]:
        config = load_config("configs/baseline.yaml")
        feat = config["features"]
        columns = list(BASELINE_44) + [
            "row_id",
            "control_success",
            "home_win_expectancy",
            "away_win_expectancy",
            "li",
        ]
        return l42_feature_lists(
            columns,
            target_col=config["data"]["target_col"],
            exclude_cols=list(feat["exclude_cols"]),
            categorical_cols=list(feat["categorical_cols"]),
        )

    def test_feature_list_matches_canonical_l42(self) -> None:
        used, nums, cats = self._l42()
        self.assertEqual(len(used), 42)
        assert_l42_contract(used, cats)
        for dropped in ID_DROP:
            self.assertNotIn(dropped, used)
            self.assertNotIn(dropped, cats)
        self.assertIn("game_type", used)
        self.assertIn("game_type", cats)

    def test_feature_list_identical_across_multipliers(self) -> None:
        used_control, nums_control, cats_control = self._l42()
        for _m in MULTIPLIERS:
            used, nums, cats = self._l42()
            self.assertEqual(used, used_control)
            self.assertEqual(nums, nums_control)
            self.assertEqual(cats, cats_control)

    def test_dropped_ids_match_canonical(self) -> None:
        self.assertEqual(list(ID_DROP), ["pitcher_id", "batter_id"])

    def test_seed_and_catboost_params_match_baseline(self) -> None:
        config = load_config("configs/baseline.yaml")
        seed = int(config["runtime"]["seed"])
        params = dict(config["models"]["params"]["catboost"])
        self.assertEqual(seed, EXPECTED_SEED)
        self.assertEqual(params, EXPECTED_CATBOOST_PARAMS)
        self.assertEqual(params["iterations"], 300)
        self.assertEqual(params["learning_rate"], 0.05)
        self.assertEqual(params["depth"], 6)
        self.assertEqual(params["loss_function"], "Logloss")
        assert_canonical_controls(params, seed)

    def test_fit_catboost_accepts_sample_weight(self) -> None:
        self.assertIn("sample_weight", inspect.signature(fit_catboost).parameters)

    def test_metric_is_canonical_function(self) -> None:
        import scripts.run_weight_lag_001 as runner

        self.assertIs(runner.evaluate_predictions, canonical_evaluate)
        self.assertIs(runner.reference_brier_score, canonical_reference_brier)
        self.assertIs(runner.fit_catboost, fit_catboost)


class RowCountLockTests(unittest.TestCase):
    def test_multiplier_does_not_change_holdout_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "season": [2019] * 5 + [2020] * 7 + [2021] * 9 + [2022] * 4,
                "x": list(range(25)),
            }
        )
        fold = build_season_holdout(frame, "season", [2019, 2020, 2021], [2022], name="lock")
        n_train = len(fold.train_idx)
        n_valid = len(fold.valid_idx)
        for m in MULTIPLIERS:
            weights, mapping = sample_weights_for_seasons(
                frame.iloc[fold.train_idx]["season"].to_numpy(), 2022, m
            )
            self.assertEqual(len(weights), n_train)
            self.assertEqual(n_valid, 4)
            self.assertEqual(set(mapping), {2019, 2020, 2021})

    def test_assert_rowcount_lock_detects_drift(self) -> None:
        rows = []
        for m in MULTIPLIERS:
            for valid in VALID_SEASONS:
                rows.append(
                    {
                        "valid_season": valid,
                        "multiplier_m": m,
                        "train_rows": 100,
                        "valid_rows": 20,
                        "feature_count": 42,
                        "feature_fingerprint": "abc",
                        "catboost_params": json.dumps(EXPECTED_CATBOOST_PARAMS, sort_keys=True),
                        "seed": 42,
                    }
                )
        assert_rowcount_and_feature_lock(rows)
        rows[0]["train_rows"] = 99
        with self.assertRaises(AssertionError):
            assert_rowcount_and_feature_lock(rows)


class ContrastTests(unittest.TestCase):
    @staticmethod
    def _rows(raw: dict[float, dict[int, float]]) -> list[dict]:
        rows = []
        for m in MULTIPLIERS:
            for valid in VALID_SEASONS:
                rows.append(
                    {
                        "valid_season": valid,
                        "multiplier_m": m,
                        "raw_bss": raw[m][valid],
                    }
                )
        return rows

    def test_delta_is_treatment_minus_control(self) -> None:
        raw = {
            1.0: {2022: 10.0, 2023: 20.0, 2024: 30.0},
            1.5: {2022: 11.0, 2023: 19.0, 2024: 33.0},
            2.0: {2022: 9.0, 2023: 21.0, 2024: 30.0},
            3.0: {2022: 10.0, 2023: 20.0, 2024: 31.0},
            4.0: {2022: 8.0, 2023: 18.0, 2024: 40.0},
        }
        out = compute_contrasts(self._rows(raw))
        by_m = {float(r["multiplier_m"]): r for r in out["fold_rows"]}
        self.assertAlmostEqual(by_m[1.0]["delta_2022"], 0.0, delta=CONTRAST_ATOL)
        self.assertAlmostEqual(by_m[1.5]["delta_2022"], 1.0, delta=CONTRAST_ATOL)
        self.assertAlmostEqual(by_m[1.5]["delta_2023"], -1.0, delta=CONTRAST_ATOL)
        self.assertAlmostEqual(by_m[1.5]["mean_delta"], (1.0 - 1.0 + 3.0) / 3.0, delta=CONTRAST_ATOL)
        self.assertEqual(len(out["fold_rows"]), 5)

    def test_incomplete_grid_is_refused(self) -> None:
        rows = self._rows(
            {m: {2022: 1.0, 2023: 1.0, 2024: 1.0} for m in MULTIPLIERS}
        )
        rows = rows[:-1]
        with self.assertRaises(RuntimeError):
            compute_contrasts(rows)


class ArtifactSchemaTests(unittest.TestCase):
    def test_smoke_and_canonical_output_dirs_are_distinct(self) -> None:
        source = Path("scripts/run_weight_lag_001.py").read_text(encoding="utf-8")
        self.assertIn('default="outputs/analysis/weight_lag_001"', source)
        self.assertIn('args.out_dir + ("_smoke" if args.smoke else "")', source)
        self.assertIn("reports/weight_lag_001.md", source)
        self.assertIn("recency_audit_001", source)
        self.assertIn("_smoke", source)

    def test_does_not_append_experiments_csv(self) -> None:
        source = Path("scripts/run_weight_lag_001.py").read_text(encoding="utf-8")
        self.assertIn("not_appended_audit_convention", source)
        self.assertNotIn("append_experiment_log", source)

    def test_cache_roundtrip(self) -> None:
        from scripts.run_recency_audit_001 import append_fit_cache, load_fit_cache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fits.jsonl"
            record = {"cell_key": "abc", "metrics": {"raw_bss": 1.25}}
            append_fit_cache(path, record)
            cache = load_fit_cache(path)
            self.assertEqual(cache["abc"]["metrics"]["raw_bss"], 1.25)


class LeakageGuardTests(unittest.TestCase):
    def test_prepare_native_frames_medians_come_from_train_only(self) -> None:
        from scripts.tree_exp_common import prepare_native_frames

        X_tr = pd.DataFrame({"n": [1.0, np.nan, 3.0], "c": ["a", None, "b"]})
        X_va = pd.DataFrame({"n": [np.nan, 100.0, np.nan], "c": [None, "z", "a"]})
        tr, va = prepare_native_frames(
            X_tr, X_va, numeric_cols=["n"], categorical_cols=["c"], backend="catboost"
        )
        self.assertEqual(float(tr["n"].iloc[1]), 2.0)
        self.assertEqual(float(va["n"].iloc[0]), 2.0)
        self.assertNotIn(100.0, set(tr["n"].tolist()))
        self.assertEqual(tr["c"].iloc[1], "__NA__")

    def test_runner_ignores_test_frame(self) -> None:
        source = inspect.getsource(
            __import__("scripts.run_weight_lag_001", fromlist=["main"]).main
        )
        self.assertIn('_ = loaded["test"]', source)
        self.assertNotIn('loaded["test"][', source)

    def test_weight_policy_is_lag_not_uniform_or_calendar(self) -> None:
        self.assertEqual(WEIGHT_POLICY, "season_lag_geometric")
        self.assertIn("lag = valid_season - train_season", LAG_DEFINITION)
        import scripts.run_weight_lag_001 as runner

        self.assertFalse(hasattr(runner, "SEASON_WEIGHTS"))


if __name__ == "__main__":
    unittest.main()
