"""Unit / contract tests for RECENCY-AUDIT-001. No full CatBoost training."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_exp004_catboost_weighted import fit_catboost
from scripts.run_full_train_exp005 import ID_DROP
from scripts.run_l42_2019_2023_lb_probe import assert_l42_contract, l42_feature_lists
from scripts.run_recency_audit_001 import (
    CONDITIONS,
    CONTRAST_IDENTITY_ATOL,
    EXPECTED_CATBOOST_PARAMS,
    EXPECTED_SEED,
    LOCKED_WINDOWS,
    VALID_SEASONS,
    VERDICT_HOLD,
    VERDICT_KEEP,
    VERDICT_SHORTEN,
    WEIGHT_POLICY,
    assert_canonical_controls,
    assert_locked_window_contract,
    assert_uniform_sample_weight,
    compute_contrasts,
    evaluate_predictions,
    iter_cell_specs,
    predeclared_verdict,
    reference_brier_score,
    uniform_sample_weight,
)
from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from src.evaluate import build_season_holdout, evaluate_predictions as canonical_evaluate
from src.metrics import reference_brier_score as canonical_reference_brier
from src.utils import load_config
from tests.test_l42_2019_2023_lb_probe import BASELINE_44


class WindowContractTests(unittest.TestCase):
    def test_nine_cells_and_locked_windows(self) -> None:
        cells = iter_cell_specs()
        self.assertEqual(len(cells), 9)
        seen = {(c.valid_season, c.condition) for c in cells}
        expected = {(y, cond) for y in VALID_SEASONS for cond in CONDITIONS}
        self.assertEqual(seen, expected)
        for cell in cells:
            self.assertEqual(
                list(cell.train_seasons),
                LOCKED_WINDOWS[cell.valid_season][cell.condition],
            )

    def test_valid_season_never_in_train(self) -> None:
        for cell in iter_cell_specs():
            self.assertNotIn(cell.valid_season, cell.train_seasons)
            self.assertTrue(max(cell.train_seasons) < cell.valid_season)

    def test_a_and_c_have_same_season_count(self) -> None:
        by_valid = {}
        for cell in iter_cell_specs():
            by_valid.setdefault(cell.valid_season, {})[cell.condition] = cell
        for valid in VALID_SEASONS:
            self.assertEqual(
                len(by_valid[valid]["A_gap"].train_seasons),
                len(by_valid[valid]["C_shifted"].train_seasons),
            )

    def test_b_equals_c_plus_oldest_season(self) -> None:
        by_valid = {}
        for cell in iter_cell_specs():
            by_valid.setdefault(cell.valid_season, {})[cell.condition] = cell
        for valid in VALID_SEASONS:
            b = by_valid[valid]["B_expanding"].train_seasons
            c = by_valid[valid]["C_shifted"].train_seasons
            oldest = min(b)
            self.assertEqual(tuple(s for s in b if s != oldest), c)
            self.assertEqual(oldest, 2019)

    def test_exact_locked_windows(self) -> None:
        self.assertEqual(LOCKED_WINDOWS[2022]["A_gap"], [2019, 2020])
        self.assertEqual(LOCKED_WINDOWS[2022]["B_expanding"], [2019, 2020, 2021])
        self.assertEqual(LOCKED_WINDOWS[2022]["C_shifted"], [2020, 2021])
        self.assertEqual(LOCKED_WINDOWS[2023]["A_gap"], [2019, 2020, 2021])
        self.assertEqual(LOCKED_WINDOWS[2023]["B_expanding"], [2019, 2020, 2021, 2022])
        self.assertEqual(LOCKED_WINDOWS[2023]["C_shifted"], [2020, 2021, 2022])
        self.assertEqual(LOCKED_WINDOWS[2024]["A_gap"], [2019, 2020, 2021, 2022])
        self.assertEqual(LOCKED_WINDOWS[2024]["B_expanding"], [2019, 2020, 2021, 2022, 2023])
        self.assertEqual(LOCKED_WINDOWS[2024]["C_shifted"], [2020, 2021, 2022, 2023])
        assert_locked_window_contract()

    def test_holdout_builder_rejects_valid_in_train(self) -> None:
        frame = pd.DataFrame({"season": [2019, 2020, 2021], "x": [1, 2, 3]})
        with self.assertRaises(ValueError):
            build_season_holdout(frame, "season", [2019, 2020, 2021], [2021], name="leak")


class WeightPolicyTests(unittest.TestCase):
    def test_every_training_weight_is_exactly_one(self) -> None:
        weights = uniform_sample_weight(17)
        self.assertEqual(len(weights), 17)
        self.assertTrue(np.all(weights == 1.0))
        assert_uniform_sample_weight(weights)

    def test_rejects_season_weights(self) -> None:
        seasonal = np.asarray([float(SEASON_WEIGHTS[2019]), float(SEASON_WEIGHTS[2023])])
        with self.assertRaises(AssertionError):
            assert_uniform_sample_weight(seasonal)

    def test_recency_runner_does_not_import_season_weights_into_fit(self) -> None:
        source = inspect.getsource(uniform_sample_weight)
        self.assertNotIn("SEASON_WEIGHTS", source)
        import scripts.run_recency_audit_001 as runner

        self.assertFalse(hasattr(runner, "SEASON_WEIGHTS"))
        self.assertEqual(runner.WEIGHT_POLICY, "uniform_1.0")


class CanonicalControlTests(unittest.TestCase):
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
        recency = l42_feature_lists(
            columns,
            target_col=config["data"]["target_col"],
            exclude_cols=list(feat["exclude_cols"]),
            categorical_cols=list(feat["categorical_cols"]),
        )
        return recency

    def test_feature_list_matches_canonical_l42(self) -> None:
        used, nums, cats = self._l42()
        canonical = l42_feature_lists(
            list(BASELINE_44)
            + ["row_id", "control_success", "home_win_expectancy", "away_win_expectancy", "li"],
            target_col="control_success",
            exclude_cols=list(load_config("configs/baseline.yaml")["features"]["exclude_cols"]),
            categorical_cols=list(
                load_config("configs/baseline.yaml")["features"]["categorical_cols"]
            ),
        )
        self.assertEqual(used, canonical[0])
        self.assertEqual(nums, canonical[1])
        self.assertEqual(cats, canonical[2])
        self.assertEqual(len(used), 42)
        assert_l42_contract(used, cats)
        for dropped in ID_DROP:
            self.assertNotIn(dropped, used)
            self.assertNotIn(dropped, cats)
        self.assertIn("game_type", used)
        self.assertIn("game_type", cats)

    def test_dropped_ids_match_canonical(self) -> None:
        self.assertEqual(list(ID_DROP), ["pitcher_id", "batter_id"])

    def test_seed_and_catboost_params_match_baseline(self) -> None:
        config = load_config("configs/baseline.yaml")
        seed = int(config["runtime"]["seed"])
        params = dict(config["models"]["params"]["catboost"])
        self.assertEqual(seed, EXPECTED_SEED)
        self.assertEqual(params, EXPECTED_CATBOOST_PARAMS)
        assert_canonical_controls(params, seed)

    def test_metric_is_canonical_function(self) -> None:
        self.assertIs(evaluate_predictions, canonical_evaluate)
        self.assertIs(reference_brier_score, canonical_reference_brier)
        import scripts.run_recency_audit_001 as runner

        self.assertIs(runner.evaluate_predictions, canonical_evaluate)
        self.assertIs(runner.fit_catboost, fit_catboost)


class ContrastTests(unittest.TestCase):
    @staticmethod
    def _rows(a: dict[int, float], b: dict[int, float], c: dict[int, float]) -> list[dict]:
        rows = []
        for valid in VALID_SEASONS:
            rows.append({"valid_season": valid, "condition": "A_gap", "raw_bss": a[valid]})
            rows.append({"valid_season": valid, "condition": "B_expanding", "raw_bss": b[valid]})
            rows.append({"valid_season": valid, "condition": "C_shifted", "raw_bss": c[valid]})
        return rows

    def test_identity_holds(self) -> None:
        a = {2022: 10.0, 2023: 20.0, 2024: 30.0}
        b = {2022: 13.0, 2023: 17.0, 2024: 29.0}
        c = {2022: 12.0, 2023: 19.0, 2024: 31.0}
        out = compute_contrasts(self._rows(a, b, c))
        for row in out["fold_rows"]:
            reconstructed = row["C_minus_A"] + row["B_minus_C"]
            self.assertAlmostEqual(row["B_minus_A"], reconstructed, delta=CONTRAST_IDENTITY_ATOL)
            self.assertLessEqual(abs(row["identity_residual"]), CONTRAST_IDENTITY_ATOL)

    def test_incomplete_results_are_refused(self) -> None:
        rows = self._rows(
            {2022: 1, 2023: 1, 2024: 1},
            {2022: 2, 2023: 2, 2024: 2},
            {2022: 3, 2023: 3, 2024: 3},
        )
        rows = [r for r in rows if not (r["valid_season"] == 2024 and r["condition"] == "C_shifted")]
        with self.assertRaises(RuntimeError):
            compute_contrasts(rows)

    def test_shorten_candidate(self) -> None:
        # B < C on all folds; C worst >= B worst.
        b_minus_c = {2022: -5.0, 2023: -1.0, 2024: -2.0}
        b_raw = {2022: 10.0, 2023: 8.0, 2024: 9.0}
        c_raw = {2022: 15.0, 2023: 9.0, 2024: 11.0}
        verdict = predeclared_verdict(b_minus_c=b_minus_c, b_raw=b_raw, c_raw=c_raw)
        self.assertEqual(verdict["bucket"], VERDICT_SHORTEN)
        self.assertTrue(all(verdict["shorten_checks"].values()))

    def test_keep_candidate(self) -> None:
        b_minus_c = {2022: 4.0, 2023: 1.0, 2024: 2.0}
        b_raw = {2022: 14.0, 2023: 9.0, 2024: 11.0}
        c_raw = {2022: 10.0, 2023: 8.0, 2024: 9.0}
        verdict = predeclared_verdict(b_minus_c=b_minus_c, b_raw=b_raw, c_raw=c_raw)
        self.assertEqual(verdict["bucket"], VERDICT_KEEP)

    def test_hold_when_mean_and_majority_conflict(self) -> None:
        # mean positive from a large 2023, but 2/3 folds negative.
        b_minus_c = {2022: -1.0, 2023: 10.0, 2024: -1.0}
        b_raw = {2022: 10.0, 2023: 20.0, 2024: 10.0}
        c_raw = {2022: 11.0, 2023: 10.0, 2024: 11.0}
        verdict = predeclared_verdict(b_minus_c=b_minus_c, b_raw=b_raw, c_raw=c_raw)
        self.assertEqual(verdict["bucket"], VERDICT_HOLD)
        self.assertGreater(verdict["mean_B_minus_C"], 0)
        self.assertEqual(verdict["n_negative_folds"], 2)

    def test_hold_when_worst_fold_deteriorates(self) -> None:
        # mean and majority favor dropping old history, but C worst < B worst.
        b_minus_c = {2022: -3.0, 2023: -3.0, 2024: -3.0}
        b_raw = {2022: 12.0, 2023: 11.0, 2024: 10.0}
        c_raw = {2022: 15.0, 2023: 14.0, 2024: 5.0}
        verdict = predeclared_verdict(b_minus_c=b_minus_c, b_raw=b_raw, c_raw=c_raw)
        self.assertEqual(verdict["bucket"], VERDICT_HOLD)
        self.assertFalse(verdict["shorten_checks"]["C_worst_not_below_B_worst"])


class ArtifactSchemaTests(unittest.TestCase):
    def test_contrasts_csv_schema_includes_aggregates(self) -> None:
        from scripts.run_recency_audit_001 import contrasts_table

        a = {2022: 1.0, 2023: 2.0, 2024: 3.0}
        b = {2022: 1.5, 2023: 1.0, 2024: 2.5}
        c = {2022: 2.0, 2023: 2.5, 2024: 4.0}
        rows = []
        for valid in VALID_SEASONS:
            rows.append({"valid_season": valid, "condition": "A_gap", "raw_bss": a[valid]})
            rows.append({"valid_season": valid, "condition": "B_expanding", "raw_bss": b[valid]})
            rows.append({"valid_season": valid, "condition": "C_shifted", "raw_bss": c[valid]})
        table = contrasts_table(compute_contrasts(rows))
        self.assertEqual(
            list(table.columns),
            ["valid_season", "B_minus_A", "C_minus_A", "B_minus_C"],
        )
        self.assertEqual(
            list(table["valid_season"]),
            ["2022", "2023", "2024", "mean", "std", "worst", "best"],
        )

    def test_smoke_and_canonical_output_dirs_are_distinct(self) -> None:
        source = Path("scripts/run_recency_audit_001.py").read_text(encoding="utf-8")
        self.assertIn('default="outputs/analysis/recency_audit_001"', source)
        self.assertIn('args.out_dir + ("_smoke" if args.smoke else "")', source)
        self.assertIn("reports/recency_audit_001.md", source)


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
            __import__("scripts.run_recency_audit_001", fromlist=["main"]).main
        )
        self.assertIn("_ = loaded[\"test\"]", source)
        self.assertNotIn("loaded[\"test\"][", source)


class ResumeCacheTests(unittest.TestCase):
    def test_append_cache_roundtrip(self) -> None:
        from scripts.run_recency_audit_001 import append_fit_cache, load_fit_cache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fits.jsonl"
            record = {"cell_key": "abc", "metrics": {"raw_bss": 1.25}}
            append_fit_cache(path, record)
            append_fit_cache(path, {"cell_key": "def", "metrics": {"raw_bss": 2.5}})
            cache = load_fit_cache(path)
            self.assertEqual(cache["abc"]["metrics"]["raw_bss"], 1.25)
            self.assertEqual(set(cache), {"abc", "def"})


if __name__ == "__main__":
    unittest.main()
