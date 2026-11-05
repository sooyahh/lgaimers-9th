"""MODEL-TRANSFER-AUDIT-001 aggregation: pairing, fold-level noise, transfer verdict."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_model_transfer_audit_001 import (
    BASE_VARIANT,
    ID_DROP,
    VARIANTS,
    base_feature_lists,
    build_fold_specs,
    cell_key,
    paired_deltas,
    summarize_deltas,
    transfer_verdict,
    variant_feature_lists,
)

FOLDS = ("secondary_2022", "secondary_2023", "primary_2024")


def _run(variant: str, backend: str, fold_id: str, seed: int, raw_bss: float) -> dict:
    return {
        "variant": variant,
        "backend": backend,
        "fold_id": fold_id,
        "seed": seed,
        "raw_brier_skill_score": raw_bss,
    }


def _deltas(values_by_fold: dict[str, list[float]]) -> list[dict]:
    out = []
    for fold_id, values in values_by_fold.items():
        for i, value in enumerate(values):
            out.append({"fold_id": fold_id, "seed": i, "delta_raw_bss": value})
    return out


class PairingTests(unittest.TestCase):
    def test_delta_pairs_on_same_backend_fold_and_seed(self) -> None:
        rows = [
            _run(BASE_VARIANT, "catboost", "primary_2024", 42, 100.0),
            _run(BASE_VARIANT, "catboost", "primary_2024", 1, 200.0),
            _run("FE-002", "catboost", "primary_2024", 42, 110.0),
            _run("FE-002", "catboost", "primary_2024", 1, 190.0),
        ]
        deltas = paired_deltas(rows, variant="FE-002", backend="catboost")
        self.assertEqual([d["delta_raw_bss"] for d in deltas], [10.0, -10.0])

    def test_unpaired_cells_are_dropped(self) -> None:
        rows = [
            _run(BASE_VARIANT, "catboost", "primary_2024", 42, 100.0),
            _run("FE-002", "catboost", "primary_2024", 42, 110.0),
            _run("FE-002", "catboost", "primary_2024", 7, 999.0),  # no BASE at seed 7
        ]
        deltas = paired_deltas(rows, variant="FE-002", backend="catboost")
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["seed"], 42)

    def test_does_not_pair_across_backends(self) -> None:
        rows = [
            _run(BASE_VARIANT, "lightgbm", "primary_2024", 42, 100.0),
            _run("FE-002", "catboost", "primary_2024", 42, 110.0),
        ]
        self.assertEqual(paired_deltas(rows, variant="FE-002", backend="catboost"), [])


class NoiseTests(unittest.TestCase):
    def test_fold_means_are_the_unit_of_observation(self) -> None:
        # Fold means are +10, +10, +10 regardless of the within-fold seed spread.
        stats = summarize_deltas(
            _deltas({"a": [0.0, 20.0], "b": [5.0, 15.0], "c": [9.0, 11.0]})
        )
        self.assertEqual(stats["n_folds"], 3)
        self.assertEqual(stats["n_paired_cells"], 6)
        self.assertAlmostEqual(stats["delta_mean"], 10.0)
        self.assertAlmostEqual(stats["delta_std_across_folds"], 0.0)

    def test_seed_spread_is_reported_separately_from_fold_std(self) -> None:
        stats = summarize_deltas(_deltas({"a": [0.0, 20.0], "b": [0.0, 20.0]}))
        self.assertGreater(stats["seed_spread_within_fold"], 0.0)
        self.assertTrue(stats["seed_axis_informative"])
        self.assertAlmostEqual(stats["delta_std_across_folds"], 0.0)

    def test_deterministic_backend_is_flagged(self) -> None:
        """LightGBM under the configured params is identical across seeds."""
        stats = summarize_deltas(_deltas({"a": [7.0, 7.0], "b": [-3.0, -3.0]}))
        self.assertAlmostEqual(stats["seed_spread_within_fold"], 0.0)
        self.assertFalse(stats["seed_axis_informative"])

    def test_duplicated_seeds_do_not_shrink_the_standard_error(self) -> None:
        one_seed = summarize_deltas(_deltas({"a": [30.0], "b": [10.0], "c": [20.0]}))
        three_identical_seeds = summarize_deltas(
            _deltas({"a": [30.0] * 3, "b": [10.0] * 3, "c": [20.0] * 3})
        )
        self.assertAlmostEqual(one_seed["delta_sem"], three_identical_seeds["delta_sem"])

    def test_large_mean_relative_to_fold_spread_is_material(self) -> None:
        stats = summarize_deltas(_deltas({"a": [100.0], "b": [102.0], "c": [98.0]}))
        self.assertEqual(stats["materiality"], "MATERIAL")

    def test_mean_inside_fold_spread_is_within_noise(self) -> None:
        stats = summarize_deltas(_deltas({"a": [-20.0], "b": [25.0], "c": [-2.0]}))
        self.assertEqual(stats["materiality"], "WITHIN_NOISE")

    def test_single_fold_cannot_be_material(self) -> None:
        stats = summarize_deltas(_deltas({"a": [100.0, 101.0]}))
        self.assertEqual(stats["materiality"], "n/a")


class VerdictTests(unittest.TestCase):
    @staticmethod
    def _block(mean: float, materiality: str, by_fold: dict[str, float]) -> dict:
        return {"delta_mean": mean, "materiality": materiality, "delta_mean_by_fold": by_fold}

    def test_same_sign_and_material_transfers(self) -> None:
        verdict = transfer_verdict(
            {
                "catboost": self._block(20.0, "MATERIAL", {"a": 20.0, "b": 20.0}),
                "lightgbm": self._block(15.0, "MATERIAL", {"a": 15.0, "b": 15.0}),
            }
        )
        self.assertEqual(verdict["verdict"], "TRANSFERS")
        self.assertEqual(verdict["fold_sign_agreement"], "2/2")

    def test_opposite_sign_and_material_is_model_specific(self) -> None:
        verdict = transfer_verdict(
            {
                "catboost": self._block(20.0, "MATERIAL", {"a": 20.0}),
                "lightgbm": self._block(-15.0, "MATERIAL", {"a": -15.0}),
            }
        )
        self.assertEqual(verdict["verdict"], "MODEL_SPECIFIC")
        self.assertFalse(verdict["sign_agreement_on_overall_mean"])

    def test_within_noise_everywhere_is_inconclusive_not_a_reject(self) -> None:
        verdict = transfer_verdict(
            {
                "catboost": self._block(-7.0, "WITHIN_NOISE", {"a": -7.0}),
                "lightgbm": self._block(-2.0, "WITHIN_NOISE", {"a": -2.0}),
            }
        )
        self.assertEqual(verdict["verdict"], "INCONCLUSIVE_NOISE")
        self.assertIn("measurement", verdict["reading"])

    def test_material_in_one_backend_only_is_partial(self) -> None:
        verdict = transfer_verdict(
            {
                "catboost": self._block(40.0, "MATERIAL", {"a": 40.0}),
                "lightgbm": self._block(1.0, "WITHIN_NOISE", {"a": 1.0}),
            }
        )
        self.assertEqual(verdict["verdict"], "PARTIAL_catboost")


class FeatureListTests(unittest.TestCase):
    COLUMNS = [
        "row_id",
        "season",
        "control_success",
        "pitcher_id",
        "batter_id",
        "game_type",
        "asof_pitcher_ball_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_success_rate",
        "asof_pitcher_n",
    ]
    EXCLUDE = ["row_id"]
    CATEGORICAL = ["pitcher_id", "batter_id", "game_type"]

    def _base(self, feature_set: str):
        frame = pd.DataFrame({c: [0] for c in self.COLUMNS})
        return base_feature_lists(
            frame,
            target_col="control_success",
            exclude_cols=list(self.EXCLUDE),
            categorical_cols=list(self.CATEGORICAL),
            feature_set=feature_set,
        )

    def test_l42_drops_the_id_columns(self) -> None:
        used, _, cats = self._base("L42")
        for col in ID_DROP:
            self.assertNotIn(col, used)
            self.assertNotIn(col, cats)
        self.assertIn("game_type", used)

    def test_full44_keeps_the_id_columns(self) -> None:
        used, _, cats = self._base("full44")
        for col in ID_DROP:
            self.assertIn(col, used)
            self.assertIn(col, cats)

    def test_drop_variant_removes_exactly_one_column(self) -> None:
        used, nums, cats = self._base("L42")
        v_used, v_nums, _ = variant_feature_lists(
            VARIANTS["FE-002A"],
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            train_columns=list(self.COLUMNS),
        )
        self.assertEqual(len(v_used), len(used) - 1)
        self.assertNotIn("asof_pitcher_prev1_game_success_rate", v_used)
        self.assertNotIn("asof_pitcher_prev1_game_success_rate", v_nums)

    def test_shrinkage_variant_keeps_the_column_count(self) -> None:
        """FE-002 replaces values in place; it must not change the feature count."""
        used, nums, cats = self._base("L42")
        v_used, _, _ = variant_feature_lists(
            VARIANTS["FE-002"],
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            train_columns=list(self.COLUMNS),
        )
        self.assertEqual(v_used, used)

    def test_added_column_must_be_fold_injected(self) -> None:
        used, nums, cats = self._base("L42")
        columns = list(self.COLUMNS) + ["tm_expected_rel_speed"]
        with self.assertRaises(AssertionError):
            variant_feature_lists(
                VARIANTS["TM-002A"],
                used=used,
                numeric_cols=nums,
                categorical_cols=cats,
                train_columns=columns,
            )

    def test_added_column_appears_once_as_numeric(self) -> None:
        used, nums, cats = self._base("L42")
        v_used, v_nums, _ = variant_feature_lists(
            VARIANTS["TM-002A"],
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            train_columns=list(self.COLUMNS),
        )
        self.assertEqual(v_used.count("tm_expected_rel_speed"), 1)
        self.assertIn("tm_expected_rel_speed", v_nums)

    def test_missing_drop_column_aborts(self) -> None:
        used, nums, cats = self._base("L42")
        with self.assertRaises(AssertionError):
            variant_feature_lists(
                VARIANTS["FE-001A"],
                used=used,
                numeric_cols=nums,
                categorical_cols=cats,
                train_columns=["season", "control_success"],
            )


class CacheKeyTests(unittest.TestCase):
    BASE = {
        "feature_set": "L42",
        "variant": "FE-002",
        "backend": "catboost",
        "fold_id": "primary_2024",
        "seed": 42,
        "params": {"depth": 6},
    }

    def test_key_is_stable(self) -> None:
        self.assertEqual(cell_key(**self.BASE), cell_key(**self.BASE))

    def test_every_axis_changes_the_key(self) -> None:
        for field, value in [
            ("feature_set", "full44"),
            ("variant", "FE-002A"),
            ("backend", "lightgbm"),
            ("fold_id", "secondary_2023"),
            ("seed", 1),
            ("params", {"depth": 7}),
        ]:
            with self.subTest(field=field):
                self.assertNotEqual(cell_key(**self.BASE), cell_key(**{**self.BASE, field: value}))


class FoldSpecTests(unittest.TestCase):
    CV = {
        "primary": {"train_seasons": [2019, 2020, 2021, 2022, 2023], "valid_seasons": [2024]},
        "secondary_folds": [
            {"train_seasons": [2019, 2020, 2021], "valid_seasons": [2022]},
            {"train_seasons": [2019, 2020, 2021, 2022], "valid_seasons": [2023]},
        ],
    }

    def test_locked_folds_are_reproduced_in_order(self) -> None:
        specs = build_fold_specs(self.CV)
        self.assertEqual([s.fold_id for s in specs], list(FOLDS))
        self.assertEqual([s.role for s in specs], ["secondary", "secondary", "primary"])

    def test_multi_season_valid_fold_aborts(self) -> None:
        cv = {
            "primary": {"train_seasons": [2019], "valid_seasons": [2023, 2024]},
            "secondary_folds": self.CV["secondary_folds"],
        }
        with self.assertRaises(AssertionError):
            build_fold_specs(cv)

    def test_missing_secondary_folds_aborts(self) -> None:
        with self.assertRaises(ValueError):
            build_fold_specs({"primary": self.CV["primary"], "secondary_folds": []})


if __name__ == "__main__":
    unittest.main()
