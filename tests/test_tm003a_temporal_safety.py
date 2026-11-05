"""TM-003A temporal safety: row-level prior-history Trackman only, no 2025, no ID join."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.trackman_profiles import (
    CORE_PITCH_GROUPS,
    FORBIDDEN_TM_SEASON,
    MIN_CELL_N,
    TM_EXPECTED_IVB_BY_COUNT,
    TM_EXPECTED_IVB_GLOBAL,
    assert_row_level_trackman_safety,
    compute_tm003a_features,
    filter_trackman_prior_history_ivb,
    prior_history_seasons,
)


def _tm_rows(
    season: int,
    group: str,
    count: str,
    ivb: float,
    n: int,
) -> list[dict[str, object]]:
    balls, strikes = (int(x) for x in count.split("-"))
    return [
        {
            "season": season,
            "pitch_type_group": group,
            "balls_before": balls,
            "strikes_before": strikes,
            "induced_vert_break": ivb,
            "pitcher_trackman_id": 1000 + i,
        }
        for i in range(n)
    ]


def _full_group_tm(season: int, ivb_by_group: dict[str, float], n: int = MIN_CELL_N) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, ivb in ivb_by_group.items():
        rows.extend(_tm_rows(season, group, "0-0", ivb, n))
    return rows


def _main_row(
    season: int,
    count: str = "0-0",
    mix: tuple[float, float, float] = (0.5, 0.3, 0.2),
    pitchmix_n: int = 10,
    row_id: int = 1,
) -> dict[str, object]:
    balls, strikes = (int(x) for x in count.split("-"))
    return {
        "row_id": row_id,
        "season": season,
        "balls_before": balls,
        "strikes_before": strikes,
        "asof_pitcher_pitchmix_n": pitchmix_n,
        "asof_pitcher_fastball_rate": mix[0],
        "asof_pitcher_breaking_rate": mix[1],
        "asof_pitcher_offspeed_rate": mix[2],
        "control_success": 1,
    }


class Tm003ATemporalSafetyTests(unittest.TestCase):
    def test_prior_history_seasons_by_row_year(self) -> None:
        self.assertEqual(prior_history_seasons(2019), ())
        self.assertEqual(prior_history_seasons(2020), (2019,))
        self.assertEqual(prior_history_seasons(2022), (2019, 2020, 2021))
        self.assertEqual(prior_history_seasons(2023), (2019, 2020, 2021, 2022))
        self.assertEqual(prior_history_seasons(2024), (2019, 2020, 2021, 2022, 2023))
        self.assertEqual(prior_history_seasons(2025), (2019, 2020, 2021, 2022, 2023, 2024))
        self.assertNotIn(FORBIDDEN_TM_SEASON, prior_history_seasons(2025))
        self.assertNotIn(FORBIDDEN_TM_SEASON, prior_history_seasons(2026))

    def test_2022_validation_excludes_2022_and_later_trackman(self) -> None:
        tm = pd.DataFrame(
            _full_group_tm(2021, {"fastball": 1.0, "breaking": 2.0, "offspeed": 3.0})
            + _full_group_tm(2022, {"fastball": 100.0, "breaking": 100.0, "offspeed": 100.0})
            + _full_group_tm(2025, {"fastball": 999.0, "breaking": 999.0, "offspeed": 999.0})
        )
        hist = filter_trackman_prior_history_ivb(tm, 2022)
        self.assertTrue(hist.empty or max(hist["season"]) < 2022)
        self.assertNotIn(2022, set(hist["season"]))
        self.assertNotIn(2025, set(hist["season"]))
        with self.assertRaises(AssertionError):
            assert_row_level_trackman_safety(tm.loc[tm["season"] >= 2022], 2022)

    def test_2023_and_2024_row_cutoffs(self) -> None:
        tm = pd.DataFrame(
            _full_group_tm(2022, {"fastball": 1.0, "breaking": 2.0, "offspeed": 3.0})
            + _full_group_tm(2023, {"fastball": 10.0, "breaking": 20.0, "offspeed": 30.0})
            + _full_group_tm(2024, {"fastball": 40.0, "breaking": 50.0, "offspeed": 60.0})
        )
        h23 = filter_trackman_prior_history_ivb(tm, 2023)
        h24 = filter_trackman_prior_history_ivb(tm, 2024)
        h25 = filter_trackman_prior_history_ivb(tm, 2025)
        self.assertTrue((h23["season"] < 2023).all())
        self.assertTrue((h24["season"] < 2024).all())
        self.assertTrue((h25["season"] < 2025).all())
        self.assertNotIn(2025, set(h25["season"]))

    def test_same_season_trackman_never_enters_row_feature(self) -> None:
        tm = pd.DataFrame(
            _full_group_tm(2021, {"fastball": 10.0, "breaking": 1.0, "offspeed": 2.0})
            + _full_group_tm(2022, {"fastball": 90.0, "breaking": 90.0, "offspeed": 90.0})
        )
        frame = pd.DataFrame([_main_row(2022, row_id=1)])
        out = compute_tm003a_features(frame, tm)
        expected = 0.5 * 10.0 + 0.3 * 1.0 + 0.2 * 2.0
        self.assertAlmostEqual(float(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0]), expected)
        self.assertLess(float(out["tm003a_max_tm_season_used"].iloc[0]), 2022)

        tm_future_only = pd.DataFrame(
            _full_group_tm(2022, {"fastball": 90.0, "breaking": 90.0, "offspeed": 90.0})
        )
        out_future = compute_tm003a_features(frame, tm_future_only)
        self.assertTrue(np.isnan(out_future[TM_EXPECTED_IVB_BY_COUNT].iloc[0]))

    def test_2024_fold_train_2023_row_cannot_see_2023_trackman(self) -> None:
        tm = pd.DataFrame(
            _full_group_tm(2022, {"fastball": 4.0, "breaking": 5.0, "offspeed": 6.0})
            + _full_group_tm(2023, {"fastball": 40.0, "breaking": 50.0, "offspeed": 60.0})
        )
        frame = pd.DataFrame(
            [
                _main_row(2023, row_id=1),
                _main_row(2024, row_id=2),
            ]
        )
        out = compute_tm003a_features(frame, tm)
        expected_2023 = 0.5 * 4.0 + 0.3 * 5.0 + 0.2 * 6.0
        expected_2024 = (
            0.5 * ((4.0 + 40.0) / 2.0)
            + 0.3 * ((5.0 + 50.0) / 2.0)
            + 0.2 * ((6.0 + 60.0) / 2.0)
        )
        # 2023 train row in a 2024 fold still uses trackman.season < 2023.
        self.assertAlmostEqual(float(out.loc[0, TM_EXPECTED_IVB_BY_COUNT]), expected_2023)
        self.assertAlmostEqual(float(out.loc[1, TM_EXPECTED_IVB_BY_COUNT]), expected_2024)
        self.assertEqual(float(out.loc[0, "tm003a_max_tm_season_used"]), 2022)
        self.assertEqual(float(out.loc[1, "tm003a_max_tm_season_used"]), 2023)

    def test_2019_rows_are_nan_and_use_no_trackman(self) -> None:
        tm = pd.DataFrame(
            _full_group_tm(2019, {"fastball": 8.0, "breaking": 9.0, "offspeed": 10.0})
        )
        frame = pd.DataFrame([_main_row(2019, row_id=1)])
        out = compute_tm003a_features(frame, tm)
        self.assertTrue(np.isnan(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0]))
        self.assertTrue(np.isnan(out["tm003a_max_tm_season_used"].iloc[0]))
        hist = filter_trackman_prior_history_ivb(tm, 2019)
        self.assertTrue(hist.empty)

    def test_2025_trackman_never_enters_any_row_profile(self) -> None:
        tm = pd.DataFrame(
            _full_group_tm(2024, {"fastball": 7.0, "breaking": 8.0, "offspeed": 9.0})
            + _full_group_tm(2025, {"fastball": 700.0, "breaking": 800.0, "offspeed": 900.0})
        )
        frame = pd.DataFrame([_main_row(2025, row_id=1)])
        out = compute_tm003a_features(frame, tm)
        expected = 0.5 * 7.0 + 0.3 * 8.0 + 0.2 * 9.0
        self.assertAlmostEqual(float(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0]), expected)
        self.assertEqual(float(out["tm003a_max_tm_season_used"].iloc[0]), 2024)
        for row_season in (2022, 2023, 2024, 2025):
            hist = filter_trackman_prior_history_ivb(tm, row_season)
            self.assertNotIn(2025, set(hist["season"]))

    def test_other_group_and_id_columns_are_not_required(self) -> None:
        tm = pd.DataFrame(
            _full_group_tm(2021, {"fastball": 1.0, "breaking": 2.0, "offspeed": 3.0})
            + _tm_rows(2021, "other", "0-0", 50.0, MIN_CELL_N)
        )
        frame = pd.DataFrame([_main_row(2022, row_id=1)])
        out = compute_tm003a_features(frame, tm)
        expected = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0
        self.assertAlmostEqual(float(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0]), expected)
        hist = filter_trackman_prior_history_ivb(tm, 2022)
        self.assertTrue(set(CORE_PITCH_GROUPS).issubset(set(hist["pitch_type_group"])))

    def test_diagnostic_global_is_not_the_model_feature(self) -> None:
        self.assertNotEqual(TM_EXPECTED_IVB_BY_COUNT, TM_EXPECTED_IVB_GLOBAL)


if __name__ == "__main__":
    unittest.main()
