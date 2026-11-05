"""TM-003A cell aggregation, sparse fallback, expected IVB, missing values, row integrity."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.trackman_profiles import (
    MIN_CELL_N,
    MIX_RATE_COLS,
    TM_EXPECTED_IVB_BY_COUNT,
    TM_EXPECTED_IVB_GLOBAL,
    build_season_ivb_profile,
    compute_tm003a_features,
    make_count_code,
    resolve_group_ivb,
)


def _tm_rows(season: int, group: str, count: str, ivbs: list[float | None]) -> list[dict[str, object]]:
    balls, strikes = (int(x) for x in count.split("-"))
    rows = []
    for i, ivb in enumerate(ivbs):
        rows.append(
            {
                "season": season,
                "pitch_type_group": group,
                "balls_before": balls,
                "strikes_before": strikes,
                "induced_vert_break": ivb,
            }
        )
    return rows


def _pad_group(season: int, group: str, ivb: float, n: int, count: str = "3-2") -> list[dict[str, object]]:
    return _tm_rows(season, group, count, [ivb] * n)


def _main_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class Tm003AExpectedIvbTests(unittest.TestCase):
    def test_count_grouping_uses_balls_strikes_codes(self) -> None:
        frame = pd.DataFrame(
            {
                "balls_before": [0, 3, 4, np.nan],
                "strikes_before": [0, 2, 1, 1],
            }
        )
        codes = make_count_code(frame["balls_before"], frame["strikes_before"])
        self.assertEqual(list(codes.iloc[:2]), ["0-0", "3-2"])
        self.assertTrue(pd.isna(codes.iloc[2]))
        self.assertTrue(pd.isna(codes.iloc[3]))

    def test_cell_n_is_full_row_count_including_missing_ivb(self) -> None:
        ivbs: list[float | None] = [10.0] * 900 + [None] * 100
        tm = pd.DataFrame(
            _tm_rows(2019, "fastball", "0-0", ivbs)
            + _pad_group(2019, "breaking", 1.0, MIN_CELL_N)
            + _pad_group(2019, "offspeed", 2.0, MIN_CELL_N)
        )
        profile = build_season_ivb_profile(tm, 2020)
        n, mean = profile["cell_n"][("0-0", "fastball")], profile["cell_mean"][("0-0", "fastball")]
        self.assertEqual(n, 1000)
        self.assertAlmostEqual(mean, 10.0)
        value, used_fallback = resolve_group_ivb(profile, "0-0", "fastball")
        self.assertFalse(used_fallback)
        self.assertAlmostEqual(value, 10.0)

    def test_sparse_fallback_boundaries_999_1000_1001(self) -> None:
        tm = pd.DataFrame(
            _tm_rows(2019, "fastball", "0-0", [10.0] * 999)
            + _tm_rows(2019, "fastball", "0-1", [20.0] * 1000)
            + _tm_rows(2019, "fastball", "0-2", [30.0] * 1001)
            + _pad_group(2019, "breaking", 1.0, MIN_CELL_N)
            + _pad_group(2019, "offspeed", 2.0, MIN_CELL_N)
        )
        profile = build_season_ivb_profile(tm, 2020)
        global_fb = float(profile["global_mean"]["fastball"])
        v999, fb999 = resolve_group_ivb(profile, "0-0", "fastball")
        v1000, fb1000 = resolve_group_ivb(profile, "0-1", "fastball")
        v1001, fb1001 = resolve_group_ivb(profile, "0-2", "fastball")
        self.assertEqual(profile["cell_n"][("0-0", "fastball")], 999)
        self.assertEqual(profile["cell_n"][("0-1", "fastball")], 1000)
        self.assertEqual(profile["cell_n"][("0-2", "fastball")], 1001)
        self.assertTrue(fb999)
        self.assertAlmostEqual(v999, global_fb)
        self.assertFalse(fb1000)
        self.assertAlmostEqual(v1000, 20.0)
        self.assertFalse(fb1001)
        self.assertAlmostEqual(v1001, 30.0)
        expected_global = (999 * 10.0 + 1000 * 20.0 + 1001 * 30.0) / (999 + 1000 + 1001)
        self.assertAlmostEqual(global_fb, expected_global)

    def test_expected_ivb_plain_sum_matches_hand_calculation(self) -> None:
        tm = pd.DataFrame(
            _tm_rows(2019, "fastball", "0-0", [10.0] * MIN_CELL_N)
            + _tm_rows(2019, "breaking", "0-0", [1.0] * MIN_CELL_N)
            + _tm_rows(2019, "offspeed", "0-0", [2.0] * MIN_CELL_N)
        )
        frame = _main_frame(
            [
                {
                    "row_id": 1,
                    "season": 2020,
                    "balls_before": 0,
                    "strikes_before": 0,
                    "asof_pitcher_pitchmix_n": 12,
                    "asof_pitcher_fastball_rate": 0.5,
                    "asof_pitcher_breaking_rate": 0.3,
                    "asof_pitcher_offspeed_rate": 0.2,
                    "control_success": 1,
                }
            ]
        )
        out = compute_tm003a_features(frame, tm)
        expected = 0.5 * 10.0 + 0.3 * 1.0 + 0.2 * 2.0
        self.assertAlmostEqual(float(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0]), expected)
        self.assertAlmostEqual(float(out[TM_EXPECTED_IVB_GLOBAL].iloc[0]), expected)
        self.assertFalse(bool(out["tm003a_fallback_any"].iloc[0]))
        self.assertAlmostEqual(float(out["tm003a_fallback_weight_share"].iloc[0]), 0.0)

    def test_plain_sum_does_not_renormalize_mix(self) -> None:
        tm = pd.DataFrame(
            _tm_rows(2019, "fastball", "0-0", [10.0] * MIN_CELL_N)
            + _tm_rows(2019, "breaking", "0-0", [4.0] * MIN_CELL_N)
            + _tm_rows(2019, "offspeed", "0-0", [2.0] * MIN_CELL_N)
        )
        frame = _main_frame(
            [
                {
                    "row_id": 1,
                    "season": 2020,
                    "balls_before": 0,
                    "strikes_before": 0,
                    "asof_pitcher_pitchmix_n": 5,
                    "asof_pitcher_fastball_rate": 0.25,
                    "asof_pitcher_breaking_rate": 0.15,
                    "asof_pitcher_offspeed_rate": 0.10,
                    "control_success": 0,
                }
            ]
        )
        out = compute_tm003a_features(frame, tm)
        expected_plain = 0.25 * 10.0 + 0.15 * 4.0 + 0.10 * 2.0
        expected_renorm = expected_plain / 0.50
        self.assertAlmostEqual(float(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0]), expected_plain)
        self.assertNotAlmostEqual(float(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0]), expected_renorm)

    def test_fallback_weight_share_example_from_decision(self) -> None:
        tm = pd.DataFrame(
            _tm_rows(2019, "fastball", "0-0", [10.0] * MIN_CELL_N)
            + _tm_rows(2019, "breaking", "3-2", [1.0] * MIN_CELL_N)
            + _tm_rows(2019, "offspeed", "3-2", [2.0] * MIN_CELL_N)
        )
        frame = _main_frame(
            [
                {
                    "row_id": 1,
                    "season": 2020,
                    "balls_before": 0,
                    "strikes_before": 0,
                    "asof_pitcher_pitchmix_n": 8,
                    "asof_pitcher_fastball_rate": 0.60,
                    "asof_pitcher_breaking_rate": 0.30,
                    "asof_pitcher_offspeed_rate": 0.10,
                    "control_success": 1,
                }
            ]
        )
        out = compute_tm003a_features(frame, tm)
        global_br = float(build_season_ivb_profile(tm, 2020)["global_mean"]["breaking"])
        global_os = float(build_season_ivb_profile(tm, 2020)["global_mean"]["offspeed"])
        expected = 0.60 * 10.0 + 0.30 * global_br + 0.10 * global_os
        self.assertAlmostEqual(float(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0]), expected)
        self.assertTrue(bool(out["tm003a_fallback_any"].iloc[0]))
        self.assertFalse(bool(out["tm003a_fallback_fastball"].iloc[0]))
        self.assertTrue(bool(out["tm003a_fallback_breaking"].iloc[0]))
        self.assertTrue(bool(out["tm003a_fallback_offspeed"].iloc[0]))
        self.assertAlmostEqual(float(out["tm003a_fallback_weight_share"].iloc[0]), 0.40)

    def test_missing_mix_and_zero_pitchmix_are_nan_not_fallback(self) -> None:
        tm = pd.DataFrame(
            _tm_rows(2019, "fastball", "0-0", [10.0] * MIN_CELL_N)
            + _tm_rows(2019, "breaking", "0-0", [1.0] * MIN_CELL_N)
            + _tm_rows(2019, "offspeed", "0-0", [2.0] * MIN_CELL_N)
        )
        frame = _main_frame(
            [
                {
                    "row_id": 1,
                    "season": 2020,
                    "balls_before": 0,
                    "strikes_before": 0,
                    "asof_pitcher_pitchmix_n": 0,
                    "asof_pitcher_fastball_rate": 0.5,
                    "asof_pitcher_breaking_rate": 0.3,
                    "asof_pitcher_offspeed_rate": 0.2,
                    "control_success": 1,
                },
                {
                    "row_id": 2,
                    "season": 2020,
                    "balls_before": 0,
                    "strikes_before": 0,
                    "asof_pitcher_pitchmix_n": 4,
                    "asof_pitcher_fastball_rate": np.nan,
                    "asof_pitcher_breaking_rate": 0.3,
                    "asof_pitcher_offspeed_rate": 0.2,
                    "control_success": 0,
                },
            ]
        )
        out = compute_tm003a_features(frame, tm)
        self.assertTrue(np.isnan(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0]))
        self.assertTrue(np.isnan(out[TM_EXPECTED_IVB_BY_COUNT].iloc[1]))
        self.assertFalse(bool(out["tm003a_constructible"].iloc[0]))
        self.assertFalse(bool(out["tm003a_constructible"].iloc[1]))
        self.assertFalse(bool(out["tm003a_fallback_any"].iloc[0]))
        self.assertFalse(bool(out["tm003a_fallback_any"].iloc[1]))
        self.assertTrue(np.isnan(out["tm003a_fallback_weight_share"].iloc[0]))

    def test_nan_inf_ivb_do_not_poison_mean_or_create_inf_feature(self) -> None:
        ivbs: list[float | None] = [10.0] * MIN_CELL_N + [np.nan, np.inf, -np.inf]
        tm = pd.DataFrame(
            _tm_rows(2019, "fastball", "0-0", ivbs)
            + _tm_rows(2019, "breaking", "0-0", [1.0] * MIN_CELL_N)
            + _tm_rows(2019, "offspeed", "0-0", [2.0] * MIN_CELL_N)
        )
        profile = build_season_ivb_profile(tm, 2020)
        self.assertEqual(profile["cell_n"][("0-0", "fastball")], MIN_CELL_N + 3)
        self.assertAlmostEqual(profile["cell_mean"][("0-0", "fastball")], 10.0)
        frame = _main_frame(
            [
                {
                    "row_id": 1,
                    "season": 2020,
                    "balls_before": 0,
                    "strikes_before": 0,
                    "asof_pitcher_pitchmix_n": 3,
                    "asof_pitcher_fastball_rate": 0.5,
                    "asof_pitcher_breaking_rate": 0.3,
                    "asof_pitcher_offspeed_rate": 0.2,
                    "control_success": 1,
                }
            ]
        )
        out = compute_tm003a_features(frame, tm)
        value = float(out[TM_EXPECTED_IVB_BY_COUNT].iloc[0])
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, 0.5 * 10.0 + 0.3 * 1.0 + 0.2 * 2.0)

    def test_row_integrity_order_and_no_duplicates(self) -> None:
        tm = pd.DataFrame(
            _tm_rows(2019, "fastball", "0-0", [10.0] * MIN_CELL_N)
            + _tm_rows(2019, "breaking", "0-0", [1.0] * MIN_CELL_N)
            + _tm_rows(2019, "offspeed", "0-0", [2.0] * MIN_CELL_N)
            + _tm_rows(2020, "fastball", "0-0", [40.0] * MIN_CELL_N)
            + _tm_rows(2020, "breaking", "0-0", [5.0] * MIN_CELL_N)
            + _tm_rows(2020, "offspeed", "0-0", [6.0] * MIN_CELL_N)
        )
        frame = _main_frame(
            [
                {
                    "row_id": 10,
                    "season": 2021,
                    "balls_before": 0,
                    "strikes_before": 0,
                    "asof_pitcher_pitchmix_n": 2,
                    "asof_pitcher_fastball_rate": 1.0,
                    "asof_pitcher_breaking_rate": 0.0,
                    "asof_pitcher_offspeed_rate": 0.0,
                    "control_success": 1,
                },
                {
                    "row_id": 11,
                    "season": 2020,
                    "balls_before": 0,
                    "strikes_before": 0,
                    "asof_pitcher_pitchmix_n": 2,
                    "asof_pitcher_fastball_rate": 0.0,
                    "asof_pitcher_breaking_rate": 1.0,
                    "asof_pitcher_offspeed_rate": 0.0,
                    "control_success": 0,
                },
                {
                    "row_id": 12,
                    "season": 2019,
                    "balls_before": 0,
                    "strikes_before": 0,
                    "asof_pitcher_pitchmix_n": 2,
                    "asof_pitcher_fastball_rate": 0.0,
                    "asof_pitcher_breaking_rate": 0.0,
                    "asof_pitcher_offspeed_rate": 1.0,
                    "control_success": 1,
                },
            ]
        )
        shuffled = frame.iloc[[2, 0, 1]].copy()
        out = compute_tm003a_features(shuffled, tm)
        self.assertEqual(len(out), len(shuffled))
        self.assertTrue(out.index.equals(shuffled.index))
        self.assertEqual(list(out.index), list(shuffled.index))
        self.assertTrue(np.isnan(out.iloc[0][TM_EXPECTED_IVB_BY_COUNT]))
        self.assertAlmostEqual(float(out.iloc[1][TM_EXPECTED_IVB_BY_COUNT]), (10.0 + 40.0) / 2.0)
        self.assertAlmostEqual(float(out.iloc[2][TM_EXPECTED_IVB_BY_COUNT]), 1.0)
        self.assertEqual(shuffled["row_id"].tolist(), [12, 10, 11])
        self.assertTrue(shuffled["row_id"].is_unique)
        self.assertNotIn(TM_EXPECTED_IVB_GLOBAL, MIX_RATE_COLS)


if __name__ == "__main__":
    unittest.main()
