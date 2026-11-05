"""TM-002A temporal safety: prior-season Trackman only, no ID join, no 2025."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.trackman_profiles import (
    CORE_PITCH_GROUPS,
    REL_SPEED_SUMMARY,
    TM_EXPECTED_REL_SPEED,
    allowed_trackman_seasons,
    assert_trackman_temporal_safety,
    build_group_rel_speed_profile,
    compute_tm_expected_rel_speed,
    filter_trackman_history,
    profile_means,
)


def _synthetic_tm() -> pd.DataFrame:
    rows = []
    # Skewed fastball speeds so mean != median.
    speeds = {
        "fastball": [90.0, 91.0, 140.0],
        "breaking": [80.0, 81.0, 82.0],
        "offspeed": [70.0, 71.0, 72.0],
        "other": [50.0, 51.0, 52.0],
    }
    for season in (2019, 2020, 2021, 2022, 2023, 2024, 2025):
        for group, vals in speeds.items():
            for i, speed in enumerate(vals):
                rows.append(
                    {
                        "season": season,
                        "pitch_type_group": group,
                        "rel_speed": speed + 0.1 * (season - 2019),
                        "pitcher_trackman_id": 999 + i,
                    }
                )
    return pd.DataFrame(rows)


class Tm002ATemporalSafetyTests(unittest.TestCase):
    def test_allowed_seasons_by_fold(self) -> None:
        self.assertEqual(allowed_trackman_seasons(2022), (2019, 2020, 2021))
        self.assertEqual(allowed_trackman_seasons(2023), (2019, 2020, 2021, 2022))
        self.assertEqual(allowed_trackman_seasons(2024), (2019, 2020, 2021, 2022, 2023))
        self.assertEqual(
            allowed_trackman_seasons(2025),
            (2019, 2020, 2021, 2022, 2023, 2024),
        )
        self.assertNotIn(2025, allowed_trackman_seasons(2026))

    def test_2022_fold_excludes_2022_and_later(self) -> None:
        tm = _synthetic_tm()
        hist = filter_trackman_history(tm, 2022)
        self.assertEqual(sorted(hist["season"].unique().tolist()), [2019, 2020, 2021])
        with self.assertRaises(AssertionError):
            assert_trackman_temporal_safety(tm.loc[tm["season"] >= 2022], 2022)

    def test_2023_fold_excludes_2023_and_later(self) -> None:
        hist = filter_trackman_history(_synthetic_tm(), 2023)
        self.assertTrue(max(hist["season"]) < 2023)
        self.assertNotIn(2023, set(hist["season"]))

    def test_2024_fold_excludes_2024_and_later(self) -> None:
        hist = filter_trackman_history(_synthetic_tm(), 2024)
        self.assertTrue(max(hist["season"]) < 2024)
        self.assertNotIn(2024, set(hist["season"]))
        self.assertNotIn(2025, set(hist["season"]))

    def test_2025_rows_never_enter_profile(self) -> None:
        tm = _synthetic_tm()
        self.assertIn(2025, set(tm["season"]))
        for valid in (2022, 2023, 2024, 2025):
            hist = filter_trackman_history(tm, valid)
            self.assertNotIn(2025, set(hist["season"]))
            profile = build_group_rel_speed_profile(hist, valid_season=valid)
            self.assertNotIn("other", set(profile["group"]))

    def test_other_group_excluded(self) -> None:
        hist = filter_trackman_history(_synthetic_tm(), 2024)
        profile = build_group_rel_speed_profile(hist, valid_season=2024)
        self.assertEqual(tuple(profile["group"]), CORE_PITCH_GROUPS)
        self.assertEqual(REL_SPEED_SUMMARY, "mean")

    def test_summary_is_mean_not_median(self) -> None:
        hist = pd.DataFrame(
            {
                "season": [2019, 2019, 2019],
                "pitch_type_group": ["fastball", "fastball", "fastball"],
                "rel_speed": [90.0, 91.0, 140.0],
            }
        )
        # pad other core groups so profile builder succeeds
        extra = pd.DataFrame(
            {
                "season": [2019, 2019],
                "pitch_type_group": ["breaking", "offspeed"],
                "rel_speed": [80.0, 70.0],
            }
        )
        profile = build_group_rel_speed_profile(pd.concat([hist, extra], ignore_index=True), valid_season=2022)
        means = profile_means(profile)
        self.assertAlmostEqual(means["fastball"], float(np.mean([90.0, 91.0, 140.0])))
        self.assertNotAlmostEqual(means["fastball"], float(np.median([90.0, 91.0, 140.0])))

    def test_expected_formula_and_nan_rules(self) -> None:
        means = {"fastball": 140.0, "breaking": 120.0, "offspeed": 110.0}
        frame = pd.DataFrame(
            {
                "asof_pitcher_fastball_rate": [0.5, np.nan, 0.0],
                "asof_pitcher_breaking_rate": [0.3, np.nan, 0.0],
                "asof_pitcher_offspeed_rate": [0.2, np.nan, 0.0],
            }
        )
        out = compute_tm_expected_rel_speed(frame, means)
        self.assertEqual(out.name, TM_EXPECTED_REL_SPEED)
        expected_0 = (0.5 * 140 + 0.3 * 120 + 0.2 * 110) / 1.0
        self.assertAlmostEqual(float(out.iloc[0]), expected_0)
        self.assertTrue(np.isnan(out.iloc[1]))
        self.assertTrue(np.isnan(out.iloc[2]))

    def test_profiles_are_not_reused_across_folds(self) -> None:
        tm = _synthetic_tm()
        p22 = profile_means(build_group_rel_speed_profile(filter_trackman_history(tm, 2022), valid_season=2022))
        p23 = profile_means(build_group_rel_speed_profile(filter_trackman_history(tm, 2023), valid_season=2023))
        self.assertNotEqual(p22["fastball"], p23["fastball"])

    def test_no_current_pitch_or_id_join_required(self) -> None:
        hist = filter_trackman_history(_synthetic_tm(), 2024)
        self.assertEqual(set(hist.columns), {"season", "pitch_type_group", "rel_speed"})


if __name__ == "__main__":
    unittest.main()
