"""FE-002 prev1 shrinkage construction: k=500, in-place, abort on invalid."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.fe002_prev1_shrinkage import (
    ATTR_KEY,
    LONGTERM_COL,
    N_COL,
    PREV1_COL,
    SHRINKAGE_K,
    add_shrinkage_to_loaded,
    replace_prev1_with_shrinkage,
    shrink_prev1_values,
)


def _frame(prev1, longterm, n, extra=None) -> pd.DataFrame:
    data = {PREV1_COL: prev1, LONGTERM_COL: longterm, N_COL: n}
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


class Fe002Prev1ShrinkageTests(unittest.TestCase):
    def test_k_is_frozen_at_500(self) -> None:
        self.assertEqual(SHRINKAGE_K, 500)

    def test_n500_is_equal_mix(self) -> None:
        out = shrink_prev1_values(pd.Series([0.2]), pd.Series([0.8]), pd.Series([500.0]))
        self.assertAlmostEqual(float(out.iloc[0]), 0.5)

    def test_n0_uses_longterm(self) -> None:
        out = shrink_prev1_values(pd.Series([0.2]), pd.Series([0.8]), pd.Series([0.0]))
        self.assertAlmostEqual(float(out.iloc[0]), 0.8)

    def test_n1500_weight_is_075(self) -> None:
        out = shrink_prev1_values(pd.Series([0.2]), pd.Series([0.8]), pd.Series([1500.0]))
        self.assertAlmostEqual(float(out.iloc[0]), 0.75 * 0.2 + 0.25 * 0.8)

    def test_prev1_na_falls_back_to_longterm(self) -> None:
        out = shrink_prev1_values(pd.Series([np.nan]), pd.Series([0.6]), pd.Series([100.0]))
        self.assertAlmostEqual(float(out.iloc[0]), 0.6)

    def test_both_na_stay_na(self) -> None:
        out = shrink_prev1_values(pd.Series([np.nan]), pd.Series([np.nan]), pd.Series([0.0]))
        self.assertTrue(pd.isna(out.iloc[0]))

    def test_column_name_is_unchanged(self) -> None:
        frame = _frame([0.2], [0.8], [500.0], extra={"asof_pitcher_prev3_game_success_rate": [0.7]})
        out = replace_prev1_with_shrinkage(frame)
        self.assertIn(PREV1_COL, out.columns)
        self.assertEqual(list(out.columns), list(frame.columns))
        self.assertAlmostEqual(float(frame[PREV1_COL].iloc[0]), 0.2)
        self.assertAlmostEqual(float(out[PREV1_COL].iloc[0]), 0.5)
        self.assertEqual(out.attrs[ATTR_KEY], 500)

    def test_prev3_prev5_unchanged(self) -> None:
        frame = _frame(
            [0.1, 0.9],
            [0.5, 0.5],
            [500.0, 500.0],
            extra={
                "asof_pitcher_prev3_game_success_rate": [0.11, 0.22],
                "asof_pitcher_prev5_game_success_rate": [0.33, 0.44],
            },
        )
        out = replace_prev1_with_shrinkage(frame)
        pd.testing.assert_series_equal(
            out["asof_pitcher_prev3_game_success_rate"],
            frame["asof_pitcher_prev3_game_success_rate"],
        )
        pd.testing.assert_series_equal(
            out["asof_pitcher_prev5_game_success_rate"],
            frame["asof_pitcher_prev5_game_success_rate"],
        )

    def test_row_local_matches_single_row(self) -> None:
        frame = _frame([0.1, 0.9, 0.3], [0.5, 0.4, 0.6], [100.0, 500.0, 1500.0])
        out = replace_prev1_with_shrinkage(frame)
        for i in range(len(frame)):
            one = replace_prev1_with_shrinkage(frame.iloc[[i]].reset_index(drop=True))
            self.assertAlmostEqual(float(out[PREV1_COL].iloc[i]), float(one[PREV1_COL].iloc[0]))

    def test_target_permutation_does_not_change_shrunk(self) -> None:
        frame = _frame([0.1, 0.9], [0.5, 0.4], [200.0, 800.0], extra={"control_success": [0, 1]})
        shuffled = frame.copy()
        shuffled["control_success"] = [1, 0]
        a = replace_prev1_with_shrinkage(frame)[PREV1_COL]
        b = replace_prev1_with_shrinkage(shuffled)[PREV1_COL]
        pd.testing.assert_series_equal(a, b)

    def test_other_k_aborts(self) -> None:
        with self.assertRaisesRegex(AssertionError, "k must be frozen"):
            shrink_prev1_values(pd.Series([0.2]), pd.Series([0.8]), pd.Series([500.0]), k=200)

    def test_negative_n_aborts(self) -> None:
        with self.assertRaisesRegex(AssertionError, "negative"):
            shrink_prev1_values(pd.Series([0.2]), pd.Series([0.8]), pd.Series([-1.0]))

    def test_n_na_aborts(self) -> None:
        with self.assertRaisesRegex(AssertionError, "contains NA"):
            shrink_prev1_values(pd.Series([0.2]), pd.Series([0.8]), pd.Series([np.nan]))

    def test_rate_outside_unit_interval_aborts(self) -> None:
        with self.assertRaisesRegex(AssertionError, "outside \\[0, 1\\]"):
            shrink_prev1_values(pd.Series([1.2]), pd.Series([0.8]), pd.Series([10.0]))
        with self.assertRaisesRegex(AssertionError, "outside \\[0, 1\\]"):
            shrink_prev1_values(pd.Series([0.2]), pd.Series([-0.1]), pd.Series([10.0]))

    def test_non_finite_aborts(self) -> None:
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            shrink_prev1_values(pd.Series([np.inf]), pd.Series([0.8]), pd.Series([10.0]))

    def test_uncastable_aborts(self) -> None:
        with self.assertRaisesRegex(AssertionError, "uncastable"):
            shrink_prev1_values(pd.Series(["bad"]), pd.Series([0.8]), pd.Series([10.0]))

    def test_refuses_second_apply(self) -> None:
        frame = _frame([0.2], [0.8], [500.0])
        once = replace_prev1_with_shrinkage(frame)
        with self.assertRaisesRegex(AssertionError, "already applied"):
            replace_prev1_with_shrinkage(once)

    def test_refuses_count_code_or_trackman_columns(self) -> None:
        with self.assertRaisesRegex(AssertionError, "unexpected"):
            replace_prev1_with_shrinkage(_frame([0.2], [0.8], [500.0], extra={"count_code": ["0-2"]}))
        with self.assertRaisesRegex(AssertionError, "unexpected"):
            replace_prev1_with_shrinkage(
                _frame([0.2], [0.8], [500.0], extra={"tm_expected_rel_speed": [90.0]})
            )

    def test_loaded_wrapper_transforms_train_only(self) -> None:
        loaded = {
            "train": _frame([0.2], [0.8], [500.0]),
            "test": _frame([0.2], [0.8], [500.0]),
        }
        out = add_shrinkage_to_loaded(loaded)
        self.assertAlmostEqual(float(out["train"][PREV1_COL].iloc[0]), 0.5)
        self.assertAlmostEqual(float(out["test"][PREV1_COL].iloc[0]), 0.2)
        self.assertAlmostEqual(float(loaded["train"][PREV1_COL].iloc[0]), 0.2)


if __name__ == "__main__":
    unittest.main()
