"""FE-002 row-local prev1 shrinkage helper.

Replaces asof_pitcher_prev1_game_success_rate in place with

    w = n / (n + 500)
    shrunk = w * prev1 + (1 - w) * longterm

NA prev1 falls back to the same row's longterm. If both are NA the value
stays NA for the existing fold-local median impute.

This is not FE-002A (no column drop) and not FE-001 (no count_code).
No groupby, no target, no test.csv aggregates, no Trackman.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SHRINKAGE_K = 500
PREV1_COL = "asof_pitcher_prev1_game_success_rate"
LONGTERM_COL = "asof_pitcher_success_rate"
N_COL = "asof_pitcher_n"
ATTR_KEY = "fe002_prev1_shrinkage_k"


def _as_float_series(values: pd.Series, name: str, *, allow_na: bool) -> pd.Series:
    converted = pd.to_numeric(values, errors="coerce")
    uncastable = values.notna() & converted.isna()
    if bool(uncastable.any()):
        raise AssertionError(
            f"FE-002 aborted: {name} contains uncastable non-numeric values "
            f"(n={int(uncastable.sum())}). No fill was applied."
        )
    nonfinite = converted.notna() & ~np.isfinite(converted.to_numpy(dtype=float))
    if bool(nonfinite.any()):
        raise AssertionError(
            f"FE-002 aborted: {name} contains non-finite values "
            f"(n={int(nonfinite.sum())}). No fill was applied."
        )
    if not allow_na and bool(converted.isna().any()):
        raise AssertionError(
            f"FE-002 aborted: {name} contains NA (n={int(converted.isna().sum())}). "
            "w = n/(n+500) is undefined; no fill was applied."
        )
    return converted.astype(float)


def shrink_prev1_values(
    prev1: pd.Series,
    longterm: pd.Series,
    n: pd.Series,
    *,
    k: int = SHRINKAGE_K,
) -> pd.Series:
    """Row-local shrinkage. Does not use other rows or the target."""
    if k != SHRINKAGE_K:
        raise AssertionError(
            f"FE-002 aborted: shrinkage k must be frozen at {SHRINKAGE_K}, got {k}."
        )
    if len(prev1) != len(longterm) or len(prev1) != len(n):
        raise AssertionError(
            "FE-002 aborted: prev1/longterm/n length mismatch "
            f"({len(prev1)}, {len(longterm)}, {len(n)})."
        )

    n_num = _as_float_series(n, N_COL, allow_na=False)
    prev1_num = _as_float_series(prev1, PREV1_COL, allow_na=True)
    longterm_num = _as_float_series(longterm, LONGTERM_COL, allow_na=True)

    if bool((n_num < 0).any()):
        raise AssertionError(
            f"FE-002 aborted: {N_COL} contains negative values "
            f"(n={int((n_num < 0).sum())}). No clip was applied."
        )
    bad_prev1 = prev1_num.notna() & ((prev1_num < 0.0) | (prev1_num > 1.0))
    bad_long = longterm_num.notna() & ((longterm_num < 0.0) | (longterm_num > 1.0))
    if bool(bad_prev1.any()):
        raise AssertionError(
            f"FE-002 aborted: {PREV1_COL} outside [0, 1] "
            f"(n={int(bad_prev1.sum())}). No clip was applied."
        )
    if bool(bad_long.any()):
        raise AssertionError(
            f"FE-002 aborted: {LONGTERM_COL} outside [0, 1] "
            f"(n={int(bad_long.sum())}). No clip was applied."
        )

    w = n_num / (n_num + float(k))
    shrunk = w * prev1_num + (1.0 - w) * longterm_num
    # prev1 NA → same-row longterm (may still be NA).
    shrunk = shrunk.where(prev1_num.notna(), longterm_num)
    shrunk.name = PREV1_COL
    return shrunk


def replace_prev1_with_shrinkage(frame: pd.DataFrame, *, k: int = SHRINKAGE_K) -> pd.DataFrame:
    """Return a copy with PREV1_COL replaced by k=500 shrinkage values."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("replace_prev1_with_shrinkage expects a pandas DataFrame.")
    if frame.attrs.get(ATTR_KEY) is not None:
        raise AssertionError(
            "FE-002 aborted: shrinkage already applied; refusing to transform twice."
        )
    missing = [c for c in (PREV1_COL, LONGTERM_COL, N_COL) if c not in frame.columns]
    if missing:
        raise AssertionError(f"FE-002 aborted: missing columns {missing}.")
    extra = [c for c in ("count_code", "tm_expected_rel_speed", "tm_expected_ivb_by_count") if c in frame.columns]
    if extra:
        raise AssertionError(
            f"FE-002 aborted: unexpected FE-001/Trackman columns present: {extra}."
        )

    out = frame.copy()
    out[PREV1_COL] = shrink_prev1_values(out[PREV1_COL], out[LONGTERM_COL], out[N_COL], k=k)
    out.attrs[ATTR_KEY] = int(k)
    return out


def add_shrinkage_to_loaded(loaded: dict[str, Any]) -> dict[str, Any]:
    """Replace prev1 on train only. Test is left unchanged (CV does not use it)."""
    if "train" not in loaded:
        raise AssertionError("FE-002 aborted: loaded data missing 'train'.")
    out = dict(loaded)
    out["train"] = replace_prev1_with_shrinkage(loaded["train"])
    return out
