from __future__ import annotations

"""Fold-safe Trackman group profiles. No player-ID join, no current-pitch physics."""

from typing import Any

import numpy as np
import pandas as pd

CORE_PITCH_GROUPS = ("fastball", "breaking", "offspeed")
TM_EXPECTED_REL_SPEED = "tm_expected_rel_speed"
REL_SPEED_SUMMARY = "mean"
MIX_RATE_COLS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
TRACKMAN_REQUIRED_COLS = ("season", "pitch_type_group", "rel_speed")
FORBIDDEN_TM_SEASON = 2025


def allowed_trackman_seasons(valid_season: int) -> tuple[int, ...]:
    """Trackman seasons strictly before the validation year, never 2025."""
    valid_season = int(valid_season)
    years = tuple(year for year in range(2019, valid_season) if year < FORBIDDEN_TM_SEASON)
    if not years:
        raise ValueError(f"No allowed Trackman seasons for valid_season={valid_season}")
    return years


def assert_trackman_temporal_safety(
    tm: pd.DataFrame,
    valid_season: int,
    *,
    source: str = "trackman_history",
) -> None:
    if "season" not in tm.columns:
        raise AssertionError(f"{source} missing season column")
    used = {int(s) for s in pd.to_numeric(tm["season"], errors="coerce").dropna().unique()}
    allowed = set(allowed_trackman_seasons(valid_season))
    if used - allowed:
        raise AssertionError(
            f"{source} includes forbidden seasons {sorted(used - allowed)} "
            f"for valid_season={valid_season}; allowed={sorted(allowed)}"
        )
    if any(s >= int(valid_season) for s in used):
        raise AssertionError(
            f"{source} includes valid/future seasons {[s for s in sorted(used) if s >= int(valid_season)]}"
        )
    if FORBIDDEN_TM_SEASON in used:
        raise AssertionError(f"{source} includes forbidden {FORBIDDEN_TM_SEASON} Trackman")


def filter_trackman_history(tm: pd.DataFrame, valid_season: int) -> pd.DataFrame:
    missing = [c for c in TRACKMAN_REQUIRED_COLS if c not in tm.columns]
    if missing:
        raise ValueError(f"Trackman history missing required columns: {missing}")
    allowed = allowed_trackman_seasons(valid_season)
    seasons = pd.to_numeric(tm["season"], errors="coerce")
    hist = tm.loc[seasons.isin(allowed), list(TRACKMAN_REQUIRED_COLS)].copy()
    assert_trackman_temporal_safety(hist, valid_season)
    if hist.empty:
        raise AssertionError(f"No Trackman rows after cutoff season < {int(valid_season)}")
    return hist


def build_group_rel_speed_profile(
    tm_hist: pd.DataFrame,
    *,
    valid_season: int | None = None,
) -> pd.DataFrame:
    """One row per core pitch group. `other` is excluded. Summary is mean(rel_speed)."""
    if valid_season is not None:
        assert_trackman_temporal_safety(tm_hist, valid_season)
    pt = tm_hist["pitch_type_group"].astype(str).str.lower().str.strip()
    speed = pd.to_numeric(tm_hist["rel_speed"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for group in CORE_PITCH_GROUPS:
        mask = pt == group
        n = int(mask.sum())
        values = speed.loc[mask]
        n_missing = int(values.isna().sum())
        observed = values.dropna()
        if observed.empty:
            raise AssertionError(f"No observed rel_speed for group={group}")
        rows.append(
            {
                "group": group,
                "n": n,
                "n_missing": n_missing,
                "missing_rate": float(n_missing / n) if n else float("nan"),
                "rel_speed_mean": float(observed.mean()),
                "rel_speed_std": float(observed.std(ddof=0)),
                "summary": REL_SPEED_SUMMARY,
            }
        )
    profile = pd.DataFrame(rows)
    if set(profile["group"]) != set(CORE_PITCH_GROUPS):
        raise AssertionError(f"Profile groups {list(profile['group'])} != {CORE_PITCH_GROUPS}")
    if "other" in set(profile["group"]):
        raise AssertionError("TM-002A profile must not include 'other'")
    return profile


def profile_means(profile: pd.DataFrame) -> dict[str, float]:
    return {str(row.group): float(row.rel_speed_mean) for row in profile.itertuples()}


def compute_tm_expected_rel_speed(
    frame: pd.DataFrame,
    means: dict[str, float],
    *,
    fastball_col: str = MIX_RATE_COLS[0],
    breaking_col: str = MIX_RATE_COLS[1],
    offspeed_col: str = MIX_RATE_COLS[2],
) -> pd.Series:
    """Weighted mean of group rel_speed using as-of mix rates. NaN if denom==0 or mix NA."""
    missing_groups = [g for g in CORE_PITCH_GROUPS if g not in means]
    if missing_groups:
        raise KeyError(f"Profile missing groups: {missing_groups}")
    r_fastball = pd.to_numeric(frame[fastball_col], errors="coerce")
    r_breaking = pd.to_numeric(frame[breaking_col], errors="coerce")
    r_offspeed = pd.to_numeric(frame[offspeed_col], errors="coerce")
    numerator = (
        r_fastball * float(means["fastball"])
        + r_breaking * float(means["breaking"])
        + r_offspeed * float(means["offspeed"])
    )
    denominator = r_fastball + r_breaking + r_offspeed
    expected = numerator / denominator
    expected = expected.where(denominator > 0)
    expected = expected.where(np.isfinite(expected.to_numpy(dtype=float)))
    expected.name = TM_EXPECTED_REL_SPEED
    return expected


# --- TM-003A: count-conditioned expected IVB (row-level prior-history) ---

MIN_CELL_N = 1000
TM_EXPECTED_IVB_BY_COUNT = "tm_expected_ivb_by_count"
TM_EXPECTED_IVB_GLOBAL = "tm_expected_ivb_global"
IVB_COL = "induced_vert_break"
PITCHMIX_N_COL = "asof_pitcher_pitchmix_n"
LEGAL_BALLS = (0, 1, 2, 3)
LEGAL_STRIKES = (0, 1, 2)
LEGAL_COUNTS = tuple(f"{balls}-{strikes}" for balls in LEGAL_BALLS for strikes in LEGAL_STRIKES)
TRACKMAN_IVB_REQUIRED_COLS = (
    "season",
    "pitch_type_group",
    "balls_before",
    "strikes_before",
    "induced_vert_break",
)


def prior_history_seasons(row_season: int) -> tuple[int, ...]:
    """Trackman seasons strictly before the row's season. Empty for 2019; never 2025."""
    row_season = int(row_season)
    if row_season <= 2019:
        return ()
    return allowed_trackman_seasons(row_season)


def make_count_code(balls: pd.Series, strikes: pd.Series) -> pd.Series:
    """Legal `balls-strikes` codes only; illegal/missing counts become NA."""
    balls_n = pd.to_numeric(balls, errors="coerce")
    strikes_n = pd.to_numeric(strikes, errors="coerce")
    legal = balls_n.isin(LEGAL_BALLS) & strikes_n.isin(LEGAL_STRIKES)
    codes = (
        balls_n.astype("Int64").astype(str)
        + "-"
        + strikes_n.astype("Int64").astype(str)
    )
    out = pd.Series(pd.NA, index=balls.index, dtype="object")
    out.loc[legal] = codes.loc[legal]
    return out


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values.to_numpy(dtype=float)))


def prepare_trackman_ivb(tm: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in TRACKMAN_IVB_REQUIRED_COLS if c not in tm.columns]
    if missing:
        raise ValueError(f"Trackman IVB history missing required columns: {missing}")
    out = tm.loc[:, list(TRACKMAN_IVB_REQUIRED_COLS)].copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["pitch_type_group"] = out["pitch_type_group"].astype(str).str.lower().str.strip()
    out["balls_before"] = pd.to_numeric(out["balls_before"], errors="coerce")
    out["strikes_before"] = pd.to_numeric(out["strikes_before"], errors="coerce")
    out[IVB_COL] = _finite_numeric(out[IVB_COL])
    out["count"] = make_count_code(out["balls_before"], out["strikes_before"])
    return out


def assert_row_level_trackman_safety(
    tm_hist: pd.DataFrame,
    row_season: int,
    *,
    source: str = "trackman_prior_history",
) -> None:
    """Every used Trackman season must be strictly before the main row season."""
    row_season = int(row_season)
    if tm_hist.empty:
        return
    if "season" not in tm_hist.columns:
        raise AssertionError(f"{source} missing season column")
    used = {int(s) for s in pd.to_numeric(tm_hist["season"], errors="coerce").dropna().unique()}
    allowed = set(prior_history_seasons(row_season))
    if used - allowed:
        raise AssertionError(
            f"{source} includes forbidden seasons {sorted(used - allowed)} "
            f"for row_season={row_season}; allowed={sorted(allowed)}"
        )
    if any(s >= row_season for s in used):
        raise AssertionError(
            f"{source} includes own/future seasons {[s for s in sorted(used) if s >= row_season]} "
            f"for row_season={row_season}"
        )
    if FORBIDDEN_TM_SEASON in used:
        raise AssertionError(f"{source} includes forbidden {FORBIDDEN_TM_SEASON} Trackman")
    if row_season > 2019:
        assert_trackman_temporal_safety(tm_hist, row_season, source=source)


def filter_trackman_prior_history_ivb(tm: pd.DataFrame, row_season: int) -> pd.DataFrame:
    prepared = prepare_trackman_ivb(tm) if "count" not in tm.columns else tm
    allowed = prior_history_seasons(row_season)
    if not allowed:
        hist = prepared.iloc[0:0].copy()
        assert_row_level_trackman_safety(hist, row_season)
        return hist
    seasons = pd.to_numeric(prepared["season"], errors="coerce")
    hist = prepared.loc[seasons.isin(allowed)].copy()
    assert_row_level_trackman_safety(hist, row_season)
    return hist


def _group_ivb_stats(values: pd.Series) -> dict[str, float]:
    """Audit cell-n rule: n is the full cell row count; mean uses finite IVB only."""
    n = int(len(values))
    observed = _finite_numeric(values).dropna()
    n_missing = n - int(len(observed))
    mean = float(observed.mean()) if not observed.empty else float("nan")
    return {
        "n": n,
        "n_missing": n_missing,
        "ivb_mean": mean,
    }


def build_season_ivb_profile(
    tm: pd.DataFrame,
    row_season: int,
    *,
    min_n: int = MIN_CELL_N,
) -> dict[str, Any]:
    """Count×group and count-agnostic IVB profile for one main-row season."""
    if int(min_n) != MIN_CELL_N:
        raise ValueError(f"TM-003A min_n is frozen at {MIN_CELL_N}; got {min_n}")
    hist = filter_trackman_prior_history_ivb(tm, row_season)
    used_seasons = tuple(
        sorted(int(s) for s in pd.to_numeric(hist["season"], errors="coerce").dropna().unique())
    )
    cell_n: dict[tuple[str, str], int] = {}
    cell_mean: dict[tuple[str, str], float] = {}
    global_n: dict[str, int] = {}
    global_mean: dict[str, float] = {}
    global_n_missing: dict[str, int] = {}

    if not hist.empty:
        pt = hist["pitch_type_group"].astype(str)
        for group in CORE_PITCH_GROUPS:
            gmask = pt == group
            gvals = hist.loc[gmask, IVB_COL]
            gstats = _group_ivb_stats(gvals)
            global_n[group] = int(gstats["n"])
            global_n_missing[group] = int(gstats["n_missing"])
            global_mean[group] = float(gstats["ivb_mean"])
            sub = hist.loc[gmask & hist["count"].notna(), ["count", IVB_COL]]
            if sub.empty:
                continue
            for count_value, series in sub.groupby("count", dropna=True)[IVB_COL]:
                stats = _group_ivb_stats(series)
                key = (str(count_value), group)
                cell_n[key] = int(stats["n"])
                cell_mean[key] = float(stats["ivb_mean"])

    for group in CORE_PITCH_GROUPS:
        global_n.setdefault(group, 0)
        global_n_missing.setdefault(group, 0)
        global_mean.setdefault(group, float("nan"))

    return {
        "row_season": int(row_season),
        "tm_seasons": used_seasons,
        "min_n": int(MIN_CELL_N),
        "cell_n": cell_n,
        "cell_mean": cell_mean,
        "global_n": global_n,
        "global_n_missing": global_n_missing,
        "global_mean": global_mean,
    }


def resolve_group_ivb(
    profile: dict[str, Any],
    count: str | None,
    group: str,
) -> tuple[float, bool]:
    """Return (ivb, used_fallback) for one count×group under frozen min_n=1000."""
    if group not in CORE_PITCH_GROUPS:
        raise KeyError(f"Unsupported pitch group: {group}")
    global_mean = float(profile["global_mean"].get(group, float("nan")))
    if count is None or (isinstance(count, float) and not np.isfinite(count)) or pd.isna(count):
        cell_n = 0
        cell_mean = float("nan")
    else:
        key = (str(count), group)
        cell_n = int(profile["cell_n"].get(key, 0))
        cell_mean = float(profile["cell_mean"].get(key, float("nan")))
    if cell_n >= MIN_CELL_N and np.isfinite(cell_mean):
        return cell_mean, False
    return global_mean, True


def _constructible_mask(frame: pd.DataFrame) -> pd.Series:
    mix_n = pd.to_numeric(frame[PITCHMIX_N_COL], errors="coerce")
    rates = [_finite_numeric(frame[col]) for col in MIX_RATE_COLS]
    ok = mix_n > 0
    for rate in rates:
        ok = ok & rate.notna()
    return ok.fillna(False)


def compute_tm003a_features(
    frame: pd.DataFrame,
    tm: pd.DataFrame,
    *,
    profile_cache: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Row-level TM-003A feature plus diagnostics. Index matches `frame`.

    Model feature: `tm_expected_ivb_by_count` (plain mix-weighted sum).
    `tm_expected_ivb_global` is diagnostic-only and must not be trained on.
    """
    required = ["season", "balls_before", "strikes_before", PITCHMIX_N_COL, *MIX_RATE_COLS]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Main frame missing TM-003A columns: {missing}")
    if profile_cache is None:
        profile_cache = {}

    prepared_tm = prepare_trackman_ivb(tm)
    out = pd.DataFrame(index=frame.index)
    out[TM_EXPECTED_IVB_BY_COUNT] = np.nan
    out[TM_EXPECTED_IVB_GLOBAL] = np.nan
    out["tm003a_constructible"] = False
    for group in CORE_PITCH_GROUPS:
        out[f"tm003a_fallback_{group}"] = False
    out["tm003a_fallback_any"] = False
    out["tm003a_fallback_weight_share"] = np.nan
    out["tm003a_max_tm_season_used"] = np.nan
    out["tm003a_n_tm_rows"] = 0

    seasons = pd.to_numeric(frame["season"], errors="coerce")
    counts = make_count_code(frame["balls_before"], frame["strikes_before"])
    constructible = _constructible_mask(frame)
    rates = {g: _finite_numeric(frame[col]) for g, col in zip(CORE_PITCH_GROUPS, MIX_RATE_COLS)}

    unique_seasons = sorted({int(s) for s in seasons.dropna().unique()})
    for row_season in unique_seasons:
        if row_season not in profile_cache:
            profile_cache[row_season] = build_season_ivb_profile(prepared_tm, row_season)
        profile = profile_cache[row_season]
        mask = seasons == row_season
        idx = frame.index[mask]
        tm_seasons = profile["tm_seasons"]
        max_tm = max(tm_seasons) if tm_seasons else np.nan
        if tm_seasons:
            if max(tm_seasons) >= row_season:
                raise AssertionError(
                    f"row-level TM leak: row_season={row_season} used {tm_seasons}"
                )
            if FORBIDDEN_TM_SEASON in tm_seasons:
                raise AssertionError("2025 Trackman entered a TM-003A profile")
        out.loc[idx, "tm003a_max_tm_season_used"] = max_tm
        out.loc[idx, "tm003a_n_tm_rows"] = int(sum(profile["global_n"].values()))

        count_slice = counts.loc[idx]
        rate_slice = {g: rates[g].loc[idx] for g in CORE_PITCH_GROUPS}
        cons_slice = constructible.loc[idx]

        fallback_flags: dict[str, pd.Series] = {}
        cell_ivb: dict[str, pd.Series] = {}
        global_ivb_value = {g: float(profile["global_mean"][g]) for g in CORE_PITCH_GROUPS}
        na_count = None
        for group in CORE_PITCH_GROUPS:
            ivb_lookup = {}
            fb_lookup = {}
            for count_value in LEGAL_COUNTS:
                value, used_fb = resolve_group_ivb(profile, count_value, group)
                ivb_lookup[count_value] = value
                fb_lookup[count_value] = bool(used_fb)
            na_value, na_fb = resolve_group_ivb(profile, na_count, group)
            ivb_series = pd.Series(na_value, index=idx, dtype=float)
            fb_series = pd.Series(na_fb, index=idx, dtype=bool)
            for count_value in LEGAL_COUNTS:
                sel = count_slice == count_value
                if bool(sel.any()):
                    ivb_series.loc[sel] = ivb_lookup[count_value]
                    fb_series.loc[sel] = fb_lookup[count_value]
            cell_ivb[group] = ivb_series
            fallback_flags[group] = fb_series

        by_count = (
            rate_slice["fastball"] * cell_ivb["fastball"]
            + rate_slice["breaking"] * cell_ivb["breaking"]
            + rate_slice["offspeed"] * cell_ivb["offspeed"]
        )
        global_feat = (
            rate_slice["fastball"] * global_ivb_value["fastball"]
            + rate_slice["breaking"] * global_ivb_value["breaking"]
            + rate_slice["offspeed"] * global_ivb_value["offspeed"]
        )
        by_count = by_count.where(cons_slice)
        global_feat = global_feat.where(cons_slice)
        by_count = by_count.where(np.isfinite(by_count.to_numpy(dtype=float)))
        global_feat = global_feat.where(np.isfinite(global_feat.to_numpy(dtype=float)))

        out.loc[idx, TM_EXPECTED_IVB_BY_COUNT] = by_count
        out.loc[idx, TM_EXPECTED_IVB_GLOBAL] = global_feat
        out.loc[idx, "tm003a_constructible"] = cons_slice.astype(bool)
        weight = pd.Series(0.0, index=idx)
        any_fb = pd.Series(False, index=idx)
        for group in CORE_PITCH_GROUPS:
            fb = fallback_flags[group] & cons_slice
            out.loc[idx, f"tm003a_fallback_{group}"] = fb
            any_fb = any_fb | fb
            weight = weight + rate_slice[group].fillna(0.0) * fb.astype(float)
        out.loc[idx, "tm003a_fallback_any"] = any_fb
        out.loc[idx, "tm003a_fallback_weight_share"] = weight.where(cons_slice)

    row_seasons = pd.to_numeric(frame["season"], errors="coerce")
    max_used = pd.to_numeric(out["tm003a_max_tm_season_used"], errors="coerce")
    leaked = max_used.notna() & row_seasons.notna() & (max_used >= row_seasons)
    if bool(leaked.any()):
        bad_seasons = sorted({int(s) for s in row_seasons.loc[leaked].unique()})
        raise AssertionError(
            f"TM-003A row-level leakage: max Trackman season >= row season for seasons {bad_seasons}"
        )
    if len(out) != len(frame):
        raise AssertionError("TM-003A feature row count changed")
    if not out.index.equals(frame.index):
        raise AssertionError("TM-003A feature index alignment changed")
    return out


def summarize_cell_n(profile: dict[str, Any]) -> dict[str, float]:
    ns = [int(v) for v in profile["cell_n"].values()]
    if not ns:
        return {"min": float("nan"), "p10": float("nan"), "median": float("nan"), "n_cells": 0}
    arr = np.asarray(ns, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.median(arr)),
        "n_cells": int(len(arr)),
    }
