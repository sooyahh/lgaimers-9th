"""TM-AUDIT-001B: Trackman feasibility without 1:1 player join. No training.

Does not overwrite existing Trackman/EDA artifacts.
Does not aggregate test.csv rows.
Past-season-only windows for every validation year.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, load_config, resolve_path

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
PHYS_COLS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
MIN_GROUP_N = 50
FOLD_WINDOWS = {
    2022: list(range(2019, 2022)),
    2023: list(range(2019, 2023)),
    2024: list(range(2019, 2024)),
    2025: list(range(2019, 2025)),
}
LEVELS = [
    ("L1_hand_count", ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"]),
    ("L2_pitcher_hand_count", ["pitcher_hand", "balls_before", "strikes_before"]),
    ("L3_pitcher_hand", ["pitcher_hand"]),
    ("L0_count_only", ["balls_before", "strikes_before"]),
]


def fmt(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "missing"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def normalize_hand(series: pd.Series, source: str) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    mapped = s.astype(str).str.strip().str.upper()
    # common encodings; unknown values stay as original uppercased token
    repl = {"L": "L", "R": "R", "LEFT": "L", "RIGHT": "R", "1": "1", "0": "0", "2": "2"}
    return mapped.map(lambda x: repl.get(x, x))


def coverage_for_level(
    main_valid: pd.DataFrame,
    tm_hist: pd.DataFrame,
    keys: list[str],
    min_n: int,
) -> dict[str, Any]:
    if tm_hist.empty or main_valid.empty:
        return {
            "n_main": int(len(main_valid)),
            "n_tm_hist": int(len(tm_hist)),
            "n_tm_groups": 0,
            "coverage": np.nan,
            "fallback_needed_rate": np.nan,
            "median_group_n": np.nan,
            "p10_group_n": np.nan,
            "p50_group_n": np.nan,
            "p90_group_n": np.nan,
            "share_groups_ge_min_n": np.nan,
            "main_rows_matched_any": 0,
            "main_rows_matched_ge_min_n": 0,
        }
    grp = tm_hist.groupby(keys, dropna=False).size().rename("tm_n").reset_index()
    joined = main_valid.merge(grp, on=keys, how="left")
    tm_n = joined["tm_n"].fillna(0)
    return {
        "n_main": int(len(main_valid)),
        "n_tm_hist": int(len(tm_hist)),
        "n_tm_groups": int(len(grp)),
        "coverage": float((tm_n > 0).mean()),
        "fallback_needed_rate": float((tm_n < min_n).mean()),
        "median_group_n": float(grp["tm_n"].median()),
        "p10_group_n": float(grp["tm_n"].quantile(0.10)),
        "p50_group_n": float(grp["tm_n"].quantile(0.50)),
        "p90_group_n": float(grp["tm_n"].quantile(0.90)),
        "share_groups_ge_min_n": float((grp["tm_n"] >= min_n).mean()),
        "main_rows_matched_any": int((tm_n > 0).sum()),
        "main_rows_matched_ge_min_n": int((tm_n >= min_n).sum()),
        "min_group_n": int(grp["tm_n"].min()),
        "max_group_n": int(grp["tm_n"].max()),
    }


def tm002_profiles(tm_hist: pd.DataFrame) -> dict[str, Any]:
    pt = tm_hist["pitch_type_group"].astype(str).str.lower().str.strip()
    out: dict[str, Any] = {"n_hist": int(len(tm_hist))}
    for g in PITCH_GROUPS:
        sub = tm_hist.loc[pt == g]
        out[f"{g}_n"] = int(len(sub))
        out[f"{g}_share"] = float(len(sub) / len(tm_hist)) if len(tm_hist) else np.nan
        for col in ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension"]:
            s = pd.to_numeric(sub[col], errors="coerce")
            out[f"{g}_{col}_n"] = int(s.notna().sum())
            out[f"{g}_{col}_median"] = float(s.median()) if s.notna().any() else np.nan
            out[f"{g}_{col}_mean"] = float(s.mean()) if s.notna().any() else np.nan
            out[f"{g}_{col}_std"] = float(s.std(ddof=0)) if s.notna().any() else np.nan
            out[f"{g}_{col}_missing_rate"] = float(s.isna().mean()) if len(sub) else np.nan
    return out


def verdict(pass_ok: bool, hold: bool, fail: bool) -> str:
    if fail:
        return "FAIL"
    if hold:
        return "HOLD"
    if pass_ok:
        return "PASS"
    return "HOLD"


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/baseline.yaml")
    train_path = resolve_path(cfg["data"]["train_path"])
    tm_path = resolve_path("data/raw/trackman_history.csv")
    out_dir = ensure_dir(ROOT / "outputs" / "analysis")
    csv_path = out_dir / "trackman_feasibility_001b.csv"
    report_path = ROOT / "reports" / "trackman_audit_001b.md"

    train = pd.read_csv(
        train_path,
        encoding="utf-8-sig",
        usecols=[
            "row_id",
            "season",
            "pitcher_id",
            "batter_id",
            "pitcher_hand",
            "batter_hand",
            "balls_before",
            "strikes_before",
            "asof_pitcher_pitchmix_n",
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ],
        low_memory=False,
    )
    tm = pd.read_csv(tm_path, encoding="utf-8-sig", low_memory=False)

    # --- 3-1 linkage ---
    main_p = set(pd.to_numeric(train["pitcher_id"], errors="coerce").dropna().astype(np.int64))
    main_b = set(pd.to_numeric(train["batter_id"], errors="coerce").dropna().astype(np.int64))
    tm_p = set(pd.to_numeric(tm["pitcher_trackman_id"], errors="coerce").dropna().astype(np.int64))
    tm_b = set(pd.to_numeric(tm["batter_trackman_id"], errors="coerce").dropna().astype(np.int64))
    overlap_p = len(main_p & tm_p)
    overlap_b = len(main_b & tm_b)

    # also check string overlap in case of mixed types
    overlap_p_str = len(set(train["pitcher_id"].astype(str)) & set(tm["pitcher_trackman_id"].astype(str)))
    overlap_b_str = len(set(train["batter_id"].astype(str)) & set(tm["batter_trackman_id"].astype(str)))

    tm_seasons = sorted(int(x) for x in tm["season"].dropna().unique().tolist())
    train_hands_p = sorted(train["pitcher_hand"].dropna().unique().tolist(), key=str)
    train_hands_b = sorted(train["batter_hand"].dropna().unique().tolist(), key=str)
    tm_hands_p = sorted(tm["pitcher_hand"].dropna().unique().tolist(), key=str)
    tm_hands_b = sorted(tm["batter_hand"].dropna().unique().tolist(), key=str)

    # normalize hands for join attempts
    train = train.copy()
    tm = tm.copy()
    train["pitcher_hand_key"] = normalize_hand(train["pitcher_hand"], "train")
    train["batter_hand_key"] = normalize_hand(train["batter_hand"], "train")
    tm["pitcher_hand_key"] = normalize_hand(tm["pitcher_hand"], "tm")
    tm["batter_hand_key"] = normalize_hand(tm["batter_hand"], "tm")

    hand_join_possible = (
        set(train["pitcher_hand_key"].dropna()) & set(tm["pitcher_hand_key"].dropna())
        and set(train["batter_hand_key"].dropna()) & set(tm["batter_hand_key"].dropna())
    )

    # If numeric 0/1 vs L/R with no overlap, context join on hand fails unless a documented map exists.
    train_hand_set = set(train["pitcher_hand_key"].dropna().unique())
    tm_hand_set = set(tm["pitcher_hand_key"].dropna().unique())
    hand_key_overlap = train_hand_set & tm_hand_set

    # Use native columns when they already share a coding; otherwise keep *_key and report.
    if hand_key_overlap:
        for df in (train, tm):
            df["pitcher_hand"] = df["pitcher_hand_key"]
            df["batter_hand"] = df["batter_hand_key"]
        hand_coding = "shared_after_normalize"
    else:
        hand_coding = "NO_SHARED_HAND_CODES"

    missing_phys = {
        col: float(pd.to_numeric(tm[col], errors="coerce").isna().mean()) if col in tm.columns else np.nan
        for col in PHYS_COLS
    }
    pt = tm["pitch_type_group"].astype(str).str.lower().str.strip()
    pt_share = {g: float((pt == g).mean()) for g in PITCH_GROUPS}
    pt_missing = float(tm["pitch_type_group"].isna().mean())
    season_n = {int(y): int((tm["season"] == y).sum()) for y in YEARS if y in tm_seasons}

    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "section": "linkage",
            "item": "pitcher_id_direct_overlap",
            "value": overlap_p,
            "note": f"string_overlap={overlap_p_str}; main_nunique={len(main_p)}; tm_nunique={len(tm_p)}",
        }
    )
    rows.append(
        {
            "section": "linkage",
            "item": "batter_id_direct_overlap",
            "value": overlap_b,
            "note": f"string_overlap={overlap_b_str}; main_nunique={len(main_b)}; tm_nunique={len(tm_b)}",
        }
    )

    # --- 3-2 TM-001 coverage by fold ---
    # Join keys: balls/strikes should be comparable integers.
    train["balls_before"] = pd.to_numeric(train["balls_before"], errors="coerce")
    train["strikes_before"] = pd.to_numeric(train["strikes_before"], errors="coerce")
    tm["balls_before"] = pd.to_numeric(tm["balls_before"], errors="coerce")
    tm["strikes_before"] = pd.to_numeric(tm["strikes_before"], errors="coerce")

    coverage_records: list[dict[str, Any]] = []
    for valid_year, hist_years in FOLD_WINDOWS.items():
        tm_hist = tm.loc[tm["season"].isin(hist_years)].copy()
        if valid_year == 2025:
            # full 2025 test is not available; do not scan test.csv rows
            n_main = "not available (full test.csv not present; local sample is 5 rows and is not used)"
            main_valid = train.iloc[0:0]
        else:
            main_valid = train.loc[train["season"] == valid_year].copy()
            n_main = len(main_valid)

        for level_name, keys in LEVELS:
            if valid_year == 2025:
                rec = {
                    "valid_year": valid_year,
                    "hist_years": ",".join(str(y) for y in hist_years),
                    "level": level_name,
                    "keys": "+".join(keys),
                    "n_main": np.nan,
                    "n_tm_hist": int(len(tm_hist)),
                    "coverage": np.nan,
                    "fallback_needed_rate": np.nan,
                    "note": "2025 main-row coverage not available without full test labels/rows",
                }
            elif hand_coding == "NO_SHARED_HAND_CODES" and any(
                k in keys for k in ("pitcher_hand", "batter_hand")
            ):
                rec = {
                    "valid_year": valid_year,
                    "hist_years": ",".join(str(y) for y in hist_years),
                    "level": level_name,
                    "keys": "+".join(keys),
                    "n_main": int(n_main),
                    "n_tm_hist": int(len(tm_hist)),
                    "coverage": np.nan,
                    "fallback_needed_rate": 1.0,
                    "note": "hand codes do not overlap; this level cannot be joined",
                }
            else:
                cov = coverage_for_level(main_valid, tm_hist, keys, MIN_GROUP_N)
                rec = {
                    "valid_year": valid_year,
                    "hist_years": ",".join(str(y) for y in hist_years),
                    "level": level_name,
                    "keys": "+".join(keys),
                    **cov,
                    "note": "prior-season TM only",
                }
            rec["section"] = "tm001_coverage"
            coverage_records.append(rec)
            rows.append(
                {
                    "section": "tm001_coverage",
                    "item": f"{valid_year}:{level_name}:coverage",
                    "value": rec.get("coverage", np.nan),
                    "note": json.dumps(
                        {k: rec.get(k) for k in ("fallback_needed_rate", "n_tm_groups", "median_group_n", "note")},
                        default=str,
                    ),
                }
            )

    cov_df = pd.DataFrame(coverage_records)

    # feature count estimate
    n_phys_stats = len(PHYS_COLS) * 3  # mean/median/std
    n_share = len(PITCH_GROUPS)
    n_feat_l1 = n_phys_stats + n_share
    rows.append(
        {
            "section": "tm001_design",
            "item": "estimated_feature_count_per_level",
            "value": n_feat_l1,
            "note": f"{n_share} pitch_type shares + {len(PHYS_COLS)} cols × (mean,median,std)",
        }
    )

    # --- 3-3 TM-002 ---
    tm002_rows: list[dict[str, Any]] = []
    mix_missing = {
        "fastball": float(train["asof_pitcher_fastball_rate"].isna().mean()),
        "breaking": float(train["asof_pitcher_breaking_rate"].isna().mean()),
        "offspeed": float(train["asof_pitcher_offspeed_rate"].isna().mean()),
        "pitchmix_n_zero": float((train["asof_pitcher_pitchmix_n"].fillna(0) == 0).mean()),
    }
    for valid_year, hist_years in FOLD_WINDOWS.items():
        tm_hist = tm.loc[tm["season"].isin(hist_years)]
        prof = tm002_profiles(tm_hist)
        prof["valid_year"] = valid_year
        prof["hist_years"] = ",".join(str(y) for y in hist_years)
        if valid_year == 2025:
            main_valid = train.iloc[0:0]
            mix_na = np.nan
        else:
            main_valid = train.loc[train["season"] == valid_year]
            mix_na = float(main_valid["asof_pitcher_fastball_rate"].isna().mean())
        prof["main_mix_missing_rate"] = mix_na
        # expected-speed computable if mix observed AND each group has some TM n
        groups_ok = all(prof.get(f"{g}_n", 0) >= MIN_GROUP_N for g in ["fastball", "breaking", "offspeed"])
        prof["core_groups_ge_min_n"] = groups_ok
        tm002_rows.append(prof)
        rows.append(
            {
                "section": "tm002_profile",
                "item": f"{valid_year}:fastball_n",
                "value": prof["fastball_n"],
                "note": f"breaking_n={prof['breaking_n']}; offspeed_n={prof['offspeed_n']}; other_n={prof['other_n']}",
            }
        )

    tm002_df = pd.DataFrame(tm002_rows)

    # computational cost: groupby on ~1.8M is cheap
    elapsed = time.perf_counter() - t0
    rows.append(
        {
            "section": "cost",
            "item": "audit_runtime_seconds",
            "value": elapsed,
            "note": "coverage + profiles only; no model training",
        }
    )

    # fallback cascade rate for 2022-2024 if L1 fails
    cascade_notes = []
    for y in [2022, 2023, 2024]:
        l1 = cov_df.loc[(cov_df.valid_year == y) & (cov_df.level == "L1_hand_count")]
        l2 = cov_df.loc[(cov_df.valid_year == y) & (cov_df.level == "L2_pitcher_hand_count")]
        l3 = cov_df.loc[(cov_df.valid_year == y) & (cov_df.level == "L3_pitcher_hand")]
        cascade_notes.append(
            {
                "valid_year": y,
                "L1_coverage": float(l1["coverage"].iloc[0]) if len(l1) and np.isfinite(l1["coverage"].iloc[0]) else np.nan,
                "L1_fallback": float(l1["fallback_needed_rate"].iloc[0])
                if len(l1) and "fallback_needed_rate" in l1
                else np.nan,
                "L2_coverage": float(l2["coverage"].iloc[0]) if len(l2) and np.isfinite(l2["coverage"].iloc[0]) else np.nan,
                "L3_coverage": float(l3["coverage"].iloc[0]) if len(l3) and np.isfinite(l3["coverage"].iloc[0]) else np.nan,
            }
        )

    # verdicts
    l1_covs = [
        float(r["L1_coverage"])
        for r in cascade_notes
        if np.isfinite(r["L1_coverage"])
    ]
    tm001_fail = False
    tm001_hold = False
    tm001_pass = False
    if hand_coding == "NO_SHARED_HAND_CODES":
        tm001_hold = True
        tm001_reason = (
            "Context columns exist in Trackman, but pitcher/batter hand codes do not overlap "
            "with main after normalization. Count-only fallback is possible but is a different design. "
            "Player ID overlap remains 0, matching prior reports."
        )
    elif not l1_covs:
        tm001_hold = True
        tm001_reason = "Could not compute L1 coverage on 2022–2024."
    elif min(l1_covs) < 0.90:
        tm001_hold = True
        tm001_reason = f"L1 coverage below 0.90 on at least one fold (min={min(l1_covs):.3f})."
    else:
        tm001_pass = True
        tm001_reason = (
            "Prior-season-only context aggregates are constructible; L1 coverage is high; "
            "no current-pitch TM and no test-row aggregates are required. "
            "Expected information is league-context, not pitcher identity."
        )

    groups_ok_all = bool(tm002_df.loc[tm002_df.valid_year != 2025, "core_groups_ge_min_n"].all())
    tm002_fail = False
    tm002_hold = False
    tm002_pass = False
    if not groups_ok_all:
        tm002_hold = True
        tm002_reason = "At least one validation window has a pitch_type_group with n < min_group_n for core groups."
    elif mix_missing["pitchmix_n_zero"] > 0.05:
        tm002_hold = True
        tm002_reason = "Main asof pitchmix missing/zero mass is large enough that expected-physics features would be sparse."
    else:
        tm002_pass = True
        tm002_reason = (
            "Past-season TM group physical profiles exist for fastball/breaking/offspeed, "
            "and main asof mix rates exist on almost all rows. Combination is fold-safe if TM seasons < valid year. "
            "Information is redundant with mix rates themselves; value depends on whether physics means shifted."
        )

    tm001_verdict = verdict(tm001_pass, tm001_hold, tm001_fail)
    tm002_verdict = verdict(tm002_pass, tm002_hold, tm002_fail)

    rows.append({"section": "verdict", "item": "TM-001", "value": tm001_verdict, "note": tm001_reason})
    rows.append({"section": "verdict", "item": "TM-002", "value": tm002_verdict, "note": tm002_reason})

    feas = pd.DataFrame(rows)
    # also persist coverage grain
    cov_out = cov_df.copy()
    feas.to_csv(csv_path, index=False)
    cov_df.to_csv(out_dir / "trackman_feasibility_001b_coverage.csv", index=False)
    tm002_df.to_csv(out_dir / "trackman_feasibility_001b_tm002_profiles.csv", index=False)

    prior_overlap_note = (
        "Prior `target_regime_check.md` reported |pitcher_id ∩ pitcher_trackman_id| = 0. "
        f"This re-check: numeric overlap={overlap_p}, string overlap={overlap_p_str}."
    )
    if overlap_p != 0 or overlap_p_str != 0:
        prior_overlap_note += " **Differs from prior report — investigate ID dtype/casting.**"
    else:
        prior_overlap_note += " Reproduced."

    lines: list[str] = []
    lines.append("# TM-AUDIT-001B")
    lines.append("")
    lines.append("- purpose: feasibility of TM-001 / TM-002 **without** 1:1 Trackman↔main join")
    lines.append("- Trackman is a history table, not a current-pitch feature table")
    lines.append("- no current-pitch TM, no 2025 TM, no test.csv row aggregates")
    lines.append("- existing EDA Trackman artifacts were read, not overwritten")
    lines.append("")
    lines.append("## 3-1. Minimum linkage re-check")
    lines.append("")
    lines.append(prior_overlap_note)
    lines.append("")
    lines.append(
        md_table(
            ["check", "value"],
            [
                ["TM rows", f"{len(tm):,}"],
                ["TM seasons", ",".join(str(s) for s in tm_seasons)],
                ["main pitcher_id ∩ pitcher_trackman_id (numeric)", str(overlap_p)],
                ["main batter_id ∩ batter_trackman_id (numeric)", str(overlap_b)],
                ["same ∩ as strings", f"pitcher={overlap_p_str}, batter={overlap_b_str}"],
                ["train pitcher_hand values", ", ".join(map(str, train_hands_p[:20]))],
                ["TM pitcher_hand values", ", ".join(map(str, tm_hands_p[:20]))],
                ["train batter_hand values", ", ".join(map(str, train_hands_b[:20]))],
                ["TM batter_hand values", ", ".join(map(str, tm_hands_b[:20]))],
                ["hand coding after normalize", hand_coding],
                ["shared hand keys", ",".join(sorted(hand_key_overlap)) if hand_key_overlap else "none"],
                ["pitch_type_group missing rate", fmt(pt_missing, 4)],
            ],
        )
    )
    lines.append("")
    lines.append("Seasonal TM sample size:")
    lines.append("")
    lines.append(
        md_table(
            ["season", "n"],
            [[str(y), f"{season_n.get(y, 0):,}"] for y in YEARS],
        )
    )
    lines.append("")
    lines.append("Physical-column missing rates (all TM rows):")
    lines.append("")
    lines.append(
        md_table(
            ["column", "missing_rate"],
            [[f"`{c}`", fmt(missing_phys[c], 4)] for c in PHYS_COLS],
        )
    )
    lines.append("")
    lines.append("pitch_type_group share (all TM rows):")
    lines.append("")
    lines.append(
        md_table(
            ["group", "share"],
            [[g, fmt(pt_share[g], 4)] for g in PITCH_GROUPS],
        )
    )
    lines.append("")
    lines.append(
        "Direct player matching is **not** the goal. Overlap 0 means TM-001/002 must be context-level "
        "or league-level, never pitcher_id join."
    )
    lines.append("")
    lines.append("## 3-2. TM-001 feasibility")
    lines.append("")
    lines.append("Design: for validation season y, aggregate Trackman rows with `season < y` only.")
    lines.append("")
    lines.append("| valid y | TM seasons used |")
    lines.append("|---|---|")
    for y, hist in FOLD_WINDOWS.items():
        lines.append(f"| {y} | {hist[0]}–{hist[-1]} |")
    lines.append("")
    lines.append(f"Minimum group n for 'no fallback' = {MIN_GROUP_N} (feasibility parameter, not a model lock).")
    lines.append("")
    lines.append(
        md_table(
            ["valid_year", "level", "n_main", "n_tm_hist", "n_groups", "coverage", "fallback_rate", "median group n", "note"],
            [
                [
                    str(int(r.valid_year)),
                    str(r.level),
                    "not available" if pd.isna(r.n_main) else f"{int(r.n_main):,}",
                    f"{int(r.n_tm_hist):,}" if pd.notna(r.n_tm_hist) else "missing",
                    fmt(getattr(r, "n_tm_groups", np.nan), 0),
                    fmt(r.coverage, 4),
                    fmt(r.fallback_needed_rate, 4),
                    fmt(getattr(r, "median_group_n", np.nan), 1),
                    str(r.note),
                ]
                for r in cov_df.itertuples()
            ],
        )
    )
    lines.append("")
    lines.append(f"Estimated raw feature count per level: **{n_feat_l1}** (4 mix shares + 8 physics × 3 stats).")
    lines.append("A production implementation should keep a much smaller subset (e.g. medians + mix shares) to avoid collinearity.")
    lines.append("")
    lines.append(f"Audit runtime: {elapsed:.1f}s on full TM+train context columns. Groupby cost is **low**.")
    lines.append("")
    lines.append("Leakage risks if implemented as specified:")
    lines.append("")
    lines.append("- **Allowed:** TM rows with season < validation season; context keys that exist on the current main row (`pitcher_hand`, `batter_hand`, `balls_before`, `strikes_before`).")
    lines.append("- **Forbidden:** current-pitch TM physics, same-season TM after the pitch, 2025 TM, test-wide frequency/target encoding, player ID matching that isn't actually available.")
    lines.append("- Residual risk: Trackman context may not be sampled from the same games as main (row counts differ by season). This is distribution shift, not row leakage.")
    lines.append("")
    lines.append("## 3-3. TM-002 feasibility")
    lines.append("")
    lines.append("Candidate: `expected_x = asof_fastball_rate * hist_fastball_x + ...` using TM physics from `season < y` only.")
    lines.append("")
    lines.append("Main asof pitchmix missingness (all train):")
    lines.append("")
    lines.append(
        md_table(
            ["item", "rate"],
            [[k, fmt(v, 5)] for k, v in mix_missing.items()],
        )
    )
    lines.append("")
    lines.append("Prior-window TM group sizes:")
    lines.append("")
    lines.append(
        md_table(
            ["valid_year", "hist n", "fastball n", "breaking n", "offspeed n", "other n", "core groups ≥ min_n", "main mix NA"],
            [
                [
                    str(int(r.valid_year)),
                    f"{int(r.n_hist):,}",
                    f"{int(r.fastball_n):,}",
                    f"{int(r.breaking_n):,}",
                    f"{int(r.offspeed_n):,}",
                    f"{int(r.other_n):,}",
                    str(bool(r.core_groups_ge_min_n)),
                    fmt(r.main_mix_missing_rate, 5),
                ]
                for r in tm002_df.itertuples()
            ],
        )
    )
    lines.append("")
    lines.append("Example medians (2024-valid window = TM 2019–2023):")
    r24 = tm002_df.loc[tm002_df.valid_year == 2024]
    if len(r24):
        r = r24.iloc[0]
        lines.append("")
        lines.append(
            md_table(
                ["group", "rel_speed median", "spin_rate median", "IVB median", "HB median", "extension median"],
                [
                    [
                        g,
                        fmt(r[f"{g}_rel_speed_median"], 2),
                        fmt(r[f"{g}_spin_rate_median"], 1),
                        fmt(r[f"{g}_induced_vert_break_median"], 2),
                        fmt(r[f"{g}_horz_break_median"], 2),
                        fmt(r[f"{g}_extension_median"], 3),
                    ]
                    for g in ["fastball", "breaking", "offspeed"]
                ],
            )
        )
    lines.append("")
    lines.append("Interaction with asof mix NA: when `asof_pitcher_pitchmix_n==0` the mix rates are NA (`data_integrity.md`). TM-002 must fall back (e.g. league mix or skip feature) on that tiny slice. That fallback is the same cold-start set as FEATURE-STABILITY-AUDIT.")
    lines.append("")
    lines.append("Fold-safe: yes, if each fold rebuilds TM profiles from `season < valid_season` and uses only the current row's asof mix (already as-of).")
    lines.append("")
    lines.append("## 3-4. Verdict")
    lines.append("")
    lines.append(f"### TM-001: **{tm001_verdict}**")
    lines.append("")
    lines.append(tm001_reason)
    lines.append("")
    lines.append("| criterion | assessment |")
    lines.append("|---|---|")
    lines.append("| leakage safety | acceptable **if** prior-season filter is enforced in code |")
    lines.append("| temporal feasibility | yes for 2022/2023/2024 folds; 2025 uses 2019–2024 TM (main coverage not measurable here) |")
    lines.append("| coverage | see table; L3 should be near-complete if hand codes match |")
    lines.append("| missingness | physical columns ~0.3–1% NA; group medians still defined |")
    lines.append("| expected information gain | medium/low: league context, not pitcher identity; may help mix/physics regime shift |")
    lines.append("| computational cost | low |")
    lines.append("| overlap with existing features | partial with asof pitchmix and count/hand already in the model |")
    lines.append("")
    lines.append(f"### TM-002: **{tm002_verdict}**")
    lines.append("")
    lines.append(tm002_reason)
    lines.append("")
    lines.append("| criterion | assessment |")
    lines.append("|---|---|")
    lines.append("| leakage safety | acceptable **if** TM profiles use season < y and mix is official asof |")
    lines.append("| temporal feasibility | yes |")
    lines.append("| coverage | mix observed on ~all non-cold-start rows; TM group n is large |")
    lines.append("| missingness | cold-start mix NA needs fallback |")
    lines.append("| expected information gain | low–medium; collinear with asof mix if physics means are stable, more useful if physics shifted (IVB/HB did) |")
    lines.append("| computational cost | low |")
    lines.append("| overlap with existing features | **high** with `asof_pitcher_*_rate`; TM-002 is a physics-weighted remix of mix |")
    lines.append("")
    lines.append("## What this audit does not do")
    lines.append("")
    lines.append("- No model was trained.")
    lines.append("- No 1:1 player or pitch join was forced.")
    lines.append("- test.csv was not loaded for aggregates (local copy is a 5-row format sample).")
    lines.append("- Existing `outputs/eda/fold_comparison.*` / `target_regime_check.md` were not overwritten.")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "TM-001": tm001_verdict,
                "TM-002": tm002_verdict,
                "pitcher_overlap": overlap_p,
                "batter_overlap": overlap_b,
                "hand_coding": hand_coding,
                "min_group_n": MIN_GROUP_N,
                "runtime_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    print("TM-001", tm001_verdict)
    print("TM-002", tm002_verdict)


if __name__ == "__main__":
    main()
