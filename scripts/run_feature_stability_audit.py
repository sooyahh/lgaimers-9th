"""FEATURE-STABILITY-AUDIT: season-wise feature vs target stability (no training).

Reads train.csv only. Does not modify validation, metrics, or experiment artifacts.
Does not read test.csv (no test-row aggregates).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, load_config, resolve_path

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
N_Q = 5
NEAR_ZERO = 0.01
MIN_CORR_N = 200

PRIORITY_A = [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]
PRIORITY_B = ["asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"]
PRIORITY_C = [
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
PRIORITY_D = [
    "balls_before",
    "strikes_before",
    "outs_before",
    "inning",
    "base_state",
    "num_runners_on",
    "score_diff_pitcher_team",
    "li",
    "home_win_expectancy",
    "away_win_expectancy",
    "pitcher_hand",
    "batter_hand",
]
FAMILY = {
    **{c: "pitcher_history" for c in PRIORITY_A},
    **{c: "batter_history" for c in PRIORITY_B},
    **{c: "pitch_mix" for c in PRIORITY_C},
    **{c: "game_context" for c in PRIORITY_D},
}
CATEGORICAL_FORCE = {"base_state", "pitcher_hand", "batter_hand"}


def _fmt(x: Any, nd: int = 6) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "missing"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def corr_pair(y: pd.Series, x: pd.Series) -> tuple[int, float, float]:
    mask = y.notna() & x.notna()
    n = int(mask.sum())
    if n < MIN_CORR_N or y.loc[mask].nunique() < 2 or x.loc[mask].nunique() < 2:
        return n, float("nan"), float("nan")
    a = y.loc[mask].astype(float)
    b = x.loc[mask].astype(float)
    return n, float(a.corr(b, method="pearson")), float(a.corr(b, method="spearman"))


def quantile_bin_target_rates(y: pd.Series, x: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i in range(1, N_Q + 1):
        out[f"q{i}_n"] = 0
        out[f"q{i}_target_rate"] = np.nan
    mask = y.notna() & x.notna()
    if int(mask.sum()) < MIN_CORR_N or x.loc[mask].nunique() < 2:
        return out
    work = pd.DataFrame({"y": y.loc[mask].astype(float), "x": x.loc[mask].astype(float)})
    try:
        work["bin"] = pd.qcut(work["x"], q=N_Q, labels=False, duplicates="drop")
    except ValueError:
        return out
    grouped = work.groupby("bin", observed=True)["y"]
    for b, part in grouped:
        idx = int(b) + 1
        if 1 <= idx <= N_Q:
            out[f"q{idx}_n"] = int(len(part))
            out[f"q{idx}_target_rate"] = float(part.mean())
    return out


def season_numeric_row(part: pd.DataFrame, feature: str, year: int) -> dict[str, Any]:
    x = pd.to_numeric(part[feature], errors="coerce")
    y = part["control_success"].astype(float)
    n = int(len(part))
    miss = float(x.isna().mean())
    finite = x.dropna()
    n_corr, pearson, spearman = corr_pair(y, x)
    qbins = quantile_bin_target_rates(y, x)
    row: dict[str, Any] = {
        "feature": feature,
        "family": FAMILY.get(feature, "other"),
        "season": year,
        "n": n,
        "missing_rate": miss,
        "n_zero": int((x.fillna(-1) == 0).sum()) if feature.endswith("_n") else np.nan,
        "zero_rate": float((x.fillna(-1) == 0).mean()) if feature.endswith("_n") else np.nan,
        "mean": float(finite.mean()) if len(finite) else np.nan,
        "std": float(finite.std(ddof=0)) if len(finite) else np.nan,
        "median": float(finite.median()) if len(finite) else np.nan,
        "p10": float(finite.quantile(0.10)) if len(finite) else np.nan,
        "p90": float(finite.quantile(0.90)) if len(finite) else np.nan,
        "corr_n": n_corr,
        "pearson": pearson,
        "spearman": spearman,
        "q5_minus_q1_target_rate": (
            qbins["q5_target_rate"] - qbins["q1_target_rate"]
            if np.isfinite(qbins["q5_target_rate"]) and np.isfinite(qbins["q1_target_rate"])
            else np.nan
        ),
        **qbins,
    }
    return row


def classify_numeric(feature: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pearsons = [float(r["pearson"]) for r in rows if np.isfinite(r["pearson"])]
    missings = [float(r["missing_rate"]) for r in rows]
    zero_rates = [float(r["zero_rate"]) for r in rows if np.isfinite(r.get("zero_rate", np.nan))]
    q_deltas = [
        float(r["q5_minus_q1_target_rate"])
        for r in rows
        if np.isfinite(r.get("q5_minus_q1_target_rate", np.nan))
    ]
    notes: list[str] = []
    if not pearsons:
        return {
            "feature": feature,
            "stability_class": "UNKNOWN",
            "reason": "correlation not computable in any season",
            "n_sign_pos": 0,
            "n_sign_neg": 0,
            "n_near_zero": 0,
            "pearson_min": np.nan,
            "pearson_max": np.nan,
            "pearson_range": np.nan,
            "mean_abs_pearson": np.nan,
            "max_missing_rate": max(missings) if missings else np.nan,
            "max_zero_rate": max(zero_rates) if zero_rates else np.nan,
        }

    n_pos = sum(p > NEAR_ZERO for p in pearsons)
    n_neg = sum(p < -NEAR_ZERO for p in pearsons)
    n_zero = sum(abs(p) <= NEAR_ZERO for p in pearsons)
    pmin, pmax = min(pearsons), max(pearsons)
    prange = pmax - pmin
    mean_abs = float(np.mean(np.abs(pearsons)))
    max_miss = max(missings) if missings else 0.0
    max_zero = max(zero_rates) if zero_rates else 0.0
    q_pos = sum(d > 0.005 for d in q_deltas)
    q_neg = sum(d < -0.005 for d in q_deltas)

    is_count_n = feature.endswith("_n") and feature.startswith("asof_")
    is_prev = "prev" in feature
    coldish = (is_prev and max_miss >= 0.01) or (is_count_n and max_zero >= 0.01)

    if n_pos >= 1 and n_neg >= 1:
        label = "SIGN_FLIP"
        notes.append(
            f"Pearson sign differs across seasons "
            f"(pos={n_pos}, neg={n_neg}, near0={n_zero}; range={prange:.4f})"
        )
    elif coldish and mean_abs < 0.02:
        label = "SPARSE/COLD_START"
        notes.append(
            "missing/zero-n is material and the observed target association is weak"
        )
    elif mean_abs < 0.012 and prange < 0.03:
        label = "WEAK_STABLE"
        notes.append("association is small in every season; direction is not strongly identified")
    elif n_pos == 0 and n_neg == 0:
        label = "WEAK_STABLE"
        notes.append("all seasonal Pearson values are near zero")
    elif prange >= 0.04 and mean_abs >= 0.012:
        label = "UNSTABLE"
        notes.append(
            f"direction mostly consistent but magnitude moves a lot "
            f"(pearson range={prange:.4f}, mean|r|={mean_abs:.4f})"
        )
    elif (n_pos > 0 and n_neg == 0) or (n_neg > 0 and n_pos == 0):
        if mean_abs < 0.025:
            label = "WEAK_STABLE"
            notes.append("direction is consistent among non-near-zero seasons but effect is small")
        else:
            label = "STABLE"
            notes.append("direction is consistent across seasons with a non-trivial mean |r|")
    else:
        label = "UNKNOWN"
        notes.append("mixed pattern; no conservative label assigned")

    if q_pos >= 1 and q_neg >= 1 and label not in {"SIGN_FLIP", "UNKNOWN"}:
        notes.append("quantile-bin target-rate slope also changes sign across seasons")
        if label == "STABLE":
            label = "UNSTABLE"
    if coldish and label in {"STABLE", "WEAK_STABLE"}:
        notes.append("cold-start/missing mass is non-trivial even if the observed-row association is usable")

    if 2023 in [r["season"] for r in rows]:
        r23 = next(r for r in rows if r["season"] == 2023)
        r22 = next((r for r in rows if r["season"] == 2022), None)
        if r22 is not None and np.isfinite(r22["pearson"]) and np.isfinite(r23["pearson"]):
            if abs(r23["pearson"]) + 0.02 < abs(r22["pearson"]) and abs(r22["pearson"]) >= 0.02:
                notes.append("2023 |Pearson| is materially weaker than 2022")

    return {
        "feature": feature,
        "stability_class": label,
        "reason": "; ".join(notes) if notes else "see seasonal table",
        "n_sign_pos": n_pos,
        "n_sign_neg": n_neg,
        "n_near_zero": n_zero,
        "pearson_min": pmin,
        "pearson_max": pmax,
        "pearson_range": prange,
        "mean_abs_pearson": mean_abs,
        "max_missing_rate": max_miss,
        "max_zero_rate": max_zero if zero_rates else np.nan,
    }


def classify_categorical(feature: str, cat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    seasons = sorted({r["season"] for r in cat_rows})
    levels = sorted({str(r["category"]) for r in cat_rows})
    rate_map: dict[int, dict[str, float]] = {y: {} for y in seasons}
    share_map: dict[int, dict[str, float]] = {y: {} for y in seasons}
    for r in cat_rows:
        rate_map[r["season"]][str(r["category"])] = float(r["target_rate"])
        share_map[r["season"]][str(r["category"])] = float(r["share"])

    # rank correlation of category target-rates vs 2019 (or first season)
    base = seasons[0]
    rank_corrs = []
    for y in seasons[1:]:
        common = [lv for lv in levels if lv in rate_map[base] and lv in rate_map[y]]
        if len(common) < 3:
            continue
        a = pd.Series({lv: rate_map[base][lv] for lv in common})
        b = pd.Series({lv: rate_map[y][lv] for lv in common})
        rank_corrs.append(float(a.corr(b, method="spearman")))
    mean_rank = float(np.nanmean(rank_corrs)) if rank_corrs else np.nan

    share_tv = []
    for y0, y1 in zip(seasons[:-1], seasons[1:]):
        keys = set(share_map[y0]) | set(share_map[y1])
        tv = 0.5 * sum(abs(share_map[y0].get(k, 0.0) - share_map[y1].get(k, 0.0)) for k in keys)
        share_tv.append(tv)
    max_tv = max(share_tv) if share_tv else np.nan

    # 2022 vs 2023 level-rate deltas
    d22_23 = []
    if 2022 in rate_map and 2023 in rate_map:
        for lv in set(rate_map[2022]) & set(rate_map[2023]):
            d22_23.append(abs(rate_map[2023][lv] - rate_map[2022][lv]))
    max_rate_shift = max(d22_23) if d22_23 else np.nan

    if not np.isfinite(mean_rank):
        label = "UNKNOWN"
        reason = "too few overlapping categories to judge rank stability"
    elif mean_rank >= 0.7 and (not np.isfinite(max_rate_shift) or max_rate_shift < 0.04):
        label = "STABLE"
        reason = f"category target-rate ranks are consistent (mean Spearman vs first season={mean_rank:.3f})"
    elif mean_rank >= 0.4:
        label = "WEAK_STABLE" if (not np.isfinite(max_rate_shift) or max_rate_shift < 0.05) else "UNSTABLE"
        reason = (
            f"ranks somewhat consistent (mean Spearman={mean_rank:.3f}); "
            f"max |2023-2022| rate shift={_fmt(max_rate_shift, 4)}"
        )
    else:
        label = "UNSTABLE"
        reason = f"category target-rate ranking is not stable (mean Spearman={mean_rank:.3f})"

    if np.isfinite(max_tv) and max_tv >= 0.08 and label == "STABLE":
        label = "UNSTABLE"
        reason += f"; composition also shifts (max adjacent TV={max_tv:.3f})"

    return {
        "feature": feature,
        "stability_class": label,
        "reason": reason,
        "n_sign_pos": np.nan,
        "n_sign_neg": np.nan,
        "n_near_zero": np.nan,
        "pearson_min": np.nan,
        "pearson_max": np.nan,
        "pearson_range": np.nan,
        "mean_abs_pearson": mean_rank,
        "max_missing_rate": 0.0,
        "max_zero_rate": np.nan,
        "cat_mean_rank_spearman": mean_rank,
        "cat_max_share_tv": max_tv,
        "cat_max_abs_rate_shift_22_23": max_rate_shift,
    }


def save_plots(
    target_by_season: pd.DataFrame,
    numeric_df: pd.DataFrame,
    class_df: pd.DataFrame,
    plot_dir: Path,
) -> list[str]:
    ensure_dir(plot_dir)
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(target_by_season["season"], target_by_season["target_mean"], marker="o")
    ax.set_title("Train target rate by season")
    ax.set_xlabel("Season")
    ax.set_ylabel("P(control_success=1)")
    ax.set_ylim(0.45, 0.60)
    fig.tight_layout()
    p = plot_dir / "target_rate_by_season.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)).replace("\\", "/"))

    rate_feats = [
        c
        for c in PRIORITY_A + PRIORITY_B + PRIORITY_C
        if "rate" in c and c in set(numeric_df["feature"])
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for feat in rate_feats:
        sub = numeric_df.loc[numeric_df["feature"] == feat]
        ax.plot(sub["season"], sub["pearson"], marker="o", linewidth=1, label=feat.replace("asof_", ""))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Seasonal Pearson correlation with control_success")
    ax.set_xlabel("Season")
    ax.set_ylabel("Pearson r")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    p = plot_dir / "pearson_by_season_asof_rates.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)).replace("\\", "/"))

    miss_feats = [c for c in PRIORITY_A + PRIORITY_B + PRIORITY_C if c in set(numeric_df["feature"])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for feat in miss_feats:
        sub = numeric_df.loc[numeric_df["feature"] == feat]
        if float(sub["missing_rate"].max()) < 0.0001:
            continue
        ax.plot(sub["season"], sub["missing_rate"], marker="o", linewidth=1, label=feat.replace("asof_", ""))
    ax.set_title("Seasonal missing rate (features with any NA)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Missing rate")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = plot_dir / "missing_rate_by_season.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)).replace("\\", "/"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = class_df.sort_values(["stability_class", "feature"])
    colors = {
        "STABLE": "#1f4e79",
        "WEAK_STABLE": "#5b8db8",
        "UNSTABLE": "#c47b14",
        "SIGN_FLIP": "#a33b3b",
        "SPARSE/COLD_START": "#6b6b6b",
        "UNKNOWN": "#9a9a9a",
    }
    y_pos = np.arange(len(order))
    ax.barh(
        y_pos,
        order["mean_abs_pearson"].fillna(0.0),
        color=[colors.get(c, "#9a9a9a") for c in order["stability_class"]],
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order["feature"], fontsize=7)
    ax.set_xlabel("Mean |Pearson| across seasons (categorical: rank Spearman)")
    ax.set_title("Stability class vs mean association strength")
    fig.tight_layout()
    p = plot_dir / "stability_class_overview.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)).replace("\\", "/"))
    return paths


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config("configs/baseline.yaml")
    train_path = resolve_path(cfg["data"]["train_path"])
    out_dir = ensure_dir(ROOT / "outputs" / "analysis")
    plot_dir = ensure_dir(out_dir / "feature_stability")
    report_path = ROOT / "reports" / "feature_stability_audit.md"

    train = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    present = set(train.columns)
    requested = PRIORITY_A + PRIORITY_B + PRIORITY_C + PRIORITY_D
    used = [c for c in requested if c in present]
    missing_cols = [c for c in requested if c not in present]
    numeric_used = [c for c in used if c not in CATEGORICAL_FORCE]
    cat_used = [c for c in used if c in CATEGORICAL_FORCE or train[c].dtype == object]
    # discrete context: analyze both as numeric (if numeric dtype) and as category
    cat_extra = [
        c
        for c in [
            "balls_before",
            "strikes_before",
            "outs_before",
            "inning",
            "num_runners_on",
            "pitcher_hand",
            "batter_hand",
        ]
        if c in present
    ]
    cat_used = sorted(set(cat_used) | set(cat_extra) | {"base_state"})

    target_rows = []
    for y in YEARS:
        s = train.loc[train["season"] == y, "control_success"]
        target_rows.append(
            {
                "season": y,
                "n": int(len(s)),
                "target_mean": float(s.mean()),
                "target_std": float(s.std(ddof=0)),
            }
        )
    target_df = pd.DataFrame(target_rows)
    overall_target = float(train["control_success"].mean())

    numeric_rows: list[dict[str, Any]] = []
    for feat in numeric_used:
        for y in YEARS:
            part = train.loc[train["season"] == y]
            numeric_rows.append(season_numeric_row(part, feat, y))
        numeric_rows.append(season_numeric_row(train, feat, 0))  # season=0 => all seasons
    numeric_df = pd.DataFrame(numeric_rows)

    cat_rows: list[dict[str, Any]] = []
    for feat in cat_used:
        for y in YEARS:
            part = train.loc[train["season"] == y]
            n = len(part)
            vc = part[feat].value_counts(dropna=False)
            for cat, cnt in vc.items():
                sub = part.loc[part[feat].eq(cat) if pd.notna(cat) else part[feat].isna()]
                cat_rows.append(
                    {
                        "feature": feat,
                        "family": FAMILY.get(feat, "game_context"),
                        "season": y,
                        "category": "NA" if pd.isna(cat) else str(cat),
                        "n": int(cnt),
                        "share": float(cnt / n) if n else np.nan,
                        "target_rate": float(sub["control_success"].mean()) if len(sub) else np.nan,
                    }
                )
    cat_df = pd.DataFrame(cat_rows)

    class_rows: list[dict[str, Any]] = []
    for feat in numeric_used:
        season_only = [r for r in numeric_rows if r["feature"] == feat and r["season"] != 0]
        rec = classify_numeric(feat, season_only)
        rec["family"] = FAMILY.get(feat, "other")
        rec["analysis_kind"] = "numeric"
        class_rows.append(rec)
    for feat in cat_used:
        rec = classify_categorical(feat, [r for r in cat_rows if r["feature"] == feat])
        rec["family"] = FAMILY.get(feat, "game_context")
        rec["analysis_kind"] = "categorical"
        class_rows.append(rec)
    class_df = pd.DataFrame(class_rows)

    # attach class to seasonal numeric table
    class_map = {
        r["feature"]: r["stability_class"]
        for r in class_rows
        if r["analysis_kind"] == "numeric"
    }
    numeric_df["stability_class"] = numeric_df["feature"].map(class_map)

    csv_path = out_dir / "feature_stability_by_season.csv"
    numeric_df.to_csv(csv_path, index=False)
    cat_csv = out_dir / "feature_stability_categorical_by_season.csv"
    cat_df.to_csv(cat_csv, index=False)
    class_csv = out_dir / "feature_stability_classification.csv"
    class_df.to_csv(class_csv, index=False)

    plot_paths = save_plots(target_df, numeric_df.loc[numeric_df["season"] != 0], class_df, plot_dir)

    # --- report ---
    num_class = class_df.loc[class_df["analysis_kind"] == "numeric"].copy()
    by_class = num_class.groupby("stability_class")["feature"].apply(list).to_dict()

    def top_stable(n: int = 8) -> pd.DataFrame:
        sub = num_class.loc[num_class["stability_class"].isin(["STABLE", "WEAK_STABLE"])]
        return sub.sort_values("mean_abs_pearson", ascending=False).head(n)

    def top_unstable(n: int = 8) -> pd.DataFrame:
        sub = num_class.loc[num_class["stability_class"].isin(["UNSTABLE", "SIGN_FLIP", "SPARSE/COLD_START"])]
        return sub.sort_values("pearson_range", ascending=False).head(n)

    lines: list[str] = []
    lines.append("# FEATURE-STABILITY-AUDIT")
    lines.append("")
    lines.append("- purpose: measure 2019–2024 stability of priority features vs `control_success`")
    lines.append("- data: `data/raw/train.csv` only (no test-row aggregates, no retraining)")
    lines.append("- metric of association: Pearson and Spearman vs target; not a model-selection metric")
    lines.append("- classification: conservative heuristic, not a hypothesis test")
    lines.append(f"- train rows: {len(train):,}")
    lines.append("")
    lines.append("## 1. Executive summary")
    lines.append("")
    lines.append(
        f"Target rate falls from **{target_df.loc[target_df.season==2019,'target_mean'].iloc[0]:.4f}** "
        f"(2019) to **{target_df.loc[target_df.season==2024,'target_mean'].iloc[0]:.4f}** (2024), "
        "with a clear 2022→2023 drop. Several asof rate features that look usable in 2019–2022 "
        "change magnitude or **sign** in 2023–2024. Count/context features are more compositionally "
        "stable than pitcher-history rates. Prev-game asof features are dominated by cold-start NA "
        "at the start of a pitcher-season, not by a strong stable signal."
    )
    lines.append("")
    lines.append("Numeric-feature class counts:")
    lines.append("")
    for lab in ["STABLE", "WEAK_STABLE", "UNSTABLE", "SIGN_FLIP", "SPARSE/COLD_START", "UNKNOWN"]:
        feats = by_class.get(lab, [])
        lines.append(f"- **{lab}**: {len(feats)}" + (f" — `{ '`, `'.join(feats)}`" if feats else ""))
    lines.append("")
    if missing_cols:
        lines.append("Requested columns **not present** (skipped): `" + "`, `".join(missing_cols) + "`")
    else:
        lines.append("All requested priority columns are present in `train.csv`.")
    lines.append("")
    lines.append("## 2. Season-wise target change")
    lines.append("")
    lines.append(f"Overall train target mean: **{overall_target:.6f}**")
    lines.append("")
    lines.append(
        md_table(
            ["season", "n", "target_mean", "delta_vs_prev"],
            [
                [
                    str(int(r.season)),
                    f"{int(r.n):,}",
                    f"{r.target_mean:.6f}",
                    "—"
                    if i == 0
                    else f"{r.target_mean - target_df.iloc[i-1].target_mean:+.6f}",
                ]
                for i, r in target_df.iterrows()
            ],
        )
    )
    lines.append("")
    lines.append("The 2022→2023 drop is the largest adjacent change and is the main 2025-generalization risk already flagged by VAL-AUDIT-001 / gap diagnosis.")
    lines.append("")
    lines.append("## 3. Feature stability table")
    lines.append("")
    lines.append("Labels are conservative. `mean_abs_pearson` is the mean of |seasonal Pearson|. Categorical rows use mean rank-Spearman of category target rates instead.")
    lines.append("")
    cls_headers = ["feature", "family", "kind", "class", "mean|r| or rank ρ", "pearson min", "pearson max", "max missing", "note"]
    cls_body = []
    for _, r in class_df.sort_values(["family", "feature", "analysis_kind"]).iterrows():
        cls_body.append(
            [
                f"`{r.feature}`",
                str(r.family),
                str(r.analysis_kind),
                str(r.stability_class),
                _fmt(r.mean_abs_pearson, 4),
                _fmt(r.pearson_min, 4),
                _fmt(r.pearson_max, 4),
                _fmt(r.max_missing_rate, 4),
                str(r.reason),
            ]
        )
    lines.append(md_table(cls_headers, cls_body))
    lines.append("")
    lines.append("Seasonal Pearson for numeric priority features (season=0 is pooled 2019–2024, not used for classification):")
    lines.append("")
    pearson_headers = ["feature"] + [str(y) for y in YEARS] + ["pooled"]
    pearson_body = []
    for feat in numeric_used:
        sub = numeric_df.loc[numeric_df["feature"] == feat]
        row = [f"`{feat}`"]
        for y in YEARS + [0]:
            val = sub.loc[sub["season"] == y, "pearson"]
            row.append(_fmt(val.iloc[0], 4) if len(val) else "missing")
        pearson_body.append(row)
    lines.append(md_table(pearson_headers, pearson_body))
    lines.append("")
    lines.append("## 4. Most stable features (TOP)")
    lines.append("")
    stable_top = top_stable(8)
    if stable_top.empty:
        lines.append("No feature met the conservative STABLE/WEAK_STABLE bar with a usable ranking.")
    else:
        lines.append(
            md_table(
                ["feature", "class", "mean|r|", "pearson min", "pearson max", "reason"],
                [
                    [
                        f"`{r.feature}`",
                        str(r.stability_class),
                        _fmt(r.mean_abs_pearson, 4),
                        _fmt(r.pearson_min, 4),
                        _fmt(r.pearson_max, 4),
                        str(r.reason),
                    ]
                    for _, r in stable_top.iterrows()
                ],
            )
        )
    lines.append("")
    lines.append("Context features with stable category ranking (from categorical analysis):")
    lines.append("")
    cat_stable = class_df.loc[
        (class_df["analysis_kind"] == "categorical")
        & (class_df["stability_class"].isin(["STABLE", "WEAK_STABLE"]))
    ].sort_values("mean_abs_pearson", ascending=False)
    if cat_stable.empty:
        lines.append("None classified STABLE/WEAK_STABLE.")
    else:
        lines.append(
            md_table(
                ["feature", "class", "rank Spearman", "max 22→23 |rate| shift", "reason"],
                [
                    [
                        f"`{r.feature}`",
                        str(r.stability_class),
                        _fmt(r.mean_abs_pearson, 3),
                        _fmt(r.cat_max_abs_rate_shift_22_23, 4),
                        str(r.reason),
                    ]
                    for _, r in cat_stable.iterrows()
                ],
            )
        )
    lines.append("")
    lines.append("## 5. Most unstable features (TOP)")
    lines.append("")
    unstable_top = top_unstable(10)
    if unstable_top.empty:
        lines.append("No numeric feature classified UNSTABLE / SIGN_FLIP / SPARSE.")
    else:
        lines.append(
            md_table(
                ["feature", "class", "pearson range", "min", "max", "reason"],
                [
                    [
                        f"`{r.feature}`",
                        str(r.stability_class),
                        _fmt(r.pearson_range, 4),
                        _fmt(r.pearson_min, 4),
                        _fmt(r.pearson_max, 4),
                        str(r.reason),
                    ]
                    for _, r in unstable_top.iterrows()
                ],
            )
        )
    lines.append("")
    lines.append("## 6. SIGN_FLIP or sharply weakening features")
    lines.append("")
    flips = num_class.loc[num_class["stability_class"] == "SIGN_FLIP"]
    if flips.empty:
        lines.append("No numeric feature received SIGN_FLIP.")
    else:
        for _, r in flips.iterrows():
            sub = numeric_df.loc[(numeric_df.feature == r.feature) & (numeric_df.season != 0)]
            seq = ", ".join(f"{int(x.season)}:{x.pearson:.4f}" for x in sub.itertuples())
            lines.append(f"- `{r.feature}`: {seq}")
        lines.append("")
        lines.append(
            "This matches the earlier `target_regime_check.md` observation that "
            "`asof_pitcher_ball_rate` / `asof_pitcher_strike_rate` reverse around 2022→2023."
        )
    lines.append("")
    lines.append("Features whose |Pearson| is much weaker in 2023 than 2022 are noted in the class table.")
    lines.append("")
    lines.append("## 7. Cold-start related characteristics")
    lines.append("")
    n_feats = [c for c in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"] if c in numeric_used]
    if n_feats:
        lines.append("Share of rows with `*_n == 0` by season:")
        lines.append("")
        z_headers = ["feature"] + [str(y) for y in YEARS]
        z_body = []
        for feat in n_feats:
            sub = numeric_df.loc[(numeric_df.feature == feat) & (numeric_df.season != 0)]
            z_body.append([f"`{feat}`"] + [_fmt(x, 4) for x in sub["zero_rate"]])
        lines.append(md_table(z_headers, z_body))
        lines.append("")
    prev_feats = [c for c in numeric_used if "prev" in c]
    if prev_feats:
        lines.append("Prev-game asof missing rates (identical across prev1/3/5 in prior integrity report):")
        lines.append("")
        p_headers = ["feature"] + [str(y) for y in YEARS]
        p_body = []
        for feat in prev_feats:
            sub = numeric_df.loc[(numeric_df.feature == feat) & (numeric_df.season != 0)]
            p_body.append([f"`{feat}`"] + [_fmt(x, 4) for x in sub["missing_rate"]])
        lines.append(md_table(p_headers, p_body))
        lines.append("")
    lines.append(
        "Rate features (`asof_*_rate`) are missing exactly when the corresponding n is 0 "
        "(first observation of that pitcher/batter). That is a structural cold-start, not random MCAR. "
        "Prev-game features have a larger missing block (~2% overall in `data_integrity.md`) "
        "because they require prior games in the proxy order."
    )
    lines.append("")
    lines.append("## 8. 2025 generalization risks")
    lines.append("")
    lines.append("1. Target mean is still trending down through 2024; a 2025 model trained with a 2024-like base rate can be miscalibrated if 2025 continues or reverts.")
    lines.append("2. SIGN_FLIP asof rates (ball/strike, and possibly others) will not transfer if 2025 resembles 2023–2024 rather than 2019–2022.")
    lines.append("3. Absolute `season` is already known from D/G experiments to be a brittle 2024 crutch; this audit does not re-test that, but the target shift is the same mechanism.")
    lines.append("4. Cold-start pitchers/batters in a new season cannot rely on prev-game asof or long asof_n. 2025 will re-introduce this mass.")
    lines.append("5. `li` / win expectancy remain excluded from models (`feature_safety_audit.md`); their statistical association is reported here only as context, not as an FE recommendation.")
    lines.append("")
    lines.append("## 9. Observations that can feed FE candidates")
    lines.append("")
    lines.append("These are observations, not implementations:")
    lines.append("")
    lines.append("- Count (`balls_before` × `strikes_before`) has a persistent level effect (3-0 / 3-2 lower success) and a roughly parallel 2022→2023 drop (`target_regime_check.md`). Count itself looks more stable than pitcher-rate history.")
    lines.append("- Pitcher asof success/middle rates are weak but more sign-consistent than ball/strike rates; shrinking or recency-weighting them may be safer than adding more raw asof rates.")
    lines.append("- Pitch mix rates exist and are official-asof; they may track the 2022+ league mix shift better than physical Trackman if TM cannot be player-matched.")
    lines.append("- Cold-start indicators (`asof_pitcher_n==0`, prev-game NA flags) are already implicit in missingness; explicit flags/shrinkage could be a low-leakage FE if error analysis shows those slices fail.")
    lines.append("- Do **not** treat SIGN_FLIP features as 2025-safe without a fold-wise check.")
    lines.append("")
    lines.append("## 10. Items that cannot be concluded yet")
    lines.append("")
    lines.append("- Whether instability is causal (target definition / calling / mix) vs correlational.")
    lines.append("- Whether a 2025 test distribution looks like 2023, 2024, or a new regime (full test labels not available; local test.csv is a 5-row format sample).")
    lines.append("- Whether WE/LI are leak-free; association strength is not a safety proof.")
    lines.append("- Player-level Trackman physics vs these asof rates (no ID join).")
    lines.append("- Model error concentration (see ERR-AUDIT-001); this audit is feature–target only.")
    lines.append("")
    lines.append("## Appendix")
    lines.append("")
    lines.append(f"- numeric CSV: `{csv_path.relative_to(ROOT).as_posix()}`")
    lines.append(f"- categorical CSV: `{cat_csv.relative_to(ROOT).as_posix()}`")
    lines.append(f"- classification CSV: `{class_csv.relative_to(ROOT).as_posix()}`")
    for p in plot_paths:
        lines.append(f"- plot: `{p}`")
    lines.append("")
    lines.append("Heuristic notes: `NEAR_ZERO=0.01` Pearson; SIGN_FLIP if ≥1 season >+0.01 and ≥1 season <−0.01; UNSTABLE if Pearson range ≥0.04 with non-trivial mean |r|. These cutoffs organize the table; they are not significance tests.")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "n_train": int(len(train)),
                "missing_requested_columns": missing_cols,
                "used_columns": used,
                "class_counts": {k: len(v) for k, v in by_class.items()},
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


if __name__ == "__main__":
    main()
