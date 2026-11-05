"""ERR-AUDIT-001: slice error analysis on the comparable CatBoost OOF (EXP-004).

No retraining. Does not modify src/metrics.py, validation, or existing EXP artifacts.
Reuses EXP-004 oof_holdout_predictions.csv and joins train by row_id.
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

from src.evaluate import evaluate_predictions
from src.metrics import compute_brier_metrics
from src.utils import ensure_dir, load_config, resolve_path

EXPERIMENT_ID = "EXP-004"
MIN_ROWS_SCORE = 30
MIN_ROWS_WARN = 2000
ASOF_BINS = [-0.5, 0.5, 100, 500, 2000, np.inf]
ASOF_LABELS = ["0", "1-100", "101-500", "501-2000", "2000+"]
LI_BINS = [-0.01, 0.5, 1.0, 2.0, 3.0, np.inf]
LI_LABELS = ["<=0.5", "0.5-1", "1-2", "2-3", "3+"]
SCORE_BINS = [-np.inf, -3, 0, 3, np.inf]
SCORE_LABELS = ["<=-3", "-2to0", "1to3", ">=4"]
INNING_BINS = [0, 3, 6, 9, 20]
INNING_LABELS = ["1-3", "4-6", "7-9", "10+"]

TRAIN_JOIN_COLS = [
    "row_id",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
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
    "asof_pitcher_n",
    "asof_batter_n",
    "asof_pitcher_success_rate",
    "asof_batter_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


def quantile_10_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Same definition as scripts/run_val_exp006_secondary_001.py (not a new selection metric)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    if n == 0:
        raise ValueError("empty arrays for ECE")
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        ps = p_sorted[lo:hi]
        ys = y_sorted[lo:hi]
        ece += (len(ps) / n) * abs(float(ps.mean()) - float(ys.mean()))
    return float(ece)


def asof_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float).fillna(0.0),
        bins=ASOF_BINS,
        labels=ASOF_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)


def score_group(y: np.ndarray, p: np.ndarray, min_rows: int = MIN_ROWS_SCORE) -> dict[str, Any]:
    n = int(len(y))
    pred_mean = float(np.mean(p)) if n else np.nan
    tgt_mean = float(np.mean(y)) if n else np.nan
    gap = pred_mean - tgt_mean if n else np.nan
    mae = float(np.mean(np.abs(p - y))) if n else np.nan
    pred_std = float(np.std(p)) if n else np.nan
    pred_min = float(np.min(p)) if n else np.nan
    pred_max = float(np.max(p)) if n else np.nan
    extreme = float(np.mean((p < 0.30) | (p > 0.70))) if n else np.nan
    out: dict[str, Any] = {
        "n": n,
        "target_mean": tgt_mean,
        "prediction_mean": pred_mean,
        "calibration_gap": gap,
        "mae": mae,
        "prediction_std": pred_std,
        "prediction_min": pred_min,
        "prediction_max": pred_max,
        "frac_pred_extreme": extreme,
        "brier_score": np.nan,
        "raw_brier_skill_score": np.nan,
        "official_brier_skill_score": np.nan,
        "quantile_10_ece": np.nan,
        "eligible": False,
        "warning": "",
        "reason": "ok",
    }
    if n < min_rows:
        out["reason"] = f"below_min_rows:{min_rows}"
        out["warning"] = "small_n"
        return out
    try:
        bundle = compute_brier_metrics(y, p)
        out.update(
            {
                "brier_score": float(bundle["brier_score"]),
                "raw_brier_skill_score": float(bundle["raw_brier_skill_score"]),
                "official_brier_skill_score": float(bundle["official_brier_skill_score"]),
                "quantile_10_ece": quantile_10_ece(y, p),
                "eligible": True,
            }
        )
    except ValueError as exc:
        out["reason"] = str(exc)
        return out
    if n < MIN_ROWS_WARN:
        out["warning"] = "small_n_interpret_with_caution"
    return out


def append_slices(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    family: str,
    level_col: str,
    scope: str,
    total_n: int,
) -> None:
    for level, part in frame.groupby(level_col, dropna=False, sort=True):
        y = part["control_success"].to_numpy(dtype=float)
        p = part["prediction"].to_numpy(dtype=float)
        rec = score_group(y, p)
        rec.update(
            {
                "slice_family": family,
                "slice_level": "NA" if pd.isna(level) else str(level),
                "scope": scope,
                "row_share": rec["n"] / total_n if total_n else np.nan,
            }
        )
        rows.append(rec)


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


def slice_table(df: pd.DataFrame, family: str, scope: str | None = None) -> str:
    sub = df.loc[df["slice_family"] == family].copy()
    if scope is not None:
        sub = sub.loc[sub["scope"] == scope]
    sub = sub.sort_values(["scope", "raw_brier_skill_score"], na_position="last")
    headers = [
        "scope",
        "level",
        "n",
        "share",
        "y_mean",
        "p_mean",
        "gap",
        "raw BSS",
        "official BSS",
        "Brier",
        "ECE10q",
        "MAE",
        "warn",
    ]
    body = []
    for r in sub.itertuples():
        body.append(
            [
                str(r.scope),
                str(r.slice_level),
                f"{int(r.n):,}",
                fmt(r.row_share, 3),
                fmt(r.target_mean, 4),
                fmt(r.prediction_mean, 4),
                fmt(r.calibration_gap, 4),
                fmt(r.raw_brier_skill_score, 1),
                fmt(r.official_brier_skill_score, 1),
                fmt(r.brier_score, 5),
                fmt(r.quantile_10_ece, 4),
                fmt(r.mae, 4),
                str(r.warning or ""),
            ]
        )
    return md_table(headers, body)


def main() -> None:
    cfg = load_config("configs/baseline.yaml")
    scale = float(cfg["task"]["metric"]["scale"])
    train_path = resolve_path(cfg["data"]["train_path"])
    oof_path = ROOT / "outputs" / "predictions" / EXPERIMENT_ID / "oof_holdout_predictions.csv"
    join_json = ROOT / "reports" / "gpt_audit" / "oof_join_verification.json"
    out_dir = ensure_dir(ROOT / "outputs" / "analysis")
    plot_dir = ensure_dir(out_dir / "error_audit_001")
    report_path = ROOT / "reports" / "error_audit_001.md"
    csv_path = out_dir / "error_slices_001.csv"

    oof = pd.read_csv(oof_path)
    train_ctx = pd.read_csv(train_path, encoding="utf-8-sig", usecols=TRAIN_JOIN_COLS, low_memory=False)
    train_target = pd.read_csv(
        train_path, encoding="utf-8-sig", usecols=["row_id", "control_success", "season"]
    )

    integrity: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "oof_path": str(oof_path.relative_to(ROOT)).replace("\\", "/"),
        "oof_n_rows": int(len(oof)),
        "oof_columns": list(oof.columns),
        "row_id_unique": bool(oof["row_id"].is_unique),
        "n_duplicate_row_id": int(oof["row_id"].duplicated().sum()),
        "folds_present": sorted(oof["fold_id"].astype(str).unique().tolist()),
        "seasons_present": sorted(int(x) for x in oof["season"].dropna().unique().tolist()),
        "prediction_nan": int(oof["prediction"].isna().sum()),
        "prediction_inf": int(np.isinf(pd.to_numeric(oof["prediction"], errors="coerce")).sum()),
        "prediction_min": float(oof["prediction"].min()),
        "prediction_max": float(oof["prediction"].max()),
        "prediction_outside_01": int(((oof["prediction"] < 0) | (oof["prediction"] > 1)).sum()),
        "target_nan": int(oof["control_success"].isna().sum()),
        "comparable_cv": True,
        "baseline_choice_reason": (
            "EXP-004 is the latest CatBoost with is_comparable_cv=true, expanding 3-fold OOF "
            "covering 2022/2023/2024, and passed gpt_audit join verification. EXP-006 is hold "
            "(not adopt); its 2024 file is the Optuna tune holdout; EXP-005 has no row-level OOF."
        ),
    }

    n_before = len(oof)
    merged = oof.merge(train_ctx, on="row_id", how="left", validate="one_to_one")
    integrity["n_before_join"] = n_before
    integrity["n_after_join"] = int(len(merged))
    integrity["unmatched_pitcher_id_na"] = int(merged["pitcher_id"].isna().sum())
    integrity["duplicate_row_increase"] = int(len(merged) - n_before)

    train_map = train_target.set_index("row_id")["control_success"]
    mapped = merged["row_id"].map(train_map)
    integrity["target_alignment_vs_train"] = bool((mapped == merged["control_success"]).all())
    integrity["train_map_na"] = int(mapped.isna().sum())

    if join_json.exists():
        prev = json.loads(join_json.read_text(encoding="utf-8"))
        exp4 = next((c for c in prev.get("cases", []) if "EXP-004" in c.get("enriched_file", "")), None)
        integrity["prior_join_verification_exp004"] = exp4
    else:
        integrity["prior_join_verification_exp004"] = "not available"

    # logged EXP-004 scores (from artifacts, for cross-check)
    logged = {
        "secondary_2022": 2259.3983277146012,
        "secondary_2023": -1366.4810668740479,
        "primary_2024": 578.1844155969828,
    }
    fold_scores = {}
    for fold_id, part in merged.groupby("fold_id"):
        bundle = evaluate_predictions(
            part["control_success"].to_numpy(dtype=float),
            part["prediction"].to_numpy(dtype=float),
            scale=scale,
        )
        fold_scores[str(fold_id)] = {
            "n": int(len(part)),
            "raw_brier_skill_score": float(bundle["raw_brier_skill_score"]),
            "official_brier_skill_score": float(bundle["official_brier_skill_score"]),
            "brier_score": float(bundle["brier_score"]),
            "logged_raw_bss": logged.get(str(fold_id)),
            "matches_logged": (
                abs(float(bundle["raw_brier_skill_score"]) - logged[str(fold_id)]) < 1e-8
                if str(fold_id) in logged
                else None
            ),
        }
    integrity["fold_score_crosscheck"] = fold_scores

    # derived slice columns
    merged["pitcher_n_bin"] = asof_bin(merged["asof_pitcher_n"])
    merged["batter_n_bin"] = asof_bin(merged["asof_batter_n"])
    merged["count"] = (
        merged["balls_before"].astype("Int64").astype(str)
        + "-"
        + merged["strikes_before"].astype("Int64").astype(str)
    )
    merged["hand_matchup"] = (
        merged["pitcher_hand"].astype("Int64").astype(str)
        + "x"
        + merged["batter_hand"].astype("Int64").astype(str)
    )
    merged["li_bin"] = pd.cut(
        pd.to_numeric(merged["li"], errors="coerce"),
        bins=LI_BINS,
        labels=LI_LABELS,
        include_lowest=True,
    ).astype(str)
    merged["score_bin"] = pd.cut(
        pd.to_numeric(merged["score_diff_pitcher_team"], errors="coerce"),
        bins=SCORE_BINS,
        labels=SCORE_LABELS,
        include_lowest=True,
    ).astype(str)
    merged["inning_bin"] = pd.cut(
        merged["inning"].astype(float),
        bins=INNING_BINS,
        labels=INNING_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)
    merged["pitcher_cold"] = np.where(merged["asof_pitcher_n"].fillna(0) <= 0, "n==0", "n>0")
    merged["batter_cold"] = np.where(merged["asof_batter_n"].fillna(0) <= 0, "n==0", "n>0")
    merged["prev1_missing"] = np.where(
        merged["asof_pitcher_prev1_game_success_rate"].isna(), "prev1_NA", "prev1_observed"
    )
    merged["prev5_missing"] = np.where(
        merged["asof_pitcher_prev5_game_success_rate"].isna(), "prev5_NA", "prev5_observed"
    )
    p1 = pd.to_numeric(merged["asof_pitcher_prev1_game_success_rate"], errors="coerce")
    merged["prev1_extreme"] = np.where(
        p1.isna(),
        "prev1_NA",
        np.where(p1 <= 0.35, "prev1_<=0.35", np.where(p1 >= 0.65, "prev1_>=0.65", "prev1_mid")),
    )

    total_n = len(merged)
    rows: list[dict[str, Any]] = []

    # overall + by fold as a slice
    rec = score_group(
        merged["control_success"].to_numpy(dtype=float),
        merged["prediction"].to_numpy(dtype=float),
    )
    rec.update(
        {
            "slice_family": "overall",
            "slice_level": "concat_oof_2022_2024",
            "scope": "all_oof",
            "row_share": 1.0,
        }
    )
    rows.append(rec)

    for scope_name, frame in [("all_oof", merged)] + [
        (str(fid), part) for fid, part in merged.groupby("fold_id")
    ]:
        n_scope = len(frame)
        append_slices(rows, frame, "season", "season", scope_name, n_scope)
        append_slices(rows, frame, "pitcher_n_bin", "pitcher_n_bin", scope_name, n_scope)
        append_slices(rows, frame, "batter_n_bin", "batter_n_bin", scope_name, n_scope)
        append_slices(rows, frame, "pitcher_cold", "pitcher_cold", scope_name, n_scope)
        append_slices(rows, frame, "batter_cold", "batter_cold", scope_name, n_scope)
        append_slices(rows, frame, "balls_before", "balls_before", scope_name, n_scope)
        append_slices(rows, frame, "strikes_before", "strikes_before", scope_name, n_scope)
        append_slices(rows, frame, "count", "count", scope_name, n_scope)
        append_slices(rows, frame, "inning", "inning", scope_name, n_scope)
        append_slices(rows, frame, "inning_bin", "inning_bin", scope_name, n_scope)
        append_slices(rows, frame, "base_state", "base_state", scope_name, n_scope)
        append_slices(rows, frame, "num_runners_on", "num_runners_on", scope_name, n_scope)
        append_slices(rows, frame, "score_bin", "score_bin", scope_name, n_scope)
        append_slices(rows, frame, "li_bin", "li_bin", scope_name, n_scope)
        append_slices(rows, frame, "hand_matchup", "hand_matchup", scope_name, n_scope)
        append_slices(rows, frame, "prev1_missing", "prev1_missing", scope_name, n_scope)
        append_slices(rows, frame, "prev5_missing", "prev5_missing", scope_name, n_scope)
        append_slices(rows, frame, "prev1_extreme", "prev1_extreme", scope_name, n_scope)

    slice_df = pd.DataFrame(rows)
    slice_df.insert(0, "experiment_id", EXPERIMENT_ID)
    slice_df.to_csv(csv_path, index=False)

    # plots
    fig, ax = plt.subplots(figsize=(7, 4))
    season_all = slice_df.loc[
        (slice_df.slice_family == "season") & (slice_df.scope == "all_oof")
    ].sort_values("slice_level")
    ax.bar(season_all["slice_level"].astype(str), season_all["raw_brier_skill_score"])
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("EXP-004 OOF raw BSS by season")
    ax.set_xlabel("Season")
    ax.set_ylabel("raw BSS")
    fig.tight_layout()
    fig.savefig(plot_dir / "raw_bss_by_season.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    pb = slice_df.loc[
        (slice_df.slice_family == "pitcher_n_bin") & (slice_df.scope == "all_oof")
    ]
    ax.bar(pb["slice_level"].astype(str), pb["raw_brier_skill_score"])
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("EXP-004 OOF raw BSS by asof_pitcher_n bin")
    ax.set_xlabel("asof_pitcher_n bin")
    ax.set_ylabel("raw BSS")
    fig.tight_layout()
    fig.savefig(plot_dir / "raw_bss_by_pitcher_n_bin.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    li = slice_df.loc[(slice_df.slice_family == "li_bin") & (slice_df.scope == "all_oof")]
    ax.bar(li["slice_level"].astype(str), li["raw_brier_skill_score"])
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("EXP-004 OOF raw BSS by LI bin")
    ax.set_xlabel("LI bin")
    ax.set_ylabel("raw BSS")
    fig.tight_layout()
    fig.savefig(plot_dir / "raw_bss_by_li_bin.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ct = slice_df.loc[(slice_df.slice_family == "count") & (slice_df.scope == "all_oof")].sort_values(
        "slice_level"
    )
    ax.plot(ct["slice_level"].astype(str), ct["target_mean"], marker="o", label="target mean")
    ax.plot(ct["slice_level"].astype(str), ct["prediction_mean"], marker="o", label="prediction mean")
    ax.set_title("EXP-004 calibration by count (balls-strikes)")
    ax.set_xlabel("count")
    ax.set_ylabel("rate")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "calibration_by_count.png", dpi=140)
    plt.close(fig)

    def pick(family: str, level: str, scope: str = "all_oof") -> pd.Series | None:
        hit = slice_df.loc[
            (slice_df.slice_family == family)
            & (slice_df.slice_level == str(level))
            & (slice_df.scope == scope)
        ]
        return hit.iloc[0] if len(hit) else None

    def fold_pick(family: str, level: str, fold: str) -> pd.Series | None:
        return pick(family, level, fold)

    # helper numbers for the eight questions
    s2023 = pick("season", "2023")
    s2022 = pick("season", "2022")
    s2024 = pick("season", "2024")
    p0 = pick("pitcher_n_bin", "0")
    p_hi = pick("pitcher_n_bin", "2000+")
    b0 = pick("batter_n_bin", "0")
    li_hi = pick("li_bin", "3+")
    inn_late = pick("inning_bin", "7-9")
    inn_early = pick("inning_bin", "1-3")
    prev_na = pick("prev1_missing", "prev1_NA")
    prev_obs = pick("prev1_missing", "prev1_observed")

    # 2023 concentration: compare pitcher bins / count / li within 2023 vs 2022
    def worst_in(family: str, scope: str, min_n: int = 3000) -> pd.DataFrame:
        sub = slice_df.loc[
            (slice_df.slice_family == family)
            & (slice_df.scope == scope)
            & (slice_df.n >= min_n)
            & (slice_df.eligible)
        ]
        return sub.sort_values("raw_brier_skill_score")

    # failure pattern candidates from all_oof, n>=MIN_ROWS_WARN
    candidates = slice_df.loc[
        (slice_df.scope == "all_oof")
        & (slice_df.slice_family != "overall")
        & (slice_df.slice_family != "season")
        & (slice_df.eligible)
        & (slice_df.n >= MIN_ROWS_WARN)
    ].sort_values("raw_brier_skill_score")

    lines: list[str] = []
    lines.append("# ERR-AUDIT-001")
    lines.append("")
    lines.append("- baseline model: **EXP-004** CatBoost + season sample weights")
    lines.append("- OOF: `outputs/predictions/EXP-004/oof_holdout_predictions.csv`")
    lines.append("- validation: expanding secondary 2022 / 2023 + primary 2024 (`is_comparable_cv=true`)")
    lines.append("- metric: `src.metrics.compute_brier_metrics` (unchanged); quantile-10 ECE copied from VAL-EXP006-SECONDARY-001 for diagnostics only")
    lines.append("- no retraining; existing EXP artifacts were not overwritten")
    lines.append("")
    lines.append("## Why EXP-004, not EXP-006")
    lines.append("")
    lines.append(integrity["baseline_choice_reason"])
    lines.append("")
    lines.append("- EXP-006 vs EXP-005 secondary verdict = **hold** (`VAL-EXP006-SECONDARY-001`).")
    lines.append("- EXP-006 2024 predictions are the Optuna tune holdout, so 2024 error slices would be optimistically biased.")
    lines.append("- EXP-005 has no row-level OOF.")
    lines.append("- VAL-EXP006 parquet exists locally for 2022/2023 only; it is not a comparable 3-fold OOF and was not used as the primary error table.")
    lines.append("")
    lines.append("## 2-1. OOF integrity")
    lines.append("")
    lines.append(
        md_table(
            ["check", "value"],
            [
                ["OOF rows", f"{integrity['oof_n_rows']:,}"],
                ["row_id unique", str(integrity["row_id_unique"])],
                ["duplicate row_id count", str(integrity["n_duplicate_row_id"])],
                ["rows before join", str(integrity["n_before_join"])],
                ["rows after join", str(integrity["n_after_join"])],
                ["unmatched (pitcher_id NA)", str(integrity["unmatched_pitcher_id_na"])],
                ["duplicate increase vs source", str(integrity["duplicate_row_increase"])],
                ["prediction NaN", str(integrity["prediction_nan"])],
                ["prediction inf", str(integrity["prediction_inf"])],
                ["prediction min", fmt(integrity["prediction_min"], 6)],
                ["prediction max", fmt(integrity["prediction_max"], 6)],
                ["prediction outside [0,1]", str(integrity["prediction_outside_01"])],
                ["target alignment vs train", str(integrity["target_alignment_vs_train"])],
                ["folds", ", ".join(integrity["folds_present"])],
                ["seasons", ", ".join(str(s) for s in integrity["seasons_present"])],
            ],
        )
    )
    lines.append("")
    lines.append("Fold raw BSS recomputed with `evaluate_predictions` vs logged EXP-004 summary:")
    lines.append("")
    fold_body = []
    for fid, recs in fold_scores.items():
        fold_body.append(
            [
                fid,
                f"{recs['n']:,}",
                fmt(recs["raw_brier_skill_score"], 6),
                fmt(recs["logged_raw_bss"], 6) if recs["logged_raw_bss"] is not None else "not in map",
                str(recs["matches_logged"]),
            ]
        )
    lines.append(md_table(["fold", "n", "recomputed raw BSS", "logged raw BSS", "match"], fold_body))
    lines.append("")
    if integrity["unmatched_pitcher_id_na"] == 0 and integrity["target_alignment_vs_train"]:
        lines.append("Integrity: **PASS**. OOF is suitable for comparable-CV error analysis.")
    else:
        lines.append("Integrity: **FAIL / incomplete**. See fields above; slice metrics below may be unreliable.")
    lines.append("")
    lines.append("## 2-2 / 2-3. Slice metrics")
    lines.append("")
    lines.append("Raw BSS is the ranking diagnostic when official BSS is clipped at 0. Groups with n < 2,000 are flagged.")
    lines.append("")
    lines.append("### Season")
    lines.append("")
    lines.append(slice_table(slice_df, "season", "all_oof"))
    lines.append("")
    lines.append("### Pitcher sample size (`asof_pitcher_n`)")
    lines.append("")
    lines.append(slice_table(slice_df, "pitcher_n_bin", "all_oof"))
    lines.append("")
    lines.append("2023 only:")
    lines.append("")
    lines.append(slice_table(slice_df, "pitcher_n_bin", "secondary_2023"))
    lines.append("")
    lines.append("### Batter sample size (`asof_batter_n`)")
    lines.append("")
    lines.append(slice_table(slice_df, "batter_n_bin", "all_oof"))
    lines.append("")
    lines.append("### Count")
    lines.append("")
    lines.append(slice_table(slice_df, "count", "all_oof"))
    lines.append("")
    lines.append("### LI")
    lines.append("")
    lines.append(slice_table(slice_df, "li_bin", "all_oof"))
    lines.append("")
    lines.append("### Inning")
    lines.append("")
    lines.append(slice_table(slice_df, "inning_bin", "all_oof"))
    lines.append("")
    lines.append("### Base / score / hands / recent history")
    lines.append("")
    lines.append("num_runners_on:")
    lines.append("")
    lines.append(slice_table(slice_df, "num_runners_on", "all_oof"))
    lines.append("")
    lines.append("score_diff_pitcher_team bins:")
    lines.append("")
    lines.append(slice_table(slice_df, "score_bin", "all_oof"))
    lines.append("")
    lines.append("pitcher_hand × batter_hand:")
    lines.append("")
    lines.append(slice_table(slice_df, "hand_matchup", "all_oof"))
    lines.append("")
    lines.append("prev1 missing vs observed:")
    lines.append("")
    lines.append(slice_table(slice_df, "prev1_missing", "all_oof"))
    lines.append("")
    lines.append("prev1 extreme bins:")
    lines.append("")
    lines.append(slice_table(slice_df, "prev1_extreme", "all_oof"))
    lines.append("")

    lines.append("## 2-4. Failure patterns")
    lines.append("")

    def fail_block(fid: str, title: str, evidence: str, share: str, cause: str, feat: str, val: str, extra: str) -> None:
        lines.append(f"### {fid} — {title}")
        lines.append("")
        lines.append(f"- 현상: {title}")
        lines.append(f"- 근거: {evidence}")
        lines.append(f"- 영향받는 행 비율: {share}")
        lines.append(f"- 가능한 원인: {cause}")
        lines.append(f"- feature 가설: {feat}")
        lines.append(f"- validation 문제 가능성: {val}")
        lines.append(f"- 추가 검증 필요 여부: {extra}")
        lines.append("")

    # Build evidence from actual numbers
    fail_block(
        "FAILURE-01",
        "2023 OOF is globally worse than climatology, not a tiny subgroup",
        (
            f"2023 raw BSS={fmt(s2023['raw_brier_skill_score'],1) if s2023 is not None else 'missing'} "
            f"(official 0); 2022={fmt(s2022['raw_brier_skill_score'],1) if s2022 is not None else 'missing'}; "
            f"2024={fmt(s2024['raw_brier_skill_score'],1) if s2024 is not None else 'missing'}. "
            "This reproduces the logged EXP-004 secondary_2023 failure."
        ),
        f"{fmt(s2023['row_share'],3) if s2023 is not None else 'missing'} of concatenated OOF (the entire 2023 fold)",
        "seasonal target-rate shift plus feature–target sign changes (ball/strike asof rates) documented in FEATURE-STABILITY-AUDIT / target_regime_check",
        "prefer features whose 2019–2024 association does not flip; avoid leaning on SIGN_FLIP asof rates",
        "validation itself is the intended stress test; do not retune validation to hide 2023",
        "yes — confirm whether 2023 failure is uniform across count/LI/cold-start (tables above)",
    )
    fail_block(
        "FAILURE-02",
        "Pitcher cold-start / very low asof_n",
        (
            f"asof_pitcher_n==0 raw BSS={fmt(p0['raw_brier_skill_score'],1) if p0 is not None else 'missing'} "
            f"n={fmt(p0['n'],0) if p0 is not None else 'missing'}; "
            f"2000+ raw BSS={fmt(p_hi['raw_brier_skill_score'],1) if p_hi is not None else 'missing'}. "
            f"pred std at n==0={fmt(p0['prediction_std'],4) if p0 is not None else 'missing'}, "
            f"frac extreme={fmt(p0['frac_pred_extreme'],4) if p0 is not None else 'missing'}."
        ),
        f"{fmt(p0['row_share'],4) if p0 is not None else 'missing'} of OOF",
        "asof rates missing when n=0; model must fall back to global/context features",
        "explicit cold-start flag or stronger shrinkage toward league/count prior when asof_pitcher_n is tiny",
        "low — this is a feature-support issue, not a split bug",
        "yes if the n==0 slice is large enough in 2025; sample share is reported above",
    )
    fail_block(
        "FAILURE-03",
        "Batter cold-start",
        (
            f"asof_batter_n==0 raw BSS={fmt(b0['raw_brier_skill_score'],1) if b0 is not None else 'missing'} "
            f"n={fmt(b0['n'],0) if b0 is not None else 'missing'}."
        ),
        f"{fmt(b0['row_share'],4) if b0 is not None else 'missing'} of OOF",
        "same structural NA on batter asof rates",
        "batter cold-start flag / count-level prior; likely weaker than pitcher because batter asof effects are small",
        "low",
        "only if slice n is not tiny; otherwise deprioritize",
    )
    fail_block(
        "FAILURE-04",
        "Count-level miscalibration",
        "See count table: compare prediction mean vs target mean especially 3-0 / 3-2 / 0-2. Systematic gap (not just noise) is the failure.",
        "count is defined on 100% of rows; the badly calibrated cells are the ones with large |gap| in the table",
        "count is a stable context feature (stability audit) but the model may under-use it relative to drifting asof rates",
        "count × hand interaction or count-specific calibration; keep it fold-safe (row-local, no test aggregates)",
        "low",
        "yes — rank counts by |calibration_gap| and raw BSS",
    )
    fail_block(
        "FAILURE-05",
        "High LI / late inning",
        (
            f"LI 3+ raw BSS={fmt(li_hi['raw_brier_skill_score'],1) if li_hi is not None else 'missing'} "
            f"n={fmt(li_hi['n'],0) if li_hi is not None else 'missing'}; "
            f"innings 7-9 raw BSS={fmt(inn_late['raw_brier_skill_score'],1) if inn_late is not None else 'missing'} "
            f"vs 1-3={fmt(inn_early['raw_brier_skill_score'],1) if inn_early is not None else 'missing'}."
        ),
        f"LI3+ share={fmt(li_hi['row_share'],3) if li_hi is not None else 'missing'}; "
        f"7-9 share={fmt(inn_late['row_share'],3) if inn_late is not None else 'missing'}",
        "LI is excluded from the model (leakage review). High-leverage rows may still be predictable from count/score/inning. Late innings also shift pitcher mix.",
        "do not add `li` until construction is confirmed; consider allowed context (inning, score_diff, count) interactions instead",
        "using `li` as a feature would be a policy change, not an error-analysis fix",
        "yes — compare inning/score slices which are already in the model vs LI which is not",
    )
    fail_block(
        "FAILURE-06",
        "Prev-game asof missing / extreme",
        (
            f"prev1_NA raw BSS={fmt(prev_na['raw_brier_skill_score'],1) if prev_na is not None else 'missing'} "
            f"vs observed={fmt(prev_obs['raw_brier_skill_score'],1) if prev_obs is not None else 'missing'}."
        ),
        f"prev1_NA share={fmt(prev_na['row_share'],3) if prev_na is not None else 'missing'}",
        "prev* features are sparse at pitcher-season start; extremes of prev1 success may be small-sample noise",
        "missing indicator + shrink prev rates; do not treat prev1 as a current-pitch signal",
        "low (asof construction already audited as no mid-season NA re-entry)",
        "connect to FEATURE-STABILITY-AUDIT SPARSE/COLD_START labels",
    )

    lines.append("## Answers to the required questions")
    lines.append("")
    lines.append(
        "1. **Is 2023 degradation concentrated in a subgroup?** "
        "The whole 2023 fold is negative raw BSS. Subgroup tables for 2023 show whether some bins are worse, "
        "but there is no evidence that 2023 failure is *only* a small slice — it is a season-wide regime problem "
        "with possible extra concentration in cold-start / high-LI cells (see 2023 pitcher_n_bin table)."
    )
    lines.append(
        f"2. **Cold-start pitcher?** n==0 share={fmt(p0['row_share'],4) if p0 is not None else 'missing'}, "
        f"raw BSS={fmt(p0['raw_brier_skill_score'],1) if p0 is not None else 'missing'} vs 2000+ "
        f"{fmt(p_hi['raw_brier_skill_score'],1) if p_hi is not None else 'missing'}. "
        "If n==0 is worse, yes; if similar, cold-start is not the main 2023 story."
    )
    lines.append(
        f"3. **Cold-start batter?** n==0 raw BSS={fmt(b0['raw_brier_skill_score'],1) if b0 is not None else 'missing'}, "
        f"share={fmt(b0['row_share'],4) if b0 is not None else 'missing'}."
    )
    lines.append("4. **Systematic count miscalibration?** See count table gaps (p_mean − y_mean). Non-zero gaps that persist across seasons are systematic.")
    lines.append(
        f"5. **High LI worse?** LI 3+ raw BSS={fmt(li_hi['raw_brier_skill_score'],1) if li_hi is not None else 'missing'} "
        f"vs overall concat {fmt(rec['raw_brier_skill_score'],1)}."
    )
    lines.append(
        f"6. **Late game worse?** innings 7-9 raw BSS={fmt(inn_late['raw_brier_skill_score'],1) if inn_late is not None else 'missing'} "
        f"vs 1-3 {fmt(inn_early['raw_brier_skill_score'],1) if inn_early is not None else 'missing'}."
    )
    lines.append(
        f"7. **Extreme predictions at small asof_n?** n==0 pred std={fmt(p0['prediction_std'],4) if p0 is not None else 'missing'}, "
        f"min={fmt(p0['prediction_min'],4) if p0 is not None else 'missing'}, "
        f"max={fmt(p0['prediction_max'],4) if p0 is not None else 'missing'}, "
        f"frac extreme={fmt(p0['frac_pred_extreme'],4) if p0 is not None else 'missing'} "
        f"vs 2000+ std={fmt(p_hi['prediction_std'],4) if p_hi is not None else 'missing'}."
    )
    lines.append(
        "8. **Link to feature instability?** 2023 is both the SIGN_FLIP year for ball/strike asof rates and the catastrophic OOF fold. "
        "That is circumstantial, not a causal proof. Count features are more stable and still show calibration gaps, so FE on stable context is safer than doubling down on flipping asof rates."
    )
    lines.append("")
    lines.append("Lowest raw BSS slices (all_oof, n≥2000, excluding overall/season):")
    lines.append("")
    low = candidates.head(12)
    lines.append(
        md_table(
            ["family", "level", "n", "share", "raw BSS", "gap"],
            [
                [
                    str(r.slice_family),
                    str(r.slice_level),
                    f"{int(r.n):,}",
                    fmt(r.row_share, 3),
                    fmt(r.raw_brier_skill_score, 1),
                    fmt(r.calibration_gap, 4),
                ]
                for r in low.itertuples()
            ],
        )
    )
    lines.append("")
    lines.append("## [FEATURE HYPOTHESES]")
    lines.append("")
    lines.append("Maximum five. None implemented in this audit.")
    lines.append("")

    hypotheses = [
        {
            "hypothesis_id": "FEH-01",
            "관찰": "2023 fold fails globally; ball/strike asof rates SIGN_FLIP in the stability audit.",
            "제안 feature": "Do not add more raw asof ball/strike rates. Candidate: recency-capped or season-relative pitcher asof (already probed in M/N diagnostics — re-check rather than invent).",
            "예상 효과": "Reduce 2023-style regime damage if the model currently trusts flipped asof rates.",
            "leakage 위험": "low if still official asof or strictly past-only; high if any current-pitch or test-wide aggregate",
            "구현 난이도": "medium",
            "검증 방법": "same expanding folds as EXP-004; compare raw BSS 2023 and 2024; one change only",
            "우선순위": "high if M/N did not already exhaust this; otherwise HOLD",
        },
        {
            "hypothesis_id": "FEH-02",
            "관찰": "Count is row-local, stable, and shows calibration gaps; 3-ball counts have lower target rate.",
            "제안 feature": "Explicit count interaction with pitcher_hand (and optionally batter_hand), still row-local.",
            "예상 효과": "Better calibration on 3-0/3-2 without using future information.",
            "leakage 위험": "low (pre-pitch fields already in train/test rows)",
            "구현 난이도": "low",
            "검증 방법": "add one interaction feature set on EXP-004 recipe; same CV",
            "우선순위": "high",
        },
        {
            "hypothesis_id": "FEH-03",
            "관찰": "asof_pitcher_n==0 / prev* NA are structural cold-starts.",
            "제안 feature": "Missing/cold-start indicators plus shrinkage of asof rates toward count-level mean computed on train-fold rows only.",
            "예상 효과": "Less noisy predictions for first-pitcher-season rows.",
            "leakage 위험": "medium if shrinkage uses the full training table without fold isolation; must be fold-safe and never use test rows",
            "구현 난이도": "medium",
            "검증 방법": "slice raw BSS on pitcher_n_bin=0 and prev NA; overall 2023/2024 must not degrade",
            "우선순위": "medium-high",
        },
        {
            "hypothesis_id": "FEH-04",
            "관찰": "High LI / late innings may be harder; `li` is excluded.",
            "제안 feature": "Allowed context only: inning × score_diff_pitcher_team (no WE/LI).",
            "예상 효과": "Capture leverage-like situations without unverified WE/LI construction.",
            "leakage 위험": "low for official context fields; still do not add `li`/`WE`",
            "구현 난이도": "low",
            "검증 방법": "compare inning_bin and score_bin slices; reject if 2024/2023 overall raw BSS does not move",
            "우선순위": "medium",
        },
        {
            "hypothesis_id": "FEH-05",
            "관찰": "Pitch mix asof exists; league mix shifted 2022+ in Trackman margins.",
            "제안 feature": "Keep asof pitchmix; optionally combine with TM-001/002 if those audits PASS.",
            "예상 효과": "Help 2023 mix regime without player TM matching.",
            "leakage 위험": "low for asof mix; TM must be prior-season only (see TM-AUDIT-001B)",
            "구현 난이도": "low for mix-only; high if TM join logic is added",
            "검증 방법": "TM feasibility first; do not implement TM in this audit",
            "우선순위": "depends on TM verdict",
        },
    ]
    for h in hypotheses:
        lines.append(f"### {h['hypothesis_id']}")
        lines.append("")
        for k, v in h.items():
            if k == "hypothesis_id":
                continue
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Appendix")
    lines.append("")
    lines.append(f"- slices CSV: `{csv_path.relative_to(ROOT).as_posix()}`")
    lines.append("- plots under `outputs/analysis/error_audit_001/`")
    lines.append("- ECE is diagnostic only and was not used to select a model")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(integrity, ensure_ascii=False, indent=2, default=str))
    lines.append("```")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    integrity_path = out_dir / "error_audit_001_integrity.json"
    integrity_path.write_text(json.dumps(integrity, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
