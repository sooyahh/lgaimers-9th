"""TM-002A: add tm_expected_rel_speed to the EXP-004 CatBoost recipe.

Prior-season Trackman group mean rel_speed x current-row as-of pitch mix.
Does not overwrite EXP-004 / FE-001A / FE-002A artifacts. No Public LB submit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exp004_catboost_weighted import fit_catboost
from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from scripts.tree_exp_common import build_feature_lists, run_tree_experiment
from src.metrics import compute_brier_metrics
from src.trackman_profiles import (
    MIX_RATE_COLS,
    REL_SPEED_SUMMARY,
    TM_EXPECTED_REL_SPEED,
    allowed_trackman_seasons,
    build_group_rel_speed_profile,
    compute_tm_expected_rel_speed,
    filter_trackman_history,
    profile_means,
)
from src.utils import append_experiment_log, ensure_dir, load_config, save_json, to_json_text

EXPERIMENT_ID = "TM-002A"
BASELINE_ID = "EXP-004"
NEW_FEATURE = TM_EXPECTED_REL_SPEED
TRACKMAN_PATH = ROOT / "data/raw/trackman_history.csv"
EXPECTED_BASELINE = {
    "secondary_2022": 2259.3983277146012,
    "secondary_2023": -1366.4810668740479,
    "primary_2024": 578.1844155969828,
}
ASOF_BINS = [-0.5, 0.5, 100, 500, 2000, np.inf]
ASOF_LABELS = ["0", "1-100", "101-500", "501-2000", "2000+"]
PROTECTED = [
    ROOT / "outputs/predictions/EXP-004/summary.json",
    ROOT / "outputs/predictions/EXP-004/fold_metrics.csv",
    ROOT / "outputs/predictions/EXP-004/oof_holdout_predictions.csv",
    ROOT / "outputs/predictions/EXP-004/report.md",
    ROOT / "outputs/predictions/FE-001A/summary.json",
    ROOT / "outputs/predictions/FE-001A/oof_holdout_predictions.csv",
    ROOT / "outputs/predictions/FE-001A/fold_metrics.csv",
    ROOT / "outputs/predictions/FE-002A/summary.json",
    ROOT / "outputs/predictions/FE-002A/oof_holdout_predictions.csv",
    ROOT / "outputs/predictions/FE-002A/fold_metrics.csv",
]


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    st = path.stat()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": True,
        "nbytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime_ns": st.st_mtime_ns,
    }


def assert_configs_match(baseline_cfg: dict[str, Any], tm_cfg: dict[str, Any]) -> None:
    for key in ("cv", "task", "runtime"):
        if baseline_cfg[key] != tm_cfg[key]:
            raise AssertionError(f"config mismatch in {key}")
    if baseline_cfg["models"]["params"]["catboost"] != tm_cfg["models"]["params"]["catboost"]:
        raise AssertionError("CatBoost params mismatch vs baseline.yaml")
    if baseline_cfg["features"]["exclude_cols"] != tm_cfg["features"]["exclude_cols"]:
        raise AssertionError("exclude_cols mismatch; TM-002A must not drop EXP-004 features")
    if baseline_cfg["features"]["categorical_cols"] != tm_cfg["features"]["categorical_cols"]:
        raise AssertionError("categorical_cols mismatch")


def load_baseline_folds(path: Path) -> dict[str, dict[str, Any]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in summary["fold_metrics"]:
        out[row["fold_id"]] = row
        logged = EXPECTED_BASELINE.get(row["fold_id"])
        if logged is None:
            continue
        got = float(row["raw_brier_skill_score"])
        if abs(got - logged) > 1e-8:
            raise AssertionError(
                f"EXP-004 {row['fold_id']} raw BSS {got} != logged {logged}."
            )
    return out


def quantile_10_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Same definition as scripts/run_error_audit_001.py (diagnostic only)."""
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


def score_slice(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    bundle = compute_brier_metrics(y, p)
    return {
        "n": int(len(y)),
        "target_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(p)),
        "calibration_gap": float(np.mean(p) - np.mean(y)),
        "brier_score": float(bundle["brier_score"]),
        "raw_brier_skill_score": float(bundle["raw_brier_skill_score"]),
        "official_brier_skill_score": float(bundle["official_brier_skill_score"]),
        "quantile_10_ece": quantile_10_ece(y, p),
    }


def asof_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float).fillna(0.0),
        bins=ASOF_BINS,
        labels=ASOF_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)


def feature_distribution(values: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(values, errors="coerce")
    observed = s.dropna()
    out = {
        "n": int(len(s)),
        "missing_rate": float(s.isna().mean()),
        "mean": float("nan"),
        "std": float("nan"),
        "p10": float("nan"),
        "median": float("nan"),
        "p90": float("nan"),
    }
    if observed.empty:
        return out
    out.update(
        {
            "mean": float(observed.mean()),
            "std": float(observed.std(ddof=0)),
            "p10": float(observed.quantile(0.10)),
            "median": float(observed.median()),
            "p90": float(observed.quantile(0.90)),
        }
    )
    return out


def make_fold_extra_features(
    tm: pd.DataFrame,
    profile_store: list[dict[str, Any]],
    diag_store: list[dict[str, Any]],
    corr_store: list[dict[str, Any]],
):
    def fold_extra_features(
        train_df: pd.DataFrame, spec: dict[str, Any]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if len(spec["valid_seasons"]) != 1:
            raise AssertionError("TM-002A expects a single valid season per fold")
        valid_season = int(spec["valid_seasons"][0])
        allowed = list(allowed_trackman_seasons(valid_season))
        hist = filter_trackman_history(tm, valid_season)
        used_seasons = sorted(int(s) for s in hist["season"].dropna().unique())
        if set(used_seasons) - set(allowed):
            raise AssertionError(
                f"{spec['fold_id']} used TM seasons {used_seasons} outside allowed {allowed}"
            )
        if any(s >= valid_season for s in used_seasons):
            raise AssertionError(f"{spec['fold_id']} TM profile includes valid/future seasons")
        if 2025 in used_seasons:
            raise AssertionError(f"{spec['fold_id']} used 2025 Trackman")
        profile = build_group_rel_speed_profile(hist, valid_season=valid_season)
        means = profile_means(profile)
        extra = pd.DataFrame(
            {NEW_FEATURE: compute_tm_expected_rel_speed(train_df, means)},
            index=train_df.index,
        )
        train_mask = train_df["season"].isin(spec["train_seasons"])
        valid_mask = train_df["season"].isin(spec["valid_seasons"])
        feat = extra[NEW_FEATURE]
        for split, mask in (("train", train_mask), ("valid", valid_mask)):
            dist = feature_distribution(feat.loc[mask])
            diag_store.append(
                {
                    "fold_id": spec["fold_id"],
                    "valid_season": valid_season,
                    "split": split,
                    **dist,
                }
            )
        valid_frame = pd.concat(
            [
                train_df.loc[valid_mask, list(MIX_RATE_COLS)],
                extra.loc[valid_mask, [NEW_FEATURE]],
            ],
            axis=1,
        )
        corr = valid_frame.corr(numeric_only=True)[NEW_FEATURE]
        for col in MIX_RATE_COLS:
            corr_store.append(
                {
                    "fold_id": spec["fold_id"],
                    "valid_season": valid_season,
                    "mix_feature": col,
                    "pearson": float(corr[col]) if pd.notna(corr[col]) else float("nan"),
                    "n": int(valid_frame[[col, NEW_FEATURE]].dropna().shape[0]),
                }
            )
        for row in profile.itertuples():
            profile_store.append(
                {
                    "fold_id": spec["fold_id"],
                    "valid_season": valid_season,
                    "tm_cutoff": valid_season,
                    "tm_seasons": ",".join(str(s) for s in used_seasons),
                    "group": row.group,
                    "n": int(row.n),
                    "n_missing": int(row.n_missing),
                    "missing_rate": float(row.missing_rate),
                    "rel_speed_mean": float(row.rel_speed_mean),
                    "rel_speed_std": float(row.rel_speed_std),
                    "summary": REL_SPEED_SUMMARY,
                }
            )
        meta = {
            "valid_season": valid_season,
            "tm_cutoff": f"season < {valid_season}",
            "tm_seasons": used_seasons,
            "profile_summary": REL_SPEED_SUMMARY,
            "v_fastball": means["fastball"],
            "v_breaking": means["breaking"],
            "v_offspeed": means["offspeed"],
            "feature_missing_rate_train": float(feat.loc[train_mask].isna().mean()),
            "feature_missing_rate_valid": float(feat.loc[valid_mask].isna().mean()),
        }
        return extra, meta

    return fold_extra_features


def compare_slices(base_oof: pd.DataFrame, tm_oof: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "row_id",
        "asof_pitcher_n",
        "asof_pitcher_prev1_game_success_rate",
    ]
    ctx = train[cols]
    base = base_oof.merge(ctx, on="row_id", how="left", validate="one_to_one")
    tm = tm_oof.merge(ctx, on="row_id", how="left", validate="one_to_one")
    if set(base["row_id"]) != set(tm["row_id"]):
        raise AssertionError("OOF row_id sets differ")
    tm = tm.set_index("row_id").loc[base["row_id"]].reset_index()
    base = base.reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    def add(family: str, level: str, bpart: pd.DataFrame, tpart: pd.DataFrame) -> None:
        rec: dict[str, Any] = {
            "slice_family": family,
            "slice_level": level,
            "n_exp004": int(len(bpart)),
            "n_tm002a": int(len(tpart)),
        }
        if len(bpart) < 30 or len(tpart) < 30:
            rec["reason"] = "below_min_rows:30"
            rows.append(rec)
            return
        b = score_slice(bpart["control_success"].to_numpy(float), bpart["prediction"].to_numpy(float))
        t = score_slice(tpart["control_success"].to_numpy(float), tpart["prediction"].to_numpy(float))
        rec.update(
            {
                "exp004_raw_bss": b["raw_brier_skill_score"],
                "tm002a_raw_bss": t["raw_brier_skill_score"],
                "delta_raw_bss": t["raw_brier_skill_score"] - b["raw_brier_skill_score"],
                "exp004_pred_mean": b["prediction_mean"],
                "tm002a_pred_mean": t["prediction_mean"],
                "exp004_target_mean": b["target_mean"],
                "tm002a_target_mean": t["target_mean"],
                "exp004_gap": b["calibration_gap"],
                "tm002a_gap": t["calibration_gap"],
                "delta_gap": t["calibration_gap"] - b["calibration_gap"],
                "exp004_ece10q": b["quantile_10_ece"],
                "tm002a_ece10q": t["quantile_10_ece"],
                "reason": "ok",
            }
        )
        rows.append(rec)

    for season, bpart in base.groupby("season"):
        add("season", str(int(season)), bpart, tm.loc[tm["season"] == season])

    b_low = base.loc[base["asof_pitcher_prev1_game_success_rate"] <= 0.35]
    t_low = tm.loc[tm["asof_pitcher_prev1_game_success_rate"] <= 0.35]
    add("prev1", "prev1_<=0.35", b_low, t_low)

    base["pitcher_n_bin"] = asof_bin(base["asof_pitcher_n"])
    tm["pitcher_n_bin"] = asof_bin(tm["asof_pitcher_n"])
    for level in ASOF_LABELS:
        add(
            "asof_pitcher_n",
            str(level),
            base.loc[base["pitcher_n_bin"] == level],
            tm.loc[tm["pitcher_n_bin"] == level],
        )
    return pd.DataFrame(rows)


def decide_verdict(
    deltas: dict[str, float],
    *,
    leakage_ok: bool,
    near_clone: bool,
) -> tuple[str, str]:
    if not leakage_ok:
        return "REJECT", "Leakage-safe Trackman cutoff failed."
    d22, d23, d24 = deltas["d2022"], deltas["d2023"], deltas["d2024"]
    dmean, dworst = deltas["dmean"], deltas["dworst"]
    fold_deltas = [d22, d23, d24]
    n_improved = sum(d > 0 for d in fold_deltas)
    n_worse = sum(d < 0 for d in fold_deltas)
    numerically_flat = all(abs(d) < 1.0 for d in fold_deltas)
    if numerically_flat:
        return (
            "REJECT",
            "Feature adds essentially no ranking signal versus EXP-004 "
            "(fold raw BSS unchanged).",
        )
    clone_note = (
        " Mix-rate correlation is high, so the feature may be a physics-weighted remix."
        if near_clone
        else ""
    )
    if dmean < 0:
        return "REJECT", "CV mean raw BSS worsened versus EXP-004." + clone_note
    if n_worse >= 2:
        return "REJECT", "A majority of temporal folds worsened versus EXP-004." + clone_note
    if dmean > 0 and n_improved >= 2 and dworst >= 0:
        return (
            "ADOPT_CANDIDATE",
            "CV mean improved, the gain is not a single-fold artifact, "
            "and worst-fold raw BSS did not deteriorate."
            + clone_note,
        )
    if dmean > 0 and n_improved == 1:
        return "HOLD", "CV mean improved but the gain depends on a single fold." + clone_note
    if dworst < 0:
        return "HOLD", "Fold trade-off: worst fold worsened even if mean did not." + clone_note
    return "HOLD", "Mixed or modest temporal deltas; keep as a review hold." + clone_note


def write_report(**kwargs: Any) -> None:
    baseline_folds = kwargs["baseline_folds"]
    tm_folds = kwargs["tm_folds"]
    slice_df = kwargs["slice_df"]
    profile_df = kwargs["profile_df"]
    diag_df = kwargs["diag_df"]
    corr_df = kwargs["corr_df"]
    n_base_feat = kwargs["n_base_feat"]
    n_tm_feat = kwargs["n_tm_feat"]
    feature_assert = kwargs["feature_assert"]
    verdict = kwargs["verdict"]
    verdict_reason = kwargs["verdict_reason"]
    fp_ok = kwargs["fp_ok"]
    leakage_ok = kwargs["leakage_ok"]
    report_path = kwargs["report_path"]
    missing_rate = kwargs["missing_rate"]

    fmap = {r["fold_id"]: r for r in tm_folds}
    order = ["secondary_2022", "secondary_2023", "primary_2024"]
    labels = {"secondary_2022": "2022", "secondary_2023": "2023", "primary_2024": "2024"}
    b_raw = [float(baseline_folds[k]["raw_brier_skill_score"]) for k in order]
    t_raw = [float(fmap[k]["raw_brier_skill_score"]) for k in order]
    b_mean, t_mean = float(np.mean(b_raw)), float(np.mean(t_raw))
    b_std, t_std = float(np.std(b_raw, ddof=0)), float(np.std(t_raw, ddof=0))
    b_worst, t_worst = float(min(b_raw)), float(min(t_raw))

    def profile_val(fold_id: str, group: str) -> str:
        sub = profile_df.loc[
            (profile_df["fold_id"] == fold_id) & (profile_df["group"] == group)
        ]
        if sub.empty:
            return "missing"
        return f"{float(sub['rel_speed_mean'].iloc[0]):.4f}"

    def slice_table(family: str) -> str:
        lines = [
            "| slice | n | EXP-004 raw BSS | TM-002A raw BSS | Δ BSS | EXP-004 gap | TM-002A gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        sub = slice_df.loc[slice_df["slice_family"] == family]
        for r in sub.itertuples():
            if getattr(r, "reason", "ok") != "ok":
                lines.append(
                    f"| {r.slice_level} | {int(r.n_exp004):,} | missing | missing | missing | missing | missing |"
                )
                continue
            lines.append(
                f"| {r.slice_level} | {int(r.n_exp004):,} | {r.exp004_raw_bss:.1f} | "
                f"{r.tm002a_raw_bss:.1f} | {r.delta_raw_bss:+.1f} | {r.exp004_gap:+.4f} | {r.tm002a_gap:+.4f} |"
            )
        return "\n".join(lines)

    lines = [
        "# TM-002A Expected Velocity",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        f"- baseline: `{BASELINE_ID}` (existing artifacts reused; not overwritten)",
        "",
        "## 1. Hypothesis",
        "",
        "Main as-of pitch mix rates do not carry Trackman physical velocity scale. "
        "A fold-safe expected rel_speed from prior-season Trackman group means "
        "combined with the current row's as-of mix may add orthogonal information "
        "for temporal CV. This experiment does **not** require fixing 2023 to succeed.",
        "",
        "## 2. What changed",
        "",
        f"- Added exactly `{NEW_FEATURE}`",
        f"- Feature count: {n_base_feat} → {n_tm_feat}",
        f"- TM-002A − EXP-004 = `{feature_assert['tm_minus_baseline']}`",
        f"- EXP-004 − TM-002A = `{feature_assert['baseline_minus_tm']}`",
        f"- Trackman summary: `{REL_SPEED_SUMMARY}(rel_speed)` per core group; `other` excluded",
        "",
        "## 3. What stayed fixed",
        "",
        "- CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss",
        "- seed 42",
        "- season weights 2019:1.0 … 2023:2.0",
        "- expanding 2022/2023/2024 holdouts",
        "- `src.metrics` unchanged",
        "- fold-local median impute + native cats",
        "- no current-pitch Trackman, no player-ID join, no 2025 TM, no test aggregates",
        "- no spin / IVB / HB / extension / cluster bundle",
        "",
        "## 4. Formula",
        "",
        "```",
        "numerator = r_fastball*v_fastball + r_breaking*v_breaking + r_offspeed*v_offspeed",
        "denominator = r_fastball + r_breaking + r_offspeed",
        "tm_expected_rel_speed = numerator / denominator   if denominator > 0",
        "tm_expected_rel_speed = NaN                        otherwise",
        "```",
        "",
        "Observed as-of mix rates already sum to 1 (float noise ~1e-6), so the "
        "denominator is a numerical guard, not a hidden `other` residual. "
        "Cold-start mix NA (`asof_pitcher_pitchmix_n==0`) stays NaN until fold-local median impute.",
        "",
        "## 5. Baseline verification",
        "",
        "EXP-004 was not retrained. Logged fold raw BSS match expected constants to 1e-8.",
        "",
        "| Fold | n_valid | target | pred | raw BSS | official | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        r = baseline_folds[k]
        lines.append(
            f"| {labels[k]} | {int(r['n_valid']):,} | {r['valid_target_rate']:.6f} | "
            f"{r['prediction_mean']:.6f} | {r['raw_brier_skill_score']:.6f} | "
            f"{r['official_brier_skill_score']:.6f} | {r['brier_score']:.6f} |"
        )
    lines += [
        "",
        "## 6. Fold-specific Trackman profiles",
        "",
        "| Valid | TM seasons used | fastball mean | breaking mean | offspeed mean |",
        "|---|---|---:|---:|---:|",
    ]
    for k in order:
        extra = fmap[k].get("fold_extra") or {}
        seasons = extra.get("tm_seasons", "")
        if isinstance(seasons, list):
            seasons = ",".join(str(s) for s in seasons)
        lines.append(
            f"| {labels[k]} | {seasons} | {profile_val(k, 'fastball')} | "
            f"{profile_val(k, 'breaking')} | {profile_val(k, 'offspeed')} |"
        )
    lines += [
        "",
        "## 7. Fold results",
        "",
        "| Fold | n_train | n_valid | target | pred | gap | raw BSS | official | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        r = fmap[k]
        gap = float(r["prediction_mean"]) - float(r["valid_target_rate"])
        lines.append(
            f"| {labels[k]} | {int(r['n_train']):,} | {int(r['n_valid']):,} | "
            f"{r['valid_target_rate']:.6f} | {r['prediction_mean']:.6f} | {gap:+.6f} | "
            f"{r['raw_brier_skill_score']:.6f} | {r['official_brier_skill_score']:.6f} | "
            f"{r['brier_score']:.6f} |"
        )
    lines += [
        "",
        f"- TM-002A mean (ddof=0): `{t_mean:.6f}` (EXP-004 `{b_mean:.6f}`)",
        f"- TM-002A std (ddof=0): `{t_std:.6f}` (EXP-004 `{b_std:.6f}`)",
        f"- TM-002A worst: `{t_worst:.6f}` (EXP-004 `{b_worst:.6f}`)",
        "",
        "## 8. Delta vs EXP-004",
        "",
        "| Fold | EXP-004 | TM-002A | Delta |",
        "|---|---:|---:|---:|",
    ]
    for k, br, tr in zip(order, b_raw, t_raw):
        lines.append(f"| {labels[k]} | {br:.6f} | {tr:.6f} | {tr - br:+.6f} |")
    lines += [
        f"| Mean | {b_mean:.6f} | {t_mean:.6f} | {t_mean - b_mean:+.6f} |",
        f"| Std | {b_std:.6f} | {t_std:.6f} | {t_std - b_std:+.6f} |",
        f"| Worst fold | {b_worst:.6f} | {t_worst:.6f} | {t_worst - b_worst:+.6f} |",
        "",
        "## 9. Calibration / 2023 gap",
        "",
        "| Fold | target | EXP-004 pred | EXP-004 gap | TM-002A pred | TM-002A gap | Δ gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        b, t = baseline_folds[k], fmap[k]
        bg = float(b["prediction_mean"]) - float(b["valid_target_rate"])
        tg = float(t["prediction_mean"]) - float(t["valid_target_rate"])
        lines.append(
            f"| {labels[k]} | {b['valid_target_rate']:.6f} | {b['prediction_mean']:.6f} | "
            f"{bg:+.6f} | {t['prediction_mean']:.6f} | {tg:+.6f} | {tg - bg:+.6f} |"
        )
    lines += [
        "",
        f"- overall `{NEW_FEATURE}` missing rate (valid OOF rows, pre-impute): `{missing_rate:.6f}`",
        "",
        "## 10. Feature distribution (valid split, pre-impute)",
        "",
        "| Fold | n | missing | mean | std | p10 | median | p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        sub = diag_df.loc[(diag_df["fold_id"] == k) & (diag_df["split"] == "valid")]
        if sub.empty:
            lines.append(f"| {labels[k]} | missing | | | | | | |")
            continue
        r = sub.iloc[0]
        lines.append(
            f"| {labels[k]} | {int(r['n']):,} | {r['missing_rate']:.6f} | {r['mean']:.4f} | "
            f"{r['std']:.4f} | {r['p10']:.4f} | {r['median']:.4f} | {r['p90']:.4f} |"
        )
    lines += [
        "",
        "## 11. Correlation with existing mix rates (valid rows)",
        "",
        "| Fold | fastball | breaking | offspeed |",
        "|---|---:|---:|---:|",
    ]
    for k in order:
        vals = {}
        for col in MIX_RATE_COLS:
            sub = corr_df.loc[(corr_df["fold_id"] == k) & (corr_df["mix_feature"] == col)]
            vals[col] = float(sub["pearson"].iloc[0]) if len(sub) else float("nan")
        lines.append(
            f"| {labels[k]} | {vals[MIX_RATE_COLS[0]]:+.4f} | "
            f"{vals[MIX_RATE_COLS[1]]:+.4f} | {vals[MIX_RATE_COLS[2]]:+.4f} |"
        )
    lines += [
        "",
        "High correlation is diagnostic only; it is not an automatic reject.",
        "",
        "## 12. Core error slices",
        "",
        "### season",
        "",
        slice_table("season"),
        "",
        "### prev1_<=0.35",
        "",
        slice_table("prev1"),
        "",
        "### asof_pitcher_n",
        "",
        slice_table("asof_pitcher_n"),
        "",
        "## 13. Leakage and reproducibility",
        "",
        f"- EXP-004 / FE-001A / FE-002A fingerprints unchanged: **{fp_ok}**",
        f"- Trackman temporal safety assertions passed: **{leakage_ok}**",
        "- test.csv loaded only for schema checks; no test-row aggregates",
        "- no current-pitch Trackman join; TM columns used: season, pitch_type_group, rel_speed",
        "- seed 42; same CatBoost params and season weights as EXP-004",
        "- Public LB: **not submitted**",
        "",
        "## 14. What this experiment tested / did not test",
        "",
        "Tested: one global historical pitch-group rel_speed profile × as-of mix feature.",
        "",
        "Not tested: spin, movement, extension, release location, clustering, archetypes, "
        "hand mapping, player-level TM, current-pitch physics, tuning, multi-seed, ensemble.",
        "",
        "A negative result does **not** mean Trackman is useless. A positive result does "
        "**not** authorize adding a TM bundle.",
        "",
        "## 15. Verdict",
        "",
        f"**{verdict}**",
        "",
        verdict_reason,
        "",
        "Single seed 42. Not a Public LB decision.",
        "",
        "## 16. Next experiment",
        "",
        "Stop. Do not auto-run expected spin/movement/extension, TM bundle, clustering, "
        "hand mapping, tuning, multi-seed, ensemble, or Public LB.",
        "",
        f"- summary: `outputs/predictions/{EXPERIMENT_ID}/summary.json`",
        f"- OOF: `outputs/predictions/{EXPERIMENT_ID}/oof_holdout_predictions.csv`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    baseline_cfg = load_config("configs/baseline.yaml")
    tm_cfg = load_config("configs/tm002a.yaml")
    assert_configs_match(baseline_cfg, tm_cfg)

    fp_before = [file_fingerprint(p) for p in PROTECTED]
    exp004_summary = ROOT / "outputs/predictions/EXP-004/summary.json"
    baseline_folds = load_baseline_folds(exp004_summary)
    exp004_used = json.loads(exp004_summary.read_text(encoding="utf-8"))["features_used"]

    dummy_cols = list(
        dict.fromkeys(list(exp004_used) + list(baseline_cfg["features"]["exclude_cols"]) + ["control_success"])
    )
    used_b, _, _ = build_feature_lists(
        dummy_cols,
        target_col="control_success",
        exclude_cols=list(baseline_cfg["features"]["exclude_cols"]),
        categorical_cols=list(baseline_cfg["features"]["categorical_cols"]),
    )
    used_tm = list(used_b) + [NEW_FEATURE]
    feature_assert = {
        "n_baseline": len(used_b),
        "n_tm002a": len(used_tm),
        "baseline_minus_tm": sorted(set(used_b) - set(used_tm)),
        "tm_minus_baseline": sorted(set(used_tm) - set(used_b)),
    }
    if set(used_tm) - set(used_b) != {NEW_FEATURE} or set(used_b) - set(used_tm):
        raise AssertionError(f"feature set assert failed: {feature_assert}")
    if set(exp004_used) != set(used_b):
        raise AssertionError("Reconstructed baseline features != EXP-004 features_used")
    if len(used_b) != 44:
        raise AssertionError(f"Expected EXP-004 44 features, found {len(used_b)}")

    if not TRACKMAN_PATH.exists():
        raise FileNotFoundError(f"Trackman history not found: {TRACKMAN_PATH}")
    tm = pd.read_csv(
        TRACKMAN_PATH,
        encoding="utf-8-sig",
        usecols=["season", "pitch_type_group", "rel_speed"],
        low_memory=False,
    )
    if set(tm["pitch_type_group"].astype(str).str.lower().str.strip().unique()) - {
        "fastball",
        "breaking",
        "offspeed",
        "other",
    }:
        raise AssertionError("Unexpected pitch_type_group values in Trackman")

    profile_store: list[dict[str, Any]] = []
    diag_store: list[dict[str, Any]] = []
    corr_store: list[dict[str, Any]] = []
    result = run_tree_experiment(
        experiment_id=EXPERIMENT_ID,
        model_name="catboost",
        backend="catboost",
        config_path="configs/tm002a.yaml",
        params_key="catboost",
        fit_model=fit_catboost,
        model_filename="catboost_primary.joblib",
        sample_weight_by_season=dict(SEASON_WEIGHTS),
        extra_exclude_cols=None,
        extra_numeric_cols=[NEW_FEATURE],
        fold_extra_features=make_fold_extra_features(tm, profile_store, diag_store, corr_store),
        skip_experiment_log=True,
        log_overrides={
            "purpose": "tm002a_expected_rel_speed_vs_exp004",
            "next_action": "GPT review of TM-002A; no TM bundle; no Public LB; stop.",
        },
    )

    if set(result["features_used"]) - set(exp004_used) != {NEW_FEATURE}:
        raise AssertionError("Trained feature delta is not exactly +tm_expected_rel_speed")
    if set(exp004_used) - set(result["features_used"]):
        raise AssertionError("TM-002A dropped baseline features; aborting")
    if int(result["n_features_used"]) != 45:
        raise AssertionError(f"Expected 45 features, got {result['n_features_used']}")

    fp_after = [file_fingerprint(p) for p in PROTECTED]
    if fp_before != fp_after:
        raise AssertionError("Protected EXP-004/FE-001A/FE-002A artifacts changed during TM-002A.")

    oof_base = pd.read_csv(ROOT / "outputs/predictions/EXP-004/oof_holdout_predictions.csv")
    oof_tm = pd.read_csv(ROOT / "outputs/predictions/TM-002A/oof_holdout_predictions.csv")
    if oof_tm["prediction"].isna().any() or np.isinf(oof_tm["prediction"].to_numpy()).any():
        raise AssertionError("TM-002A predictions contain NaN/inf")
    if ((oof_tm["prediction"] < 0) | (oof_tm["prediction"] > 1)).any():
        raise AssertionError("TM-002A predictions outside [0, 1]")
    if not oof_tm["row_id"].is_unique:
        raise AssertionError("TM-002A row_id not unique")
    if set(oof_tm["row_id"]) != set(oof_base["row_id"]):
        raise AssertionError("TM-002A row_id set != EXP-004")

    leakage_ok = True
    for row in result["fold_metrics"]:
        extra = row.get("fold_extra") or {}
        valid_season = int((extra.get("valid_season") or row["valid_seasons"][0]))
        tm_seasons = extra.get("tm_seasons") or []
        if any(int(s) >= valid_season for s in tm_seasons) or 2025 in {int(s) for s in tm_seasons}:
            leakage_ok = False
            raise AssertionError(f"{row['fold_id']} Trackman leakage: seasons={tm_seasons}")

    train = pd.read_csv(
        ROOT / "data/raw/train.csv",
        encoding="utf-8-sig",
        usecols=[
            "row_id",
            "asof_pitcher_n",
            "asof_pitcher_prev1_game_success_rate",
        ],
    )
    slice_df = compare_slices(oof_base, oof_tm, train)
    out_dir = ensure_dir(ROOT / "outputs" / "predictions" / EXPERIMENT_ID)
    profile_df = pd.DataFrame(profile_store)
    diag_df = pd.DataFrame(diag_store)
    corr_df = pd.DataFrame(corr_store)
    profile_df.to_csv(out_dir / "tm_profiles.csv", index=False)
    diag_df.to_csv(out_dir / "feature_diagnostics.csv", index=False)
    corr_df.to_csv(out_dir / "mix_correlations.csv", index=False)
    slice_df.to_csv(out_dir / "slice_comparison_vs_exp004.csv", index=False)

    fmap = {r["fold_id"]: r for r in result["fold_metrics"]}
    order = ["secondary_2022", "secondary_2023", "primary_2024"]
    b_raw = np.array([float(baseline_folds[k]["raw_brier_skill_score"]) for k in order])
    t_raw = np.array([float(fmap[k]["raw_brier_skill_score"]) for k in order])
    deltas = {
        "d2022": float(t_raw[0] - b_raw[0]),
        "d2023": float(t_raw[1] - b_raw[1]),
        "d2024": float(t_raw[2] - b_raw[2]),
        "dmean": float(np.mean(t_raw) - np.mean(b_raw)),
        "dstd": float(np.std(t_raw, ddof=0) - np.std(b_raw, ddof=0)),
        "dworst": float(np.min(t_raw) - np.min(b_raw)),
    }
    missing_rate = float(diag_df.loc[diag_df["split"] == "valid", "missing_rate"].mean()) if len(diag_df) else float("nan")
    max_abs_corr = float(corr_df["pearson"].abs().max()) if len(corr_df) else float("nan")
    near_clone = bool(np.isfinite(max_abs_corr) and max_abs_corr >= 0.99)
    verdict, reason = decide_verdict(deltas, leakage_ok=leakage_ok, near_clone=near_clone)
    save_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "baseline": BASELINE_ID,
            "new_feature": NEW_FEATURE,
            "profile_definition": f"{REL_SPEED_SUMMARY}(rel_speed) by pitch_type_group; other excluded",
            "feature_assert": feature_assert,
            "deltas": deltas,
            "missing_rate_valid_mean": missing_rate,
            "max_abs_corr_with_mix": max_abs_corr,
            "near_clone": near_clone,
            "verdict": verdict,
            "verdict_reason": reason,
            "protected_fingerprints_unchanged": True,
            "leakage_ok": leakage_ok,
            "public_lb": "not_submitted",
        },
        out_dir / "comparison_vs_exp004.json",
    )
    write_report(
        baseline_folds=baseline_folds,
        tm_folds=result["fold_metrics"],
        slice_df=slice_df,
        profile_df=profile_df,
        diag_df=diag_df,
        corr_df=corr_df,
        n_base_feat=len(used_b),
        n_tm_feat=len(result["features_used"]),
        feature_assert=feature_assert,
        verdict=verdict,
        verdict_reason=reason,
        fp_ok=True,
        leakage_ok=leakage_ok,
        missing_rate=missing_rate,
        report_path=ROOT / "reports" / "tm_002a_expected_velocity.md",
    )

    gap_2023_base = float(baseline_folds["secondary_2023"]["prediction_mean"]) - float(
        baseline_folds["secondary_2023"]["valid_target_rate"]
    )
    gap_2023_tm = float(fmap["secondary_2023"]["prediction_mean"]) - float(
        fmap["secondary_2023"]["valid_target_rate"]
    )
    append_experiment_log(
        tm_cfg["output"]["experiments_path"],
        {
            "experiment_id": EXPERIMENT_ID,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "tm002a_expected_rel_speed_vs_exp004",
            "data_version": tm_cfg.get("project", {}).get("data_version", ""),
            "features": to_json_text(
                {
                    "n_before": len(used_b),
                    "n_after": len(result["features_used"]),
                    "new_feature": NEW_FEATURE,
                    "delta": feature_assert,
                }
            ),
            "model": "catboost",
            "params": to_json_text(
                {
                    **tm_cfg["models"]["params"]["catboost"],
                    "sample_weight_by_season": {
                        str(k): float(v) for k, v in sorted(SEASON_WEIGHTS.items())
                    },
                }
            ),
            "cv_strategy": to_json_text(tm_cfg["cv"]),
            "metric": "brier_skill_score",
            "metric_direction": "maximize",
            "cv_score": result["fold_raw_bss_mean"],
            "cv_std": result["fold_raw_bss_std"],
            "holdout_score": fmap["primary_2024"]["official_brier_skill_score"],
            "public_lb_score": "not_submitted",
            "cv_lb_gap": "",
            "seed": 42,
            "notes": to_json_text(
                {
                    "baseline": BASELINE_ID,
                    "hypothesis": (
                        "prior-season Trackman group mean rel_speed x as-of pitch mix "
                        "adds orthogonal physical scale vs EXP-004 mix rates"
                    ),
                    "new_feature": NEW_FEATURE,
                    "trackman_profile_definition": f"{REL_SPEED_SUMMARY}(rel_speed) by fastball/breaking/offspeed; other excluded",
                    "trackman_cutoff_by_fold": {
                        k: (fmap[k].get("fold_extra") or {}).get("tm_seasons") for k in order
                    },
                    "feature_count_before": len(used_b),
                    "feature_count_after": len(result["features_used"]),
                    "validation": "expanding 2019-21→2022, 2019-22→2023, 2019-23→2024",
                    "season_weights": {str(k): float(v) for k, v in sorted(SEASON_WEIGHTS.items())},
                    "fold_scores": {
                        k: float(fmap[k]["raw_brier_skill_score"]) for k in order
                    },
                    "cv_mean": result["fold_raw_bss_mean"],
                    "cv_std": result["fold_raw_bss_std"],
                    "worst_fold": float(np.min(t_raw)),
                    "delta_vs_exp004": deltas,
                    "missing_rate": missing_rate,
                    "gap_2023_exp004": gap_2023_base,
                    "gap_2023_tm002a": gap_2023_tm,
                    "leakage_check": "tm_season_lt_valid_no_id_join_no_current_pitch_no_test_agg",
                    "public_lb": "not_submitted",
                    "verdict": verdict,
                    "verdict_reason": reason,
                }
            ),
            "next_action": "GPT review of TM-002A; do not auto-run other Trackman features; no Public LB.",
            "status": "success",
            "error": "",
        },
    )
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "verdict": verdict,
                **deltas,
                "n_features": result["n_features_used"],
                "missing_rate": missing_rate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
