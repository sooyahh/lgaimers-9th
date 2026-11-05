"""FE-002: replace prev1 values with k=500 row-local shrinkage vs EXP-004.

Does not overwrite EXP-004 / FE-001A / FE-002A / TM-002A / TM-003A artifacts.
No Public LB submit. Not FE-002A (does not drop prev1). Not FE-001 (no count_code).

The modeling column name stays `asof_pitcher_prev1_game_success_rate`; values are
the shrunk series, not raw prev1. Slice labels use the original raw prev1 from
train.csv.

Verdict thresholds are frozen before any FE-002 scores exist:
    ADOPT_CANDIDATE if prev1<=0.35 raw BSS improves AND 3-fold mean improves
        AND no temporal fold delta < -50.
    REJECT if prev1<=0.35 does not improve, OR mean worsens with no
        temporal-stability gain (worst-fold up or std down without collapse).
    HOLD otherwise.
2023 and 2023 x n<=500 are diagnostics only and are not inputs to decide_verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.tree_exp_common as tree_exp_common
from scripts.fe002_prev1_shrinkage import (
    ATTR_KEY,
    LONGTERM_COL,
    N_COL,
    PREV1_COL,
    SHRINKAGE_K,
    add_shrinkage_to_loaded,
)
from scripts.run_exp004_catboost_weighted import fit_catboost
from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from scripts.tree_exp_common import build_feature_lists, run_tree_experiment
from src.metrics import compute_brier_metrics
from src.utils import append_experiment_log, ensure_dir, load_config, save_json, to_json_text

EXPERIMENT_ID = "FE-002"
SMOKE_EXPERIMENT_ID = "FE-002-SMOKE"
BASELINE_ID = "EXP-004"
REPLACED_FEATURE = PREV1_COL
PREV1_TRANSFORM = "row_local_shrinkage_k500"
# Material fold-change floor, frozen a priori (FE-002A used |Δ|<50 as noise vs ~1300 BSS).
# Applied to every temporal fold; not a 2023-only gate.
COLLAPSE_BSS_DELTA = 50.0
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
    ROOT / "outputs/predictions/TM-002A/summary.json",
    ROOT / "outputs/predictions/TM-002A/oof_holdout_predictions.csv",
    ROOT / "outputs/predictions/TM-002A/fold_metrics.csv",
    ROOT / "outputs/predictions/TM-003A/summary.json",
    ROOT / "outputs/predictions/TM-003A/oof_holdout_predictions.csv",
    ROOT / "outputs/predictions/TM-003A/fold_metrics.csv",
]
TM_FEATURE_NAMES = (
    "tm_expected_rel_speed",
    "tm_expected_ivb_by_count",
    "tm_expected_ivb_global",
)
FORBIDDEN_FEATURES = ("count_code",) + TM_FEATURE_NAMES
FOLD_ORDER = ["secondary_2022", "secondary_2023", "primary_2024"]
FOLD_LABELS = {
    "secondary_2022": "2022",
    "secondary_2023": "2023",
    "primary_2024": "2024",
}

_ORIGINAL_LOAD = tree_exp_common.load_competition_data


def load_competition_data_with_prev1_shrinkage(config: dict[str, Any]) -> dict[str, Any]:
    """FE-002-only wrapper. EXP-004 keeps using the unpatched loader."""
    return add_shrinkage_to_loaded(_ORIGINAL_LOAD(config))


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


def assert_configs_match_except_project(baseline_cfg: dict[str, Any], fe_cfg: dict[str, Any]) -> None:
    for key in ("cv", "task", "runtime"):
        if baseline_cfg[key] != fe_cfg[key]:
            raise AssertionError(f"config mismatch in {key}")
    if baseline_cfg["models"]["params"]["catboost"] != fe_cfg["models"]["params"]["catboost"]:
        raise AssertionError("CatBoost params mismatch vs baseline.yaml")
    if list(baseline_cfg["features"]["exclude_cols"]) != list(fe_cfg["features"]["exclude_cols"]):
        raise AssertionError("exclude_cols mismatch; FE-002 must not drop EXP-004 features")
    if list(baseline_cfg["features"]["categorical_cols"]) != list(fe_cfg["features"]["categorical_cols"]):
        raise AssertionError("categorical_cols mismatch; FE-002 must not add count_code")
    if PREV1_COL in fe_cfg["features"]["exclude_cols"]:
        raise AssertionError("FE-002 must not exclude prev1 (that is FE-002A)")
    if "count_code" in fe_cfg["features"]["categorical_cols"]:
        raise AssertionError("FE-002 must not include count_code")


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
                f"EXP-004 {row['fold_id']} raw BSS {got} != logged {logged}. "
                "Do not run FE-002 until baseline is reproduced."
            )
    return out


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
    }


def asof_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float).fillna(0.0),
        bins=ASOF_BINS,
        labels=ASOF_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)


def prev1_bin(values: pd.Series) -> pd.Series:
    p1 = pd.to_numeric(values, errors="coerce")
    out = np.where(
        p1.isna(),
        "prev1_NA",
        np.where(p1 <= 0.35, "prev1_<=0.35", np.where(p1 >= 0.65, "prev1_>=0.65", "prev1_mid")),
    )
    return pd.Series(out, index=values.index)


def decide_verdict(
    deltas: dict[str, float],
    prev1_low_delta: float,
) -> tuple[str, str]:
    """Frozen a priori. Do not pass 2023-only slice diagnostics into this function."""
    if not np.isfinite(prev1_low_delta):
        raise AssertionError("FE-002 verdict aborted: prev1<=0.35 delta is not finite.")
    required = ("d2022", "d2023", "d2024", "dmean", "dstd", "dworst")
    missing = [k for k in required if k not in deltas]
    if missing:
        raise AssertionError(f"FE-002 verdict aborted: missing delta keys {missing}.")
    d22 = float(deltas["d2022"])
    d23 = float(deltas["d2023"])
    d24 = float(deltas["d2024"])
    dmean = float(deltas["dmean"])
    dstd = float(deltas["dstd"])
    dworst = float(deltas["dworst"])
    if not all(np.isfinite(v) for v in (d22, d23, d24, dmean, dstd, dworst)):
        raise AssertionError("FE-002 verdict aborted: non-finite fold/mean deltas.")

    slice_improved = prev1_low_delta > 0.0
    mean_improved = dmean > 0.0
    obvious_collapse = min(d22, d23, d24) < -COLLAPSE_BSS_DELTA
    temporal_stability_gain = (dworst > 0.0) or (dstd < 0.0 and not obvious_collapse)

    if slice_improved and mean_improved and not obvious_collapse:
        return (
            "ADOPT_CANDIDATE",
            "Original prev1<=0.35 raw BSS and 3-fold mean raw BSS both improved "
            f"vs EXP-004, and no temporal fold dropped by more than {COLLAPSE_BSS_DELTA:g} BSS.",
        )
    if (not slice_improved) or (dmean <= 0.0 and not temporal_stability_gain):
        return (
            "REJECT",
            "Core prev1<=0.35 slice did not improve, or 3-fold mean worsened "
            "without temporal-stability gain (worst-fold up or std down, no collapse).",
        )
    return (
        "HOLD",
        "Core prev1<=0.35 slice and overall CV / temporal folds conflict; "
        "cannot adopt or reject from the frozen rule.",
    )


def compare_slices(base_oof: pd.DataFrame, fe_oof: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    cols = ["row_id", N_COL, PREV1_COL]
    ctx = train[cols]
    base = base_oof.merge(ctx, on="row_id", how="left", validate="one_to_one")
    fe = fe_oof.merge(ctx, on="row_id", how="left", validate="one_to_one")
    if set(base["row_id"]) != set(fe["row_id"]):
        raise AssertionError("OOF row_id sets differ")
    fe = fe.set_index("row_id").loc[base["row_id"]].reset_index()
    base = base.reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    def add(family: str, level: str, bpart: pd.DataFrame, fpart: pd.DataFrame) -> None:
        rec: dict[str, Any] = {
            "slice_family": family,
            "slice_level": level,
            "n_exp004": int(len(bpart)),
            "n_fe002": int(len(fpart)),
        }
        if len(bpart) < 30 or len(fpart) < 30:
            rec["reason"] = "below_min_rows:30"
            rows.append(rec)
            return
        b = score_slice(bpart["control_success"].to_numpy(float), bpart["prediction"].to_numpy(float))
        f = score_slice(fpart["control_success"].to_numpy(float), fpart["prediction"].to_numpy(float))
        rec.update(
            {
                "exp004_raw_bss": b["raw_brier_skill_score"],
                "fe002_raw_bss": f["raw_brier_skill_score"],
                "delta_raw_bss": f["raw_brier_skill_score"] - b["raw_brier_skill_score"],
                "exp004_pred_mean": b["prediction_mean"],
                "fe002_pred_mean": f["prediction_mean"],
                "exp004_target_mean": b["target_mean"],
                "fe002_target_mean": f["target_mean"],
                "exp004_gap": b["calibration_gap"],
                "fe002_gap": f["calibration_gap"],
                "delta_gap": f["calibration_gap"] - b["calibration_gap"],
                "reason": "ok",
            }
        )
        rows.append(rec)

    for season, bpart in base.groupby("season"):
        add("season", str(int(season)), bpart, fe.loc[fe["season"] == season])

    base["prev1_bin"] = prev1_bin(base[PREV1_COL])
    fe["prev1_bin"] = prev1_bin(fe[PREV1_COL])
    for level in ["prev1_<=0.35", "prev1_mid", "prev1_>=0.65", "prev1_NA"]:
        add("prev1", level, base.loc[base["prev1_bin"] == level], fe.loc[fe["prev1_bin"] == level])
    add(
        "prev1",
        "not_prev1_<=0.35",
        base.loc[base["prev1_bin"] != "prev1_<=0.35"],
        fe.loc[fe["prev1_bin"] != "prev1_<=0.35"],
    )

    for season in [2022, 2023, 2024]:
        bmask = (base["season"] == season) & (base["prev1_bin"] == "prev1_<=0.35")
        fmask = (fe["season"] == season) & (fe["prev1_bin"] == "prev1_<=0.35")
        add("season_x_prev1_low", str(season), base.loc[bmask], fe.loc[fmask])

    base["pitcher_n_bin"] = asof_bin(base[N_COL])
    fe["pitcher_n_bin"] = asof_bin(fe[N_COL])
    for level in ASOF_LABELS:
        add(
            "asof_pitcher_n",
            str(level),
            base.loc[base["pitcher_n_bin"] == level],
            fe.loc[fe["pitcher_n_bin"] == level],
        )

    n_le = pd.to_numeric(base[N_COL], errors="coerce") <= 500
    n_le_fe = pd.to_numeric(fe[N_COL], errors="coerce") <= 500
    add("low_history", "n<=500", base.loc[n_le], fe.loc[n_le_fe])
    add("low_history", "n>500", base.loc[~n_le], fe.loc[~n_le_fe])
    for season in [2022, 2023, 2024]:
        bmask = (base["season"] == season) & n_le
        fmask = (fe["season"] == season) & n_le_fe
        add("season_x_n_le_500", str(season), base.loc[bmask], fe.loc[fmask])
    return pd.DataFrame(rows)


def write_smoke_config(base_cfg: dict[str, Any], train_path: Path, iterations: int) -> Path:
    cfg = {k: v for k, v in base_cfg.items() if k != "_config_path"}
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["train_path"] = str(train_path.relative_to(ROOT)).replace("\\", "/")
    cfg["models"] = json.loads(json.dumps(cfg["models"]))
    cfg["models"]["params"]["catboost"]["iterations"] = int(iterations)
    cfg["project"] = dict(cfg["project"])
    cfg["project"]["purpose"] = "fe002_smoke"
    out = ROOT / "data/processed/fe002_smoke/fe002_smoke.yaml"
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out


def subsample_train(train: pd.DataFrame, rows_per_season: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, part in train.groupby("season", dropna=False):
        n = min(int(len(part)), int(rows_per_season))
        parts.append(part.sample(n=n, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if "row_id" in out.columns and not out["row_id"].is_unique:
        raise AssertionError("Smoke subsample lost row_id uniqueness")
    return out


def _feature_assert_from_configs(
    baseline_cfg: dict[str, Any],
    fe_cfg: dict[str, Any],
    exp004_used: list[str],
) -> dict[str, Any]:
    dummy_cols = list(
        dict.fromkeys(list(exp004_used) + list(baseline_cfg["features"]["exclude_cols"]) + ["control_success"])
    )
    used_b, _, _ = build_feature_lists(
        dummy_cols,
        target_col="control_success",
        exclude_cols=list(baseline_cfg["features"]["exclude_cols"]),
        categorical_cols=list(baseline_cfg["features"]["categorical_cols"]),
    )
    used_f, _, _ = build_feature_lists(
        dummy_cols,
        target_col="control_success",
        exclude_cols=list(fe_cfg["features"]["exclude_cols"]),
        categorical_cols=list(fe_cfg["features"]["categorical_cols"]),
    )
    feature_assert = {
        "n_baseline": len(used_b),
        "n_fe002": len(used_f),
        "baseline_minus_fe": sorted(set(used_b) - set(used_f)),
        "fe_minus_baseline": sorted(set(used_f) - set(used_b)),
        "replaced_feature": REPLACED_FEATURE,
        "prev1_transform": PREV1_TRANSFORM,
        "shrinkage_k": SHRINKAGE_K,
        "raw_prev1_not_used_in_model": True,
    }
    if set(used_b) != set(used_f):
        raise AssertionError(f"FE-002 feature names must match EXP-004: {feature_assert}")
    if set(exp004_used) != set(used_b):
        raise AssertionError("Reconstructed baseline features != EXP-004 features_used")
    if REPLACED_FEATURE not in used_f:
        raise AssertionError("prev1 column missing from FE-002 feature names")
    leaked = [c for c in FORBIDDEN_FEATURES if c in used_f]
    if leaked:
        raise AssertionError(f"Forbidden features in FE-002: {leaked}")
    return feature_assert


def write_report(**kwargs: Any) -> None:
    baseline_folds = kwargs["baseline_folds"]
    fe_folds = kwargs["fe_folds"]
    slice_df = kwargs["slice_df"]
    n_base_feat = kwargs["n_base_feat"]
    n_fe_feat = kwargs["n_fe_feat"]
    feature_assert = kwargs["feature_assert"]
    verdict = kwargs["verdict"]
    verdict_reason = kwargs["verdict_reason"]
    fp_ok = kwargs["fp_ok"]
    report_path = kwargs["report_path"]

    fmap = {r["fold_id"]: r for r in fe_folds}
    b_raw = [float(baseline_folds[k]["raw_brier_skill_score"]) for k in FOLD_ORDER]
    f_raw = [float(fmap[k]["raw_brier_skill_score"]) for k in FOLD_ORDER]
    b_mean, f_mean = float(np.mean(b_raw)), float(np.mean(f_raw))
    b_std, f_std = float(np.std(b_raw, ddof=0)), float(np.std(f_raw, ddof=0))
    b_worst, f_worst = float(min(b_raw)), float(min(f_raw))

    def slice_rows(family: str) -> str:
        lines = [
            "| slice | n | EXP-004 raw BSS | FE-002 raw BSS | Δ BSS | EXP-004 gap | FE-002 gap |",
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
                f"{r.fe002_raw_bss:.1f} | {r.delta_raw_bss:+.1f} | {r.exp004_gap:+.4f} | {r.fe002_gap:+.4f} |"
            )
        return "\n".join(lines)

    lines = [
        "# FE-002 Prev1 Shrinkage (k=500)",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        f"- baseline: `{BASELINE_ID}` (existing artifacts reused; not overwritten)",
        f"- prev1 column: `{PREV1_COL}`",
        f"- prev1 values in the model: `{PREV1_TRANSFORM}` (not raw prev1)",
        f"- shrinkage_k: `{SHRINKAGE_K}`",
        "",
        "## 1. Hypothesis",
        "",
        "Extreme / thin `asof_pitcher_prev1_game_success_rate` overstates last-game noise. "
        "A row-local shrink toward the same row's `asof_pitcher_success_rate`, with "
        "`w = n / (n + 500)`, should improve EXP-004 raw BSS on the original "
        "`prev1 <= 0.35` slice without dropping a temporal fold.",
        "",
        "## 2. What changed",
        "",
        f"- In-place replacement of `{PREV1_COL}` with k=500 shrinkage values",
        f"- Feature **names** unchanged: {n_base_feat} → {n_fe_feat}",
        f"- name delta baseline − FE-002 = `{feature_assert['baseline_minus_fe']}`",
        f"- name delta FE-002 − baseline = `{feature_assert['fe_minus_baseline']}`",
        "- Raw prev1 is **not** a model input",
        "- Not FE-002A: the column is not dropped",
        "- Not FE-001: `count_code` is not added",
        "- No Trackman columns",
        "",
        "## 3. What stayed fixed",
        "",
        "- CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss",
        "- seed 42",
        "- season weights 2019:1.0 … 2023:2.0",
        "- expanding 2022/2023/2024 holdouts",
        "- `src.metrics` unchanged",
        "- fold-local median impute + native cats",
        "- prev3/prev5, count numerics, other asof rates untouched",
        "",
        "## 4. Frozen verdict rule",
        "",
        f"- ADOPT_CANDIDATE: original prev1<=0.35 raw BSS up, 3-fold mean raw BSS up, "
        f"no fold delta < -{COLLAPSE_BSS_DELTA:g}",
        "- REJECT: prev1<=0.35 not improved, or mean down without temporal-stability gain",
        "- HOLD: slice vs overall CV / folds conflict",
        "- 2023 and 2023 × n<=500 are diagnostics only; they do not enter `decide_verdict`",
        "",
        "## 5. Baseline verification",
        "",
        "EXP-004 was not retrained. Logged fold raw BSS match expected constants to 1e-8.",
        "",
        "| Fold | n_valid | target | pred | raw BSS | official | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in FOLD_ORDER:
        r = baseline_folds[k]
        lines.append(
            f"| {FOLD_LABELS[k]} | {int(r['n_valid']):,} | {r['valid_target_rate']:.6f} | "
            f"{r['prediction_mean']:.6f} | {r['raw_brier_skill_score']:.6f} | "
            f"{r['official_brier_skill_score']:.6f} | {r['brier_score']:.6f} |"
        )
    lines += [
        "",
        "## 6. Fold results",
        "",
        "| Fold | n_train | n_valid | target | pred | raw BSS | official | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in FOLD_ORDER:
        r = fmap[k]
        lines.append(
            f"| {FOLD_LABELS[k]} | {int(r['n_train']):,} | {int(r['n_valid']):,} | "
            f"{r['valid_target_rate']:.6f} | {r['prediction_mean']:.6f} | "
            f"{r['raw_brier_skill_score']:.6f} | {r['official_brier_skill_score']:.6f} | "
            f"{r['brier_score']:.6f} |"
        )
    lines += [
        "",
        f"- FE-002 mean (ddof=0): `{f_mean:.6f}` (EXP-004 `{b_mean:.6f}`)",
        f"- FE-002 std (ddof=0): `{f_std:.6f}` (EXP-004 `{b_std:.6f}`)",
        f"- FE-002 worst: `{f_worst:.6f}` (EXP-004 `{b_worst:.6f}`)",
        "",
        "## 7. Delta vs EXP-004",
        "",
        "| Fold | EXP-004 | FE-002 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for k, br, fr in zip(FOLD_ORDER, b_raw, f_raw):
        lines.append(f"| {FOLD_LABELS[k]} | {br:.6f} | {fr:.6f} | {fr - br:+.6f} |")
    lines += [
        f"| Mean | {b_mean:.6f} | {f_mean:.6f} | {f_mean - b_mean:+.6f} |",
        f"| Std | {b_std:.6f} | {f_std:.6f} | {f_std - b_std:+.6f} |",
        f"| Worst fold | {b_worst:.6f} | {f_worst:.6f} | {f_worst - b_worst:+.6f} |",
        "",
        "## 8. Prev1 slice results (original raw prev1 labels)",
        "",
        "Bins match ERR-AUDIT-001. Slice membership uses raw `train.csv` prev1, "
        "not the shrunk modeling values.",
        "",
        slice_rows("prev1"),
        "",
        "## 9. Season × prev1<=0.35",
        "",
        slice_rows("season_x_prev1_low"),
        "",
        "## 10. Low-history diagnostics (not used in verdict)",
        "",
        "### asof_pitcher_n bins",
        "",
        slice_rows("asof_pitcher_n"),
        "",
        "### n<=500 vs n>500",
        "",
        slice_rows("low_history"),
        "",
        "### season × n<=500 (includes required 2023 × n<=500)",
        "",
        slice_rows("season_x_n_le_500"),
        "",
        "## 11. Season folds",
        "",
        slice_rows("season"),
        "",
        "## 12. Leakage and reproducibility",
        "",
        f"- EXP-004 / FE-001A / FE-002A / TM fingerprints unchanged: **{fp_ok}**",
        "- shrinkage is a function of the current row's prev1, longterm, and n only",
        "- test.csv loaded only for schema checks; no test-row aggregates",
        "- seed 42; same CatBoost params and season weights as EXP-004",
        "- Public LB: **not submitted**",
        "",
        "## 13. Verdict",
        "",
        f"**{verdict}**",
        "",
        verdict_reason,
        "",
        "Single seed 42. Not a Public LB decision.",
        "",
        "## 14. Next experiment",
        "",
        "Stop. Do not auto-run another FE or TM experiment or Public LB.",
        "",
        f"- summary: `outputs/predictions/{EXPERIMENT_ID}/summary.json`",
        f"- OOF: `outputs/predictions/{EXPERIMENT_ID}/oof_holdout_predictions.csv`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FE-002 prev1 shrinkage vs EXP-004.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--smoke-rows-per-season", type=int, default=800)
    parser.add_argument("--smoke-iterations", type=int, default=20)
    args = parser.parse_args()
    smoke = bool(args.smoke)
    canonical = bool(args.canonical)
    if smoke and canonical:
        raise AssertionError("refusing --canonical with --smoke")
    if not smoke and not canonical:
        raise SystemExit("Specify exactly one of --smoke or --canonical.")

    baseline_cfg = load_config("configs/baseline.yaml")
    fe_cfg = load_config("configs/fe002.yaml")
    assert_configs_match_except_project(baseline_cfg, fe_cfg)

    experiment_id = SMOKE_EXPERIMENT_ID if smoke else EXPERIMENT_ID
    config_path: str | Path = "configs/fe002.yaml"
    fp_before = [file_fingerprint(p) for p in PROTECTED]
    exp004_used: list[str] = []
    baseline_folds: dict[str, dict[str, Any]] = {}
    feature_assert: dict[str, Any] = {}

    if smoke:
        raw_train_path = ROOT / str(fe_cfg["data"]["train_path"])
        raw_train = pd.read_csv(raw_train_path, encoding="utf-8-sig", low_memory=False)
        smoke_dir = ensure_dir(ROOT / "data/processed/fe002_smoke")
        smoke_train = subsample_train(raw_train, args.smoke_rows_per_season, seed=42)
        smoke_train_path = smoke_dir / "train.csv"
        smoke_train.to_csv(smoke_train_path, index=False)
        config_path = write_smoke_config(fe_cfg, smoke_train_path, args.smoke_iterations)
        smoke_cfg = load_config(config_path)
        if smoke_cfg["models"]["params"]["catboost"]["iterations"] != int(args.smoke_iterations):
            raise AssertionError("smoke iterations were not applied")
        if smoke_cfg["models"]["params"]["catboost"]["iterations"] == baseline_cfg["models"]["params"]["catboost"]["iterations"]:
            raise AssertionError("smoke must not use canonical CatBoost iterations")
    else:
        exp004_summary = ROOT / "outputs/predictions/EXP-004/summary.json"
        baseline_folds = load_baseline_folds(exp004_summary)
        exp004_used = json.loads(exp004_summary.read_text(encoding="utf-8"))["features_used"]
        feature_assert = _feature_assert_from_configs(baseline_cfg, fe_cfg, exp004_used)

    tree_exp_common.load_competition_data = load_competition_data_with_prev1_shrinkage
    try:
        result = run_tree_experiment(
            experiment_id=experiment_id,
            model_name="catboost",
            backend="catboost",
            config_path=config_path,
            params_key="catboost",
            fit_model=fit_catboost,
            model_filename="catboost_primary.joblib",
            sample_weight_by_season=dict(SEASON_WEIGHTS),
            extra_exclude_cols=None,
            skip_experiment_log=True,
            log_overrides={
                "purpose": "fe002_smoke" if smoke else "fe002_prev1_row_local_shrinkage_k500_vs_exp004",
                "next_action": (
                    "Smoke only; do not treat scores as FE-002 canonical."
                    if smoke
                    else "GPT review of FE-002 fold/prev1-slice deltas; no Public LB."
                ),
            },
        )
    finally:
        tree_exp_common.load_competition_data = _ORIGINAL_LOAD

    fp_after = [file_fingerprint(p) for p in PROTECTED]
    if fp_before != fp_after:
        raise AssertionError("Protected artifacts changed during FE-002; aborting.")

    used = list(result["features_used"])
    if PREV1_COL not in used:
        raise AssertionError("Trained FE-002 features missing prev1 column name")
    if "asof_pitcher_prev3_game_success_rate" not in used:
        raise AssertionError("Trained FE-002 dropped prev3")
    if "asof_pitcher_prev5_game_success_rate" not in used:
        raise AssertionError("Trained FE-002 dropped prev5")
    leaked = [c for c in FORBIDDEN_FEATURES if c in used]
    if leaked:
        raise AssertionError(f"Forbidden features leaked into FE-002: {leaked}")
    expected_weights = {str(k): float(v) for k, v in sorted(SEASON_WEIGHTS.items())}
    if result.get("sample_weight_by_season") != expected_weights:
        raise AssertionError(f"season weights mismatch: {result.get('sample_weight_by_season')}")

    oof_fe = pd.read_csv(ROOT / "outputs/predictions" / experiment_id / "oof_holdout_predictions.csv")
    if oof_fe["prediction"].isna().any() or np.isinf(oof_fe["prediction"].to_numpy()).any():
        raise AssertionError(f"{experiment_id} predictions contain NaN/inf")
    if ((oof_fe["prediction"] < 0) | (oof_fe["prediction"] > 1)).any():
        raise AssertionError(f"{experiment_id} predictions outside [0, 1]")

    if smoke:
        integrity = {
            "experiment_id": experiment_id,
            "smoke": True,
            "n_features_used": int(result["n_features_used"]),
            "has_prev1_column": PREV1_COL in used,
            "prev1_transform": PREV1_TRANSFORM,
            "shrinkage_k": SHRINKAGE_K,
            "raw_prev1_not_used_in_model": True,
            "keeps_prev3_prev5": True,
            "forbidden_features_absent": leaked == [],
            "prediction_in_unit_interval": True,
            "protected_artifacts_unchanged": True,
            "catboost_iterations": int(
                load_config(config_path)["models"]["params"]["catboost"]["iterations"]
            ),
            "note": "Smoke scores are not FE-002 canonical results and must not be compared to EXP-004.",
        }
        out_dir = ensure_dir(ROOT / "outputs" / "predictions" / experiment_id)
        save_json(integrity, out_dir / "smoke_summary.json")
        print(json.dumps(integrity, indent=2, ensure_ascii=False))
        return

    if set(result["features_used"]) != set(exp004_used):
        raise AssertionError("Trained FE-002 feature names != EXP-004")
    if int(result["n_features_used"]) != 44:
        raise AssertionError(f"Expected 44 features, got {result['n_features_used']}")

    oof_base = pd.read_csv(ROOT / "outputs/predictions/EXP-004/oof_holdout_predictions.csv")
    train = pd.read_csv(
        ROOT / "data/raw/train.csv",
        encoding="utf-8-sig",
        usecols=["row_id", N_COL, PREV1_COL],
    )
    slice_df = compare_slices(oof_base, oof_fe, train)
    out_dir = ensure_dir(ROOT / "outputs" / "predictions" / EXPERIMENT_ID)
    slice_df.to_csv(out_dir / "slice_comparison_vs_exp004.csv", index=False)

    fmap = {r["fold_id"]: r for r in result["fold_metrics"]}
    b_raw = np.array([float(baseline_folds[k]["raw_brier_skill_score"]) for k in FOLD_ORDER])
    f_raw = np.array([float(fmap[k]["raw_brier_skill_score"]) for k in FOLD_ORDER])
    deltas = {
        "d2022": float(f_raw[0] - b_raw[0]),
        "d2023": float(f_raw[1] - b_raw[1]),
        "d2024": float(f_raw[2] - b_raw[2]),
        "dmean": float(np.mean(f_raw) - np.mean(b_raw)),
        "dstd": float(np.std(f_raw, ddof=0) - np.std(b_raw, ddof=0)),
        "dworst": float(np.min(f_raw) - np.min(b_raw)),
    }
    prev1_row = slice_df.loc[
        (slice_df.slice_family == "prev1") & (slice_df.slice_level == "prev1_<=0.35")
    ]
    if len(prev1_row) != 1:
        raise AssertionError("FE-002 aborted: prev1<=0.35 slice row missing")
    prev1_low_delta = float(prev1_row["delta_raw_bss"].iloc[0])
    verdict, reason = decide_verdict(deltas, prev1_low_delta)
    comparison = {
        "experiment_id": EXPERIMENT_ID,
        "baseline": BASELINE_ID,
        "replaced_feature": REPLACED_FEATURE,
        "prev1_transform": PREV1_TRANSFORM,
        "shrinkage_k": SHRINKAGE_K,
        "raw_prev1_not_used_in_model": True,
        "feature_assert": feature_assert,
        "deltas": deltas,
        "prev1_low_delta": prev1_low_delta,
        "verdict": verdict,
        "verdict_reason": reason,
        "collapse_bss_delta": COLLAPSE_BSS_DELTA,
        "exp004_fingerprints_unchanged": True,
        "public_lb": "not_submitted",
    }
    save_json(comparison, out_dir / "comparison_vs_exp004.json")

    write_report(
        baseline_folds=baseline_folds,
        fe_folds=result["fold_metrics"],
        slice_df=slice_df,
        n_base_feat=len(exp004_used),
        n_fe_feat=len(result["features_used"]),
        feature_assert=feature_assert,
        verdict=verdict,
        verdict_reason=reason,
        fp_ok=True,
        report_path=ROOT / "reports" / "fe_002_prev1_shrinkage.md",
    )

    append_experiment_log(
        fe_cfg["output"]["experiments_path"],
        {
            "experiment_id": EXPERIMENT_ID,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "fe002_prev1_row_local_shrinkage_k500_vs_exp004",
            "data_version": fe_cfg.get("project", {}).get("data_version", ""),
            "features": to_json_text(
                {
                    "n": len(result["features_used"]),
                    "exclude": list(fe_cfg["features"]["exclude_cols"]),
                    "categorical": list(result["categorical_cols"]),
                    "encoding": "native_catboost",
                    "replaced_feature": REPLACED_FEATURE,
                    "prev1_transform": PREV1_TRANSFORM,
                    "shrinkage_k": SHRINKAGE_K,
                    "raw_prev1_not_used_in_model": True,
                    f"{ATTR_KEY}": SHRINKAGE_K,
                }
            ),
            "model": "catboost",
            "params": to_json_text(
                {
                    **fe_cfg["models"]["params"]["catboost"],
                    "sample_weight_by_season": {
                        str(k): float(v) for k, v in sorted(SEASON_WEIGHTS.items())
                    },
                }
            ),
            "cv_strategy": to_json_text(fe_cfg["cv"]),
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
                    "replaced_feature": REPLACED_FEATURE,
                    "prev1_transform": PREV1_TRANSFORM,
                    "shrinkage_k": SHRINKAGE_K,
                    "raw_prev1_not_used_in_model": True,
                    "verdict": verdict,
                    "feature_assert": feature_assert,
                    "delta_vs_exp004": deltas,
                    "prev1_low_delta": prev1_low_delta,
                    "leakage_check": "same_row_prev1_longterm_n_only_no_test_aggregates_no_trackman",
                    "public_lb": "not_submitted",
                }
            ),
            "next_action": "GPT review of FE-002 fold/prev1-slice deltas; no Public LB.",
            "status": "success",
            "error": "",
        },
    )
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "verdict": verdict, **deltas, "prev1_low_delta": prev1_low_delta}, indent=2))


if __name__ == "__main__":
    main()
