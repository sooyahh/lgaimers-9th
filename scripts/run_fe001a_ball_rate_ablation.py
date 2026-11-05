"""FE-001A: drop asof_pitcher_ball_rate from the EXP-004 CatBoost recipe.

Does not overwrite EXP-004 artifacts. No Public LB submit.
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
from src.utils import ensure_dir, load_config, save_json

EXPERIMENT_ID = "FE-001A"
BASELINE_ID = "EXP-004"
REMOVED_FEATURE = "asof_pitcher_ball_rate"
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


def assert_configs_match_except_exclude(baseline_cfg: dict[str, Any], fe_cfg: dict[str, Any]) -> None:
    for key in ("cv", "task", "runtime"):
        if baseline_cfg[key] != fe_cfg[key]:
            raise AssertionError(f"config mismatch in {key}")
    if baseline_cfg["models"]["params"]["catboost"] != fe_cfg["models"]["params"]["catboost"]:
        raise AssertionError("CatBoost params mismatch vs baseline.yaml")
    b_ex = set(baseline_cfg["features"]["exclude_cols"])
    f_ex = set(fe_cfg["features"]["exclude_cols"])
    if f_ex - b_ex != {REMOVED_FEATURE}:
        raise AssertionError(f"exclude delta FE-baseline = {f_ex - b_ex}")
    if b_ex - f_ex:
        raise AssertionError(f"baseline exclude not subset of FE: {b_ex - f_ex}")
    if baseline_cfg["features"]["categorical_cols"] != fe_cfg["features"]["categorical_cols"]:
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
                f"EXP-004 {row['fold_id']} raw BSS {got} != logged {logged}. "
                "Do not run FE-001A until baseline is reproduced."
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


def compare_slices(base_oof: pd.DataFrame, fe_oof: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "row_id",
        "asof_pitcher_n",
        "asof_pitcher_prev1_game_success_rate",
        "balls_before",
        "strikes_before",
    ]
    ctx = train[cols]
    base = base_oof.merge(ctx, on="row_id", how="left", validate="one_to_one")
    fe = fe_oof.merge(ctx, on="row_id", how="left", validate="one_to_one")
    if len(base) != len(fe):
        raise AssertionError(f"OOF row count mismatch: EXP-004={len(base)} FE-001A={len(fe)}")
    if not base["row_id"].equals(fe["row_id"]):
        # allow same multiset in different order
        if set(base["row_id"]) != set(fe["row_id"]):
            raise AssertionError("OOF row_id sets differ")
        fe = fe.set_index("row_id").loc[base["row_id"]].reset_index()
        base = base.reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    def add(family: str, level: str, bpart: pd.DataFrame, fpart: pd.DataFrame) -> None:
        if len(bpart) < 30 or len(fpart) < 30:
            rows.append(
                {
                    "slice_family": family,
                    "slice_level": level,
                    "n_exp004": int(len(bpart)),
                    "n_fe001a": int(len(fpart)),
                    "reason": "below_min_rows:30",
                }
            )
            return
        b = score_slice(bpart["control_success"].to_numpy(float), bpart["prediction"].to_numpy(float))
        f = score_slice(fpart["control_success"].to_numpy(float), fpart["prediction"].to_numpy(float))
        rows.append(
            {
                "slice_family": family,
                "slice_level": level,
                "n_exp004": b["n"],
                "n_fe001a": f["n"],
                "exp004_raw_bss": b["raw_brier_skill_score"],
                "fe001a_raw_bss": f["raw_brier_skill_score"],
                "delta_raw_bss": f["raw_brier_skill_score"] - b["raw_brier_skill_score"],
                "exp004_pred_mean": b["prediction_mean"],
                "fe001a_pred_mean": f["prediction_mean"],
                "exp004_target_mean": b["target_mean"],
                "fe001a_target_mean": f["target_mean"],
                "exp004_gap": b["calibration_gap"],
                "fe001a_gap": f["calibration_gap"],
                "delta_gap": f["calibration_gap"] - b["calibration_gap"],
                "reason": "ok",
            }
        )

    for season, bpart in base.groupby("season"):
        fpart = fe.loc[fe["season"] == season]
        add("season", str(int(season)), bpart, fpart)

    base["pitcher_n_bin"] = asof_bin(base["asof_pitcher_n"])
    fe["pitcher_n_bin"] = asof_bin(fe["asof_pitcher_n"])
    for level, bpart in base.groupby("pitcher_n_bin"):
        fpart = fe.loc[fe["pitcher_n_bin"] == level]
        add("asof_pitcher_n", str(level), bpart, fpart)

    b_low = base.loc[base["asof_pitcher_prev1_game_success_rate"] <= 0.35]
    f_low = fe.loc[fe["asof_pitcher_prev1_game_success_rate"] <= 0.35]
    add("prev1", "prev1_<=0.35", b_low, f_low)

    b_c = base.loc[(base["balls_before"] == 0) & (base["strikes_before"] == 2)]
    f_c = fe.loc[(fe["balls_before"] == 0) & (fe["strikes_before"] == 2)]
    add("count", "0-2", b_c, f_c)
    return pd.DataFrame(rows)


def fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "missing"
    return f"{float(x):.{nd}f}"


def decide_verdict(deltas: dict[str, float]) -> tuple[str, str]:
    d22, d23, d24 = deltas["d2022"], deltas["d2023"], deltas["d2024"]
    dmean, dstd, dworst = deltas["dmean"], deltas["dstd"], deltas["dworst"]
    # Conservative, not a significance test.
    if d23 > 50 and d22 > -30 and d24 > -30 and (dmean > 0 or dworst > 20):
        return (
            "ADOPT",
            "2023/worst fold improved with limited 2022/2024 loss; treat as adopt *candidate* only (single seed).",
        )
    if abs(d23) < 15 and d22 < 0 and d24 < 0 and dmean < 0:
        return "REJECT", "No meaningful 2023 gain and other folds / mean worsened."
    if d23 < 5 and dmean <= 0 and dworst <= 0:
        return "REJECT", "2023 did not improve in a material way; mean and worst fold did not improve."
    return (
        "HOLD",
        "Fold deltas are mixed or modest; do not adopt from a single-seed ablation.",
    )


def write_report(
    *,
    baseline_folds: dict[str, dict[str, Any]],
    fe_folds: list[dict[str, Any]],
    slice_df: pd.DataFrame,
    n_base_feat: int,
    n_fe_feat: int,
    fingerprints_before: list[dict[str, Any]],
    fingerprints_after: list[dict[str, Any]],
    feature_assert: dict[str, Any],
    verdict: str,
    verdict_reason: str,
    report_path: Path,
) -> None:
    fmap = {r["fold_id"]: r for r in fe_folds}
    order = ["secondary_2022", "secondary_2023", "primary_2024"]
    labels = {"secondary_2022": "2022", "secondary_2023": "2023", "primary_2024": "2024"}
    b_raw = [float(baseline_folds[k]["raw_brier_skill_score"]) for k in order]
    f_raw = [float(fmap[k]["raw_brier_skill_score"]) for k in order]
    b_mean, f_mean = float(np.mean(b_raw)), float(np.mean(f_raw))
    b_std, f_std = float(np.std(b_raw, ddof=0)), float(np.std(f_raw, ddof=0))
    b_worst, f_worst = float(min(b_raw)), float(min(f_raw))

    lines = [
        "# FE-001A Ball Rate Ablation",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        f"- baseline: `{BASELINE_ID}` (existing artifacts reused; not retrained, not overwritten)",
        "",
        "## 1. Hypothesis",
        "",
        "FEATURE-STABILITY-AUDIT labeled `asof_pitcher_ball_rate` as **SIGN_FLIP**. "
        "If EXP-004’s 2023 collapse is partly caused by trusting that flipped association, "
        "removing only this column should improve 2023 (and/or worst-fold) raw BSS "
        "without large 2022/2024 damage.",
        "",
        "## 2. What changed",
        "",
        f"- Removed exactly one feature: `{REMOVED_FEATURE}`",
        f"- Feature count: {n_base_feat} → {n_fe_feat} (delta {n_fe_feat - n_base_feat})",
        f"- Assert baseline − FE-001A = `{feature_assert['baseline_minus_fe']}`",
        f"- Assert FE-001A − baseline = `{feature_assert['fe_minus_baseline']}`",
        "",
        "## 3. What stayed fixed",
        "",
        "- Model: CatBoost native cats, `iterations=300`, `learning_rate=0.05`, `depth=6`, `Logloss`",
        "- Seed: 42",
        "- Season weights: `{2019:1.0, 2020:1.2, 2021:1.4, 2022:1.6, 2023:2.0}`",
        "- Validation: expanding 2019–21→2022, 2019–22→2023, 2019–23→2024",
        "- Metric: `src.metrics.compute_brier_metrics` / official BSS clip-at-zero",
        "- Preprocessing: fold-local numeric median impute + native cat strings (`tree_exp_common`)",
        "- No tuning, no ensemble, no Trackman, no prev* edits, no test.csv aggregates",
        "",
        "## 4. Baseline reproduction",
        "",
        "EXP-004 was **not re-run** (would overwrite protected artifacts). "
        "Logged fold raw BSS were checked against `outputs/predictions/EXP-004/summary.json` "
        "and match the expected constants to 1e-8.",
        "",
        "| Fold | n_valid | target mean | pred mean | raw BSS | official BSS | Brier |",
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
        f"- EXP-004 fold raw BSS mean (ddof=0): `{b_mean:.6f}`",
        f"- EXP-004 fold raw BSS std (ddof=0): `{b_std:.6f}`",
        f"- EXP-004 worst fold: `{b_worst:.6f}` (2023)",
        "",
        "## 5. Fold results (FE-001A)",
        "",
        "| Fold | n_train | n_valid | target mean | pred mean | raw BSS | official BSS | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        r = fmap[k]
        lines.append(
            f"| {labels[k]} | {int(r['n_train']):,} | {int(r['n_valid']):,} | "
            f"{r['valid_target_rate']:.6f} | {r['prediction_mean']:.6f} | "
            f"{r['raw_brier_skill_score']:.6f} | {r['official_brier_skill_score']:.6f} | "
            f"{r['brier_score']:.6f} |"
        )
    lines += [
        "",
        f"- FE-001A fold raw BSS mean (ddof=0): `{f_mean:.6f}`",
        f"- FE-001A fold raw BSS std (ddof=0): `{f_std:.6f}`",
        f"- FE-001A worst fold: `{f_worst:.6f}`",
        "",
        "## 6. Delta vs EXP-004",
        "",
        "| Fold | EXP-004 raw BSS | FE-001A raw BSS | Delta |",
        "|---|---:|---:|---:|",
    ]
    for k, br, fr in zip(order, b_raw, f_raw):
        lines.append(f"| {labels[k]} | {br:.6f} | {fr:.6f} | {fr - br:+.6f} |")
    lines += [
        f"| Mean | {b_mean:.6f} | {f_mean:.6f} | {f_mean - b_mean:+.6f} |",
        f"| Std | {b_std:.6f} | {f_std:.6f} | {f_std - b_std:+.6f} |",
        f"| Worst fold | {b_worst:.6f} | {f_worst:.6f} | {f_worst - b_worst:+.6f} |",
        "",
        "Prediction mean vs target mean:",
        "",
        "| Fold | target | EXP-004 pred | EXP-004 gap | FE-001A pred | FE-001A gap | Δ gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        b, f = baseline_folds[k], fmap[k]
        bg = float(b["prediction_mean"]) - float(b["valid_target_rate"])
        fg = float(f["prediction_mean"]) - float(f["valid_target_rate"])
        lines.append(
            f"| {labels[k]} | {b['valid_target_rate']:.6f} | {b['prediction_mean']:.6f} | "
            f"{bg:+.6f} | {f['prediction_mean']:.6f} | {fg:+.6f} | {fg - bg:+.6f} |"
        )
    lines += [
        "",
        "## 7. Error slice comparison",
        "",
        "Minimal slices from ERR-AUDIT-001. Concatenated OOF (2022–2024 valid rows).",
        "",
        "| slice | level | n | EXP-004 raw BSS | FE-001A raw BSS | Δ BSS | EXP-004 gap | FE-001A gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in slice_df.itertuples():
        if getattr(r, "reason", "ok") != "ok":
            lines.append(
                f"| {r.slice_family} | {r.slice_level} | {int(r.n_exp004):,} | "
                f"missing | missing | missing | missing | missing |"
            )
            continue
        lines.append(
            f"| {r.slice_family} | {r.slice_level} | {int(r.n_exp004):,} | "
            f"{r.exp004_raw_bss:.1f} | {r.fe001a_raw_bss:.1f} | {r.delta_raw_bss:+.1f} | "
            f"{r.exp004_gap:+.4f} | {r.fe001a_gap:+.4f} |"
        )
    fp_ok = fingerprints_before == fingerprints_after
    lines += [
        "",
        "## 8. Leakage/reproducibility check",
        "",
        f"- EXP-004 artifact fingerprints unchanged: **{fp_ok}**",
        "- test.csv was loaded by `load_competition_data` for schema checks only; not used in features/aggregates",
        "- Fold train seasons are strictly earlier than valid seasons (`build_season_holdout`)",
        "- Seed 42; CatBoost `random_seed` from `build_model`",
        "- Public LB: **not submitted / pending**",
        "",
        "## 9. Interpretation",
        "",
        "See fold and slice tables. Removing a SIGN_FLIP asof rate is a negative-control on whether "
        "that column was a main driver of 2023 failure. Residual 2023 error after ablation would "
        "mean the collapse is mostly elsewhere (target shift, other asof rates, `season`, etc.).",
        "",
        "## 10. ADOPT / HOLD / REJECT",
        "",
        f"**{verdict}**",
        "",
        verdict_reason,
        "",
        "Single seed (42). Not a Public LB decision.",
        "",
        "## 11. Next experiment recommendation",
        "",
        "Do **not** auto-run `asof_pitcher_strike_rate` ablation. Stop for GPT review. "
        "If FE-001A is REJECT/HOLD with little 2023 movement, prefer the previously planned "
        "row-local prev1 shrinkage (FE-002 in `reports/next_experiment_decision.md`) over another rate drop.",
        "",
        f"- FE-001A summary: `outputs/predictions/{EXPERIMENT_ID}/summary.json`",
        f"- FE-001A OOF: `outputs/predictions/{EXPERIMENT_ID}/oof_holdout_predictions.csv`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    baseline_cfg = load_config("configs/baseline.yaml")
    fe_cfg = load_config("configs/fe001a.yaml")
    assert_configs_match_except_exclude(baseline_cfg, fe_cfg)

    fp_before = [file_fingerprint(p) for p in PROTECTED]
    exp004_summary = ROOT / "outputs/predictions/EXP-004/summary.json"
    baseline_folds = load_baseline_folds(exp004_summary)
    exp004_used = json.loads(exp004_summary.read_text(encoding="utf-8"))["features_used"]

    dummy_cols = list(exp004_used) + list(baseline_cfg["features"]["exclude_cols"]) + ["control_success"]
    dummy_cols = list(dict.fromkeys(dummy_cols))
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
        "n_fe001a": len(used_f),
        "baseline_minus_fe": sorted(set(used_b) - set(used_f)),
        "fe_minus_baseline": sorted(set(used_f) - set(used_b)),
    }
    if set(used_b) - set(used_f) != {REMOVED_FEATURE} or set(used_f) - set(used_b):
        raise AssertionError(f"feature set assert failed: {feature_assert}")
    if set(exp004_used) != set(used_b):
        raise AssertionError("Reconstructed baseline features != EXP-004 summary features_used")

    result = run_tree_experiment(
        experiment_id=EXPERIMENT_ID,
        model_name="catboost",
        backend="catboost",
        config_path="configs/fe001a.yaml",
        params_key="catboost",
        fit_model=fit_catboost,
        model_filename="catboost_primary.joblib",
        sample_weight_by_season=dict(SEASON_WEIGHTS),
        extra_exclude_cols=None,
        log_overrides={
            "purpose": "fe001a_drop_asof_pitcher_ball_rate_vs_exp004",
            "next_action": "GPT review of FE-001A; do not submit Public LB; do not auto-run strike_rate ablation.",
            "notes": {
                "baseline": BASELINE_ID,
                "removed_feature": REMOVED_FEATURE,
                "public_lb": "not_submitted_pending",
                "leakage_check": "same_row_features_only_no_test_aggregates",
                "n_features_baseline": len(used_b),
                "n_features_fe001a": len(used_f),
            },
        },
    )

    fp_after = [file_fingerprint(p) for p in PROTECTED]
    if fp_before != fp_after:
        raise AssertionError("EXP-004 artifacts changed during FE-001A; aborting comparison.")

    oof_base = pd.read_csv(ROOT / "outputs/predictions/EXP-004/oof_holdout_predictions.csv")
    oof_fe = pd.read_csv(ROOT / "outputs/predictions/FE-001A/oof_holdout_predictions.csv")
    if oof_fe["prediction"].isna().any() or np.isinf(oof_fe["prediction"].to_numpy()).any():
        raise AssertionError("FE-001A predictions contain NaN/inf")
    if ((oof_fe["prediction"] < 0) | (oof_fe["prediction"] > 1)).any():
        raise AssertionError("FE-001A predictions outside [0, 1]")

    train = pd.read_csv(
        ROOT / "data/raw/train.csv",
        encoding="utf-8-sig",
        usecols=[
            "row_id",
            "asof_pitcher_n",
            "asof_pitcher_prev1_game_success_rate",
            "balls_before",
            "strikes_before",
        ],
    )
    slice_df = compare_slices(oof_base, oof_fe, train)
    out_dir = ensure_dir(ROOT / "outputs" / "predictions" / EXPERIMENT_ID)
    slice_path = out_dir / "slice_comparison_vs_exp004.csv"
    slice_df.to_csv(slice_path, index=False)

    fmap = {r["fold_id"]: r for r in result["fold_metrics"]}
    order = ["secondary_2022", "secondary_2023", "primary_2024"]
    b_raw = np.array([float(baseline_folds[k]["raw_brier_skill_score"]) for k in order])
    f_raw = np.array([float(fmap[k]["raw_brier_skill_score"]) for k in order])
    deltas = {
        "d2022": float(f_raw[0] - b_raw[0]),
        "d2023": float(f_raw[1] - b_raw[1]),
        "d2024": float(f_raw[2] - b_raw[2]),
        "dmean": float(np.mean(f_raw) - np.mean(b_raw)),
        "dstd": float(np.std(f_raw, ddof=0) - np.std(b_raw, ddof=0)),
        "dworst": float(np.min(f_raw) - np.min(b_raw)),
    }
    verdict, reason = decide_verdict(deltas)
    save_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "baseline": BASELINE_ID,
            "removed_feature": REMOVED_FEATURE,
            "feature_assert": feature_assert,
            "deltas": deltas,
            "verdict": verdict,
            "verdict_reason": reason,
            "exp004_fingerprints_unchanged": True,
        },
        out_dir / "comparison_vs_exp004.json",
    )

    write_report(
        baseline_folds=baseline_folds,
        fe_folds=result["fold_metrics"],
        slice_df=slice_df,
        n_base_feat=len(used_b),
        n_fe_feat=len(used_f),
        fingerprints_before=fp_before,
        fingerprints_after=fp_after,
        feature_assert=feature_assert,
        verdict=verdict,
        verdict_reason=reason,
        report_path=ROOT / "reports" / "fe_001a_ball_rate_ablation.md",
    )
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "verdict": verdict, **deltas}, indent=2))


if __name__ == "__main__":
    main()
