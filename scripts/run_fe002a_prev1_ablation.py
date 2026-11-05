"""FE-002A: drop asof_pitcher_prev1_game_success_rate from EXP-004. No shrinkage."""

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

EXPERIMENT_ID = "FE-002A"
BASELINE_ID = "EXP-004"
REMOVED_FEATURE = "asof_pitcher_prev1_game_success_rate"
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
        raise AssertionError("CatBoost params mismatch")
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
                f"EXP-004 {row['fold_id']} raw BSS {got} != logged {logged}."
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
            "n_fe002a": int(len(fpart)),
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
                "fe002a_raw_bss": f["raw_brier_skill_score"],
                "delta_raw_bss": f["raw_brier_skill_score"] - b["raw_brier_skill_score"],
                "exp004_pred_mean": b["prediction_mean"],
                "fe002a_pred_mean": f["prediction_mean"],
                "exp004_target_mean": b["target_mean"],
                "fe002a_target_mean": f["target_mean"],
                "exp004_gap": b["calibration_gap"],
                "fe002a_gap": f["calibration_gap"],
                "delta_gap": f["calibration_gap"] - b["calibration_gap"],
                "reason": "ok",
            }
        )
        rows.append(rec)

    for season, bpart in base.groupby("season"):
        add("season", str(int(season)), bpart, fe.loc[fe["season"] == season])

    base["prev1_bin"] = prev1_bin(base["asof_pitcher_prev1_game_success_rate"])
    fe["prev1_bin"] = prev1_bin(fe["asof_pitcher_prev1_game_success_rate"])
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

    base["pitcher_n_bin"] = asof_bin(base["asof_pitcher_n"])
    fe["pitcher_n_bin"] = asof_bin(fe["asof_pitcher_n"])
    for level in ASOF_LABELS:
        add(
            "asof_pitcher_n",
            str(level),
            base.loc[base["pitcher_n_bin"] == level],
            fe.loc[fe["pitcher_n_bin"] == level],
        )
        bmask = (base["pitcher_n_bin"] == level) & (base["prev1_bin"] == "prev1_<=0.35")
        fmask = (fe["pitcher_n_bin"] == level) & (fe["prev1_bin"] == "prev1_<=0.35")
        add("asof_n_x_prev1_low", str(level), base.loc[bmask], fe.loc[fmask])

    add(
        "count",
        "0-2",
        base.loc[(base["balls_before"] == 0) & (base["strikes_before"] == 2)],
        fe.loc[(fe["balls_before"] == 0) & (fe["strikes_before"] == 2)],
    )
    return pd.DataFrame(rows)


def decide_verdict(
    deltas: dict[str, float],
    prev1_low_delta: float | None,
) -> tuple[str, str, str]:
    d22, d23, d24 = deltas["d2022"], deltas["d2023"], deltas["d2024"]
    dmean, dworst = deltas["dmean"], deltas["dworst"]
    pdelta = prev1_low_delta if prev1_low_delta is not None else float("nan")

    improved_2023 = d23 > 50
    small_side_loss = d22 > -40 and d24 > -40
    prev1_helped = np.isfinite(pdelta) and pdelta > 20
    mean_or_worst = dmean > 0 or dworst > 20
    # 2023 |Δ| < 50 is treated as no material fold change (noise vs ~1300 BSS scale).
    fold_2023_flat = abs(d23) < 50
    prev1_hurt_or_flat = (not np.isfinite(pdelta)) or pdelta <= 10

    if improved_2023 and small_side_loss and prev1_helped and mean_or_worst:
        return (
            "ADOPT_CANDIDATE",
            "2023/worst-fold and prev1<=0.35 improved with limited 2022/2024 loss.",
            "A",
        )
    if dmean <= 0 and prev1_hurt_or_flat and (d23 < 0 or fold_2023_flat):
        return (
            "REJECT",
            "Removing prev1 did not improve CV mean or the prev1<=0.35 slice; "
            "2023 fold change is not a material recovery.",
            "B",
        )
    return (
        "HOLD",
        "Fold or slice deltas are mixed; cannot separate noisy-feature vs hard-cohort yet.",
        "C",
    )


def write_report(**kwargs: Any) -> None:
    baseline_folds = kwargs["baseline_folds"]
    fe_folds = kwargs["fe_folds"]
    slice_df = kwargs["slice_df"]
    n_base_feat = kwargs["n_base_feat"]
    n_fe_feat = kwargs["n_fe_feat"]
    feature_assert = kwargs["feature_assert"]
    verdict = kwargs["verdict"]
    verdict_reason = kwargs["verdict_reason"]
    conclusion = kwargs["conclusion"]
    fp_ok = kwargs["fp_ok"]
    report_path = kwargs["report_path"]

    fmap = {r["fold_id"]: r for r in fe_folds}
    order = ["secondary_2022", "secondary_2023", "primary_2024"]
    labels = {"secondary_2022": "2022", "secondary_2023": "2023", "primary_2024": "2024"}
    b_raw = [float(baseline_folds[k]["raw_brier_skill_score"]) for k in order]
    f_raw = [float(fmap[k]["raw_brier_skill_score"]) for k in order]
    b_mean, f_mean = float(np.mean(b_raw)), float(np.mean(f_raw))
    b_std, f_std = float(np.std(b_raw, ddof=0)), float(np.std(f_raw, ddof=0))
    b_worst, f_worst = float(min(b_raw)), float(min(f_raw))

    def slice_rows(family: str) -> str:
        lines = [
            "| slice | n | EXP-004 raw BSS | FE-002A raw BSS | Δ BSS | EXP-004 gap | FE-002A gap |",
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
                f"{r.fe002a_raw_bss:.1f} | {r.delta_raw_bss:+.1f} | {r.exp004_gap:+.4f} | {r.fe002a_gap:+.4f} |"
            )
        return "\n".join(lines)

    lines = [
        "# FE-002A Prev1 Ablation",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        f"- baseline: `{BASELINE_ID}` (existing artifacts reused; not overwritten)",
        "",
        "## 1. Hypothesis",
        "",
        "If `prev1 <= 0.35` fails because `asof_pitcher_prev1_game_success_rate` is a noisy/unstable "
        "signal, dropping **only** that column should improve EXP-004 temporal CV, especially 2023 "
        "and the prev1-low slice.",
        "",
        "If dropping it **hurts**, prev1 itself is useful and the low-prev1 cohort is simply hard.",
        "",
        "## 2. Why this experiment",
        "",
        "ERR-AUDIT-001: `prev1_<=0.35` was the largest non-season failure (n=78,607, raw BSS −184). "
        "A failure **slice** is not the same as a bad **feature**. FE-002A is a single ablation to "
        "separate those. No shrinkage/smoothing is added.",
        "",
        "## 3. What changed",
        "",
        f"- Removed exactly `{REMOVED_FEATURE}`",
        f"- Feature count: {n_base_feat} → {n_fe_feat}",
        f"- baseline − FE-002A = `{feature_assert['baseline_minus_fe']}`",
        f"- FE-002A − baseline = `{feature_assert['fe_minus_baseline']}`",
        "",
        "## 4. What stayed fixed",
        "",
        "- CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss",
        "- seed 42",
        "- season weights 2019:1.0 … 2023:2.0",
        "- expanding 2022/2023/2024 holdouts",
        "- `src.metrics` unchanged",
        "- fold-local median impute + native cats",
        "- prev3/prev5, Trackman, other asof rates untouched",
        "- no shrinkage feature",
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
        "## 6. Fold results",
        "",
        "| Fold | n_train | n_valid | target | pred | raw BSS | official | Brier |",
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
        f"- FE-002A mean (ddof=0): `{f_mean:.6f}` (EXP-004 `{b_mean:.6f}`)",
        f"- FE-002A std (ddof=0): `{f_std:.6f}` (EXP-004 `{b_std:.6f}`)",
        f"- FE-002A worst: `{f_worst:.6f}` (EXP-004 `{b_worst:.6f}`)",
        "",
        "## 7. Delta vs EXP-004",
        "",
        "| Fold | EXP-004 | FE-002A | Delta |",
        "|---|---:|---:|---:|",
    ]
    for k, br, fr in zip(order, b_raw, f_raw):
        lines.append(f"| {labels[k]} | {br:.6f} | {fr:.6f} | {fr - br:+.6f} |")
    lines += [
        f"| Mean | {b_mean:.6f} | {f_mean:.6f} | {f_mean - b_mean:+.6f} |",
        f"| Std | {b_std:.6f} | {f_std:.6f} | {f_std - b_std:+.6f} |",
        f"| Worst fold | {b_worst:.6f} | {f_worst:.6f} | {f_worst - b_worst:+.6f} |",
        "",
        "## 8. Prev1 slice results",
        "",
        "Bins match ERR-AUDIT-001 (`prev1_<=0.35` / mid / `>=0.65` / NA).",
        "",
        slice_rows("prev1"),
        "",
        "## 9. Season × prev1 analysis",
        "",
        "`prev1_<=0.35` inside each OOF season:",
        "",
        slice_rows("season_x_prev1_low"),
        "",
        "## 10. Low-history interaction",
        "",
        "Canonical `asof_pitcher_n` bins, then the same bins restricted to `prev1_<=0.35`.",
        "",
        "### asof_pitcher_n",
        "",
        slice_rows("asof_pitcher_n"),
        "",
        "### asof_pitcher_n × prev1_<=0.35",
        "",
        slice_rows("asof_n_x_prev1_low"),
        "",
        "### count 0-2 (auxiliary, from FE-001A)",
        "",
        slice_rows("count"),
        "",
        "## 11. Calibration / prediction mean",
        "",
        "| Fold | target | EXP-004 pred | EXP-004 gap | FE-002A pred | FE-002A gap | Δ gap |",
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
        "## 12. Leakage and reproducibility",
        "",
        f"- EXP-004 and FE-001A fingerprints unchanged: **{fp_ok}**",
        "- test.csv loaded only for schema checks; no test-row aggregates",
        "- seed 42; same CatBoost params and season weights as EXP-004",
        "- Public LB: **not submitted**",
        "",
        "## 13. Interpretation",
        "",
        "A low-prev1 **slice** can fail even when the prev1 **feature** is useful "
        "(the cohort is hard). Conversely, if dropping prev1 improves that slice and 2023, "
        "the column itself is a noisy contributor. This run is the test of that distinction. "
        "No shrinkage was introduced, so the result is not mixed with a second hypothesis.",
        "",
        "## 14. Verdict",
        "",
        f"**{verdict}**",
        "",
        verdict_reason,
        "",
        f"Conclusion label: **{conclusion}** "
        "(A=prev1 feature noisy; B=cohort hard / feature useful; C=cannot separate).",
        "",
        "Single seed 42. Not a Public LB decision.",
        "",
        "## 15. Next experiment",
        "",
        "Stop. Do not auto-run prev1 shrinkage, prev3/prev5 ablation, strike_rate ablation, "
        "TM-002, or Public LB. Recommend at most one follow-up after GPT review.",
        "",
        f"- summary: `outputs/predictions/{EXPERIMENT_ID}/summary.json`",
        f"- OOF: `outputs/predictions/{EXPERIMENT_ID}/oof_holdout_predictions.csv`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    baseline_cfg = load_config("configs/baseline.yaml")
    fe_cfg = load_config("configs/fe002a.yaml")
    assert_configs_match_except_exclude(baseline_cfg, fe_cfg)

    fp_before = [file_fingerprint(p) for p in PROTECTED]
    exp004_summary = ROOT / "outputs/predictions/EXP-004/summary.json"
    baseline_folds = load_baseline_folds(exp004_summary)
    exp004_used = json.loads(exp004_summary.read_text(encoding="utf-8"))["features_used"]

    dummy_cols = list(dict.fromkeys(list(exp004_used) + list(baseline_cfg["features"]["exclude_cols"]) + ["control_success"]))
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
        "n_fe002a": len(used_f),
        "baseline_minus_fe": sorted(set(used_b) - set(used_f)),
        "fe_minus_baseline": sorted(set(used_f) - set(used_b)),
    }
    if set(used_b) - set(used_f) != {REMOVED_FEATURE} or set(used_f) - set(used_b):
        raise AssertionError(f"feature set assert failed: {feature_assert}")
    if set(exp004_used) != set(used_b):
        raise AssertionError("Reconstructed baseline features != EXP-004 features_used")
    if REMOVED_FEATURE not in exp004_used:
        raise AssertionError("prev1 success rate missing from EXP-004 feature set")

    result = run_tree_experiment(
        experiment_id=EXPERIMENT_ID,
        model_name="catboost",
        backend="catboost",
        config_path="configs/fe002a.yaml",
        params_key="catboost",
        fit_model=fit_catboost,
        model_filename="catboost_primary.joblib",
        sample_weight_by_season=dict(SEASON_WEIGHTS),
        extra_exclude_cols=None,
        log_overrides={
            "purpose": "fe002a_drop_asof_pitcher_prev1_game_success_rate_vs_exp004",
            "next_action": "GPT review of FE-002A; no shrinkage; no Public LB; stop.",
            "notes": {
                "baseline": BASELINE_ID,
                "removed_feature": REMOVED_FEATURE,
                "public_lb": "not_submitted",
                "leakage_check": "same_row_features_only_no_test_aggregates",
                "n_features_baseline": len(used_b),
                "n_features_fe002a": len(used_f),
                "shrinkage": False,
            },
        },
    )

    fp_after = [file_fingerprint(p) for p in PROTECTED]
    if fp_before != fp_after:
        raise AssertionError("Protected EXP-004/FE-001A artifacts changed during FE-002A.")

    oof_base = pd.read_csv(ROOT / "outputs/predictions/EXP-004/oof_holdout_predictions.csv")
    oof_fe = pd.read_csv(ROOT / "outputs/predictions/FE-002A/oof_holdout_predictions.csv")
    if oof_fe["prediction"].isna().any() or np.isinf(oof_fe["prediction"].to_numpy()).any():
        raise AssertionError("FE-002A predictions contain NaN/inf")
    if ((oof_fe["prediction"] < 0) | (oof_fe["prediction"] > 1)).any():
        raise AssertionError("FE-002A predictions outside [0, 1]")
    if not oof_fe["row_id"].is_unique:
        raise AssertionError("FE-002A row_id not unique")
    if set(oof_fe["row_id"]) != set(oof_base["row_id"]):
        raise AssertionError("FE-002A row_id set != EXP-004")

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
    slice_df.to_csv(out_dir / "slice_comparison_vs_exp004.csv", index=False)

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
    prev1_row = slice_df.loc[
        (slice_df.slice_family == "prev1") & (slice_df.slice_level == "prev1_<=0.35")
    ]
    prev1_low_delta = float(prev1_row["delta_raw_bss"].iloc[0]) if len(prev1_row) else None
    verdict, reason, conclusion = decide_verdict(deltas, prev1_low_delta)
    save_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "baseline": BASELINE_ID,
            "removed_feature": REMOVED_FEATURE,
            "feature_assert": feature_assert,
            "deltas": deltas,
            "prev1_low_delta": prev1_low_delta,
            "verdict": verdict,
            "verdict_reason": reason,
            "conclusion": conclusion,
            "protected_fingerprints_unchanged": True,
        },
        out_dir / "comparison_vs_exp004.json",
    )
    write_report(
        baseline_folds=baseline_folds,
        fe_folds=result["fold_metrics"],
        slice_df=slice_df,
        n_base_feat=len(used_b),
        n_fe_feat=len(used_f),
        feature_assert=feature_assert,
        verdict=verdict,
        verdict_reason=reason,
        conclusion=conclusion,
        fp_ok=True,
        report_path=ROOT / "reports" / "fe_002a_prev1_ablation.md",
    )
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "verdict": verdict, "conclusion": conclusion, **deltas, "prev1_low_delta": prev1_low_delta}, indent=2))


if __name__ == "__main__":
    main()
