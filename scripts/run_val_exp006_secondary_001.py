"""VAL-EXP006-SECONDARY-001: frozen EXP-006 vs untuned EXP-005 on untouched 2022/2023 folds.

Single changed factor: CatBoost hyperparameters.
No Optuna, no feature/weight changes, no 2024 re-scoring as selection criterion,
no overwrite of EXP-000..006 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from scripts.tree_exp_common import (
    build_feature_lists,
    positive_class_proba,
    prepare_native_frames,
    pred_stats,
)
from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.models import build_model
from src.utils import ensure_dir, load_config, resolve_path, save_json, set_seed

EXPERIMENT_ID = "VAL-EXP006-SECONDARY-001"
ID_DROP = ["pitcher_id", "batter_id"]
HOLD_WEIGHTS = {int(k): float(v) for k, v in SEASON_WEIGHTS.items()}
SEED = 42

# A. Untuned reference — EXP-005 / baseline.yaml catboost
UNTUNED_PARAMS: dict[str, Any] = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
}

# B. Frozen tuned candidate — EXP-006 best_params (no re-search)
TUNED_PARAMS: dict[str, Any] = {
    "depth": 7,
    "learning_rate": 0.028199996263578355,
    "l2_leaf_reg": 1.2151617026673374,
    "iterations": 361,
    "bagging_temperature": 0.9656320330745594,
    "loss_function": "Logloss",
}

MODELS = {
    "untuned_exp005": UNTUNED_PARAMS,
    "tuned_exp006": TUNED_PARAMS,
}

FOLDS = [
    {
        "fold_id": "secondary_2022",
        "train_seasons": [2019, 2020, 2021],
        "valid_seasons": [2022],
    },
    {
        "fold_id": "secondary_2023",
        "train_seasons": [2019, 2020, 2021, 2022],
        "valid_seasons": [2023],
    },
]

# Fixed asof sample-size bins (count features; not retuned)
ASOF_BINS = [-0.5, 0.5, 100, 500, 2000, np.inf]
ASOF_LABELS = ["0", "1-100", "101-500", "501-2000", "2000+"]

PROTECTED_ARTIFACTS = [
    "outputs/predictions/EXP-005/summary.json",
    "outputs/predictions/EXP-006/summary.json",
    "outputs/predictions/EXP-006/holdout_2024_predictions.csv",
    "outputs/models/EXP-005/R_L_keep_game_type/feature_meta.json",
    "outputs/models/EXP-006",
]


def file_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        h = hashlib.sha256()
        total = 0
        for f in files:
            data = f.read_bytes()
            total += len(data)
            h.update(str(f.relative_to(path)).encode())
            h.update(data)
        return {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "kind": "dir",
            "n_files": len(files),
            "nbytes": total,
            "sha256": h.hexdigest(),
            "mtime_ns": max((f.stat().st_mtime_ns for f in files), default=0),
        }
    data = path.read_bytes()
    st = path.stat()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "kind": "file",
        "n_files": 1,
        "nbytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime_ns": st.st_mtime_ns,
    }


def snapshot_protected() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rel in PROTECTED_ARTIFACTS:
        p = ROOT / rel
        if p.exists():
            out[rel] = file_fingerprint(p)
        else:
            out[rel] = {"path": rel, "missing": True}
    return out


def assert_protected_unchanged(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    changed = []
    for key in before:
        if before[key] != after.get(key):
            changed.append(key)
    return {
        "ok": len(changed) == 0,
        "changed": changed,
        "n_checked": len(before),
    }


def make_weights(seasons: np.ndarray, train_seasons: list[int]) -> np.ndarray:
    weight_map = {s: HOLD_WEIGHTS[s] for s in train_seasons}
    missing = sorted({int(s) for s in seasons if int(s) not in weight_map})
    if missing:
        raise ValueError(f"weight map missing seasons: {missing}")
    return np.asarray([float(weight_map[int(s)]) for s in seasons], dtype=float)


def fit_catboost(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    categorical_cols: list[str],
    params: dict[str, Any],
    seed: int,
    sample_weight: np.ndarray,
) -> Any:
    model = build_model("catboost", dict(params), seed)
    model.fit(X_tr, y_tr, cat_features=categorical_cols, sample_weight=sample_weight)
    return model


def quantile_10_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with quantile (equal-count) bins on predictions."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    if n == 0:
        raise ValueError("empty arrays for ECE")
    # rank-based quantile bins; handles duplicate probabilities
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


def asof_bin_series(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float).fillna(0.0),
        bins=ASOF_BINS,
        labels=ASOF_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)


def score_bundle(y_true: np.ndarray, pred: np.ndarray, scale: float) -> dict[str, Any]:
    metrics = evaluate_predictions(y_true, pred, scale=scale)
    return {
        "n": int(metrics["n_samples"]),
        "target_rate": float(metrics["target_rate"]),
        "brier_score": float(metrics["brier_score"]),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        "quantile_10_ece": quantile_10_ece(y_true, pred, n_bins=10),
        **pred_stats(pred),
    }


def subgroup_rows(
    *,
    model_name: str,
    fold_id: str,
    frame: pd.DataFrame,
    target_col: str,
    pred_col: str,
    scale: float,
    min_rows: int = 30,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("game_type", "game_type"),
        ("pitcher_status", "pitcher_status"),
        ("batter_status", "batter_status"),
        ("pitcher_asof_n_bin", "pitcher_asof_n_bin"),
        ("batter_asof_n_bin", "batter_asof_n_bin"),
    ]
    for subgroup_family, col in specs:
        for level, part in frame.groupby(col, dropna=False, sort=True):
            y = part[target_col].to_numpy(dtype=float)
            p = part[pred_col].to_numpy(dtype=float)
            row: dict[str, Any] = {
                "model": model_name,
                "fold_id": fold_id,
                "subgroup_family": subgroup_family,
                "subgroup_level": str(level),
                "n": int(len(part)),
            }
            if len(part) < min_rows:
                row.update(
                    {
                        "eligible": False,
                        "target_rate": float(np.mean(y)) if len(y) else np.nan,
                        "brier_score": np.nan,
                        "raw_brier_skill_score": np.nan,
                        "official_brier_skill_score": np.nan,
                        "quantile_10_ece": np.nan,
                        "reason": f"below_min_rows:{min_rows}",
                    }
                )
            else:
                try:
                    bundle = score_bundle(y, p, scale)
                    row.update(
                        {
                            "eligible": True,
                            "target_rate": bundle["target_rate"],
                            "brier_score": bundle["brier_score"],
                            "raw_brier_skill_score": bundle["raw_brier_skill_score"],
                            "official_brier_skill_score": bundle["official_brier_skill_score"],
                            "quantile_10_ece": bundle["quantile_10_ece"],
                            "reason": "ok",
                        }
                    )
                except ValueError as exc:
                    row.update(
                        {
                            "eligible": False,
                            "target_rate": float(np.mean(y)),
                            "brier_score": np.nan,
                            "raw_brier_skill_score": np.nan,
                            "official_brier_skill_score": np.nan,
                            "quantile_10_ece": np.nan,
                            "reason": str(exc),
                        }
                    )
            rows.append(row)
    return rows


def load_primary_reference() -> dict[str, Any]:
    """Read-only reference of existing 2024 primary scores (not re-run)."""
    exp005 = json.loads(
        (ROOT / "outputs/predictions/EXP-005/summary.json").read_text(encoding="utf-8")
    )
    exp006 = json.loads(
        (ROOT / "outputs/predictions/EXP-006/summary.json").read_text(encoding="utf-8")
    )
    return {
        "note": (
            "2024 was used for EXP-006 tuning; these are existing primary scores only. "
            "Not used as a new secondary validation fold or model-selection criterion."
        ),
        "untuned_exp005_primary_2024_raw_bss": float(
            exp005["S"]["S1_L_keep_game_type_primary_2024_raw_bss"]
        ),
        "tuned_exp006_primary_2024_raw_bss": float(exp006["catboost"]["best_raw_bss"]),
        "tuned_minus_untuned_primary_2024_raw_bss_delta": float(
            exp006["catboost"]["best_raw_bss"]
            - exp005["S"]["S1_L_keep_game_type_primary_2024_raw_bss"]
        ),
    }


def decide(
    paired: pd.DataFrame,
) -> dict[str, Any]:
    """Apply verdict rules on untouched folds only (2022, 2023)."""
    deltas = {
        r["fold_id"]: float(r["delta_raw_bss_tuned_minus_untuned"])
        for _, r in paired.iterrows()
    }
    ece_deltas = {
        r["fold_id"]: float(r["delta_quantile_10_ece_tuned_minus_untuned"])
        for _, r in paired.iterrows()
    }
    d22 = deltas["secondary_2022"]
    d23 = deltas["secondary_2023"]
    mean_delta = float(np.mean([d22, d23]))
    e22 = ece_deltas["secondary_2022"]
    e23 = ece_deltas["secondary_2023"]
    calib_not_worse = (e22 <= 0.0) and (e23 <= 0.0)
    both_positive = (d22 > 0.0) and (d23 > 0.0)
    both_nonpositive = (d22 <= 0.0) and (d23 <= 0.0)
    mixed = (d22 > 0.0) != (d23 > 0.0)

    if both_nonpositive:
        verdict = "reject_candidate"
        reason = "both untouched folds have tuned-untuned raw BSS delta <= 0"
    elif both_positive and mean_delta > 0.0 and calib_not_worse:
        # small gains may still be seed noise → note in reason; keep as improve_candidate
        verdict = "improve_candidate"
        reason = (
            "both folds positive raw-BSS delta, mean delta positive, "
            "quantile-10 ECE not worse on either fold"
        )
    elif mixed:
        verdict = "hold"
        reason = "delta signs differ across 2022 and 2023"
    elif both_positive and not calib_not_worse:
        verdict = "hold"
        reason = "raw BSS improved but calibration (quantile-10 ECE) worsened on at least one fold"
    else:
        verdict = "hold"
        reason = "does not meet improve or reject criteria cleanly"

    return {
        "verdict": verdict,
        "reason": reason,
        "delta_raw_bss_by_fold": deltas,
        "mean_delta_raw_bss": mean_delta,
        "delta_quantile_10_ece_by_fold": ece_deltas,
        "calibration_not_worse_both_folds": calib_not_worse,
        "both_folds_positive_raw_bss_delta": both_positive,
        "both_folds_nonpositive_raw_bss_delta": both_nonpositive,
    }


def main() -> None:
    wall0 = time.perf_counter()
    out_dir = resolve_path(f"outputs/predictions/{EXPERIMENT_ID}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already exists and is non-empty; refusing to overwrite."
        )
    ensure_dir(out_dir)

    before_fp = snapshot_protected()
    config = load_config("configs/baseline.yaml")
    set_seed(SEED)
    metric_settings = resolve_metric_settings(config.get("task", {}))
    scale = float(metric_settings["scale"])

    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    target_col = data_cfg["target_col"]
    id_col = data_cfg["id_col"]
    season_col = data_cfg["season_col"]
    exclude_cols = list(feat_cfg.get("exclude_cols") or [])
    categorical_cols_cfg = list(feat_cfg.get("categorical_cols") or [])

    loaded = load_competition_data(config)
    train_df = loaded["train"]
    _ = loaded["test"]

    used0, nums0, cats0 = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols_cfg,
    )
    used = [c for c in used0 if c not in ID_DROP]
    cats = [c for c in cats0 if c not in ID_DROP]
    nums = [c for c in nums0 if c in used]
    if len(used) != 42:
        raise AssertionError(f"Expected 42 L features, got {len(used)}")
    if "game_type" not in used:
        raise AssertionError("game_type must be kept")
    if any(c in used for c in ("pitcher_id", "batter_id", "li", "home_win_expectancy", "away_win_expectancy")):
        raise AssertionError("forbidden feature present in used set")

    fold_metric_rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    subgroup_parts: list[dict[str, Any]] = []
    join_checks: list[dict[str, Any]] = []

    for fold_spec in FOLDS:
        fold = build_season_holdout(
            train_df,
            season_col,
            fold_spec["train_seasons"],
            fold_spec["valid_seasons"],
            name=fold_spec["fold_id"],
        )
        train_idx = fold.train_idx
        valid_idx = fold.valid_idx
        train_ids_pitcher = set(train_df.iloc[train_idx]["pitcher_id"].astype(str))
        train_ids_batter = set(train_df.iloc[train_idx]["batter_id"].astype(str))

        X_tr_raw = train_df.iloc[train_idx][used]
        y_tr = train_df.iloc[train_idx][target_col]
        X_va_raw = train_df.iloc[valid_idx][used]
        y_va = train_df.iloc[valid_idx][target_col].to_numpy(dtype=float)
        X_tr, X_va = prepare_native_frames(
            X_tr_raw,
            X_va_raw,
            numeric_cols=nums,
            categorical_cols=cats,
            backend="catboost",
        )
        sw = make_weights(
            train_df.iloc[train_idx][season_col].to_numpy(),
            fold_spec["train_seasons"],
        )

        va_meta = train_df.iloc[valid_idx][
            [id_col, season_col, target_col, "game_type", "pitcher_id", "batter_id", "asof_pitcher_n", "asof_batter_n"]
        ].copy()
        va_meta["pitcher_status"] = np.where(
            va_meta["pitcher_id"].astype(str).isin(train_ids_pitcher), "existing", "new"
        )
        va_meta["batter_status"] = np.where(
            va_meta["batter_id"].astype(str).isin(train_ids_batter), "existing", "new"
        )
        va_meta["pitcher_asof_n_bin"] = asof_bin_series(va_meta["asof_pitcher_n"])
        va_meta["batter_asof_n_bin"] = asof_bin_series(va_meta["asof_batter_n"])

        n_before = int(len(va_meta))
        # join integrity vs train on row_id
        train_key = train_df[[id_col, target_col]].drop_duplicates(subset=[id_col])
        joined = va_meta.merge(train_key, on=id_col, how="left", suffixes=("", "_train"))
        n_after = int(len(joined))
        unmatched = int(joined[f"{target_col}_train"].isna().sum())
        target_align = bool(
            (joined[target_col].to_numpy() == joined[f"{target_col}_train"].to_numpy()).all()
        )
        row_id_unique = bool(va_meta[id_col].is_unique)
        join_checks.append(
            {
                "fold_id": fold_spec["fold_id"],
                "n_rows_before_join": n_before,
                "n_rows_after_join": n_after,
                "row_count_preserved": n_before == n_after,
                "unmatched_rows": unmatched,
                "row_id_unique_in_fold_valid": row_id_unique,
                "target_alignment_vs_train": target_align,
            }
        )
        if not (n_before == n_after and unmatched == 0 and row_id_unique and target_align):
            raise RuntimeError(f"join verification failed for {fold_spec['fold_id']}: {join_checks[-1]}")

        for model_name, params in MODELS.items():
            t0 = time.perf_counter()
            print(
                f"Fitting {model_name} on {fold_spec['fold_id']} "
                f"(train={fold_spec['train_seasons']} → valid={fold_spec['valid_seasons']})...",
                flush=True,
            )
            set_seed(SEED)
            model = fit_catboost(X_tr, y_tr, cats, params, SEED, sample_weight=sw)
            pred, pos_index = positive_class_proba(model, X_va)
            bundle = score_bundle(y_va, pred, scale)
            elapsed = float(time.perf_counter() - t0)
            print(
                f"  {model_name} {fold_spec['fold_id']}: raw_bss={bundle['raw_brier_skill_score']:.3f} "
                f"ece10q={bundle['quantile_10_ece']:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )

            fold_metric_rows.append(
                {
                    "model": model_name,
                    "fold_id": fold_spec["fold_id"],
                    "train_seasons": json.dumps(fold_spec["train_seasons"]),
                    "valid_seasons": json.dumps(fold_spec["valid_seasons"]),
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "sample_weight_by_season": json.dumps(
                        {str(s): HOLD_WEIGHTS[s] for s in fold_spec["train_seasons"]},
                        ensure_ascii=False,
                    ),
                    "params": json.dumps(params, ensure_ascii=False),
                    "positive_class_index": pos_index,
                    "elapsed_seconds": elapsed,
                    **bundle,
                }
            )

            oof = va_meta[
                [
                    id_col,
                    season_col,
                    target_col,
                    "game_type",
                    "pitcher_id",
                    "batter_id",
                    "asof_pitcher_n",
                    "asof_batter_n",
                    "pitcher_status",
                    "batter_status",
                    "pitcher_asof_n_bin",
                    "batter_asof_n_bin",
                ]
            ].copy()
            oof["model"] = model_name
            oof["fold_id"] = fold_spec["fold_id"]
            oof["prediction"] = pred
            oof_parts.append(oof)

            sub_frame = oof.copy()
            subgroup_parts.extend(
                subgroup_rows(
                    model_name=model_name,
                    fold_id=fold_spec["fold_id"],
                    frame=sub_frame,
                    target_col=target_col,
                    pred_col="prediction",
                    scale=scale,
                )
            )

    oof_df = pd.concat(oof_parts, ignore_index=True)
    # OOF uniqueness: exactly one row per (model, row_id)
    dup_model_row = int(oof_df.duplicated(subset=["model", id_col]).sum())
    if dup_model_row != 0:
        raise RuntimeError(f"OOF has duplicate (model, row_id): {dup_model_row}")
    # Within each model, row_id must be unique across folds (2022∩2023 empty)
    for model_name, part in oof_df.groupby("model"):
        if not part[id_col].is_unique:
            raise RuntimeError(f"row_id not unique within model={model_name}")

    # Global join verification summary
    train_map = train_df.set_index(id_col)[target_col]
    aligned = oof_df[id_col].map(train_map)
    oof_target_match = bool((aligned.to_numpy() == oof_df[target_col].to_numpy()).all())
    oof_map_na = int(aligned.isna().sum())

    join_verification = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "oof_n_rows": int(len(oof_df)),
        "oof_n_rows_expected": int(
            sum(
                len(train_df[train_df[season_col].isin(f["valid_seasons"])])
                for f in FOLDS
            )
            * len(MODELS)
        ),
        "duplicate_model_row_id_count": dup_model_row,
        "model_row_id_unique_ok": dup_model_row == 0,
        "target_alignment_vs_train_control_success": oof_target_match,
        "train_map_na": oof_map_na,
        "note_exp006_long_oof": (
            "This OOF is newly generated per (model, row_id); "
            "EXP-006 long enriched OOF duplicates were not used as independent samples."
        ),
        "per_fold": join_checks,
    }
    if join_verification["oof_n_rows"] != join_verification["oof_n_rows_expected"]:
        raise RuntimeError(
            f"OOF row count mismatch: {join_verification['oof_n_rows']} vs "
            f"{join_verification['oof_n_rows_expected']}"
        )
    if not (oof_target_match and oof_map_na == 0):
        raise RuntimeError("OOF target alignment failed")

    fold_metrics = pd.DataFrame(fold_metric_rows)
    # Paired deltas: tuned - untuned per fold
    paired_rows = []
    for fold_id in [f["fold_id"] for f in FOLDS]:
        u = fold_metrics[
            (fold_metrics["model"] == "untuned_exp005") & (fold_metrics["fold_id"] == fold_id)
        ].iloc[0]
        t = fold_metrics[
            (fold_metrics["model"] == "tuned_exp006") & (fold_metrics["fold_id"] == fold_id)
        ].iloc[0]
        paired_rows.append(
            {
                "fold_id": fold_id,
                "n": int(u["n"]),
                "untuned_raw_bss": float(u["raw_brier_skill_score"]),
                "tuned_raw_bss": float(t["raw_brier_skill_score"]),
                "delta_raw_bss_tuned_minus_untuned": float(
                    t["raw_brier_skill_score"] - u["raw_brier_skill_score"]
                ),
                "untuned_official_bss": float(u["official_brier_skill_score"]),
                "tuned_official_bss": float(t["official_brier_skill_score"]),
                "delta_official_bss_tuned_minus_untuned": float(
                    t["official_brier_skill_score"] - u["official_brier_skill_score"]
                ),
                "untuned_brier": float(u["brier_score"]),
                "tuned_brier": float(t["brier_score"]),
                "delta_brier_tuned_minus_untuned": float(t["brier_score"] - u["brier_score"]),
                "untuned_quantile_10_ece": float(u["quantile_10_ece"]),
                "tuned_quantile_10_ece": float(t["quantile_10_ece"]),
                "delta_quantile_10_ece_tuned_minus_untuned": float(
                    t["quantile_10_ece"] - u["quantile_10_ece"]
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    decision = decide(paired)
    primary_ref = load_primary_reference()

    subgroup_metrics = pd.DataFrame(subgroup_parts)

    # Writes
    fold_path = out_dir / "fold_metrics.csv"
    paired_path = out_dir / "paired_delta.csv"
    subgroup_path = out_dir / "subgroup_metrics.csv"
    oof_parquet = out_dir / "oof_predictions.parquet"
    oof_csv = out_dir / "oof_predictions.csv"
    join_path = out_dir / "join_verification.json"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"

    fold_metrics.to_csv(fold_path, index=False)
    paired.to_csv(paired_path, index=False)
    subgroup_metrics.to_csv(subgroup_path, index=False)
    try:
        oof_df.to_parquet(oof_parquet, index=False)
        oof_written = str(oof_parquet)
    except Exception:
        oof_df.to_csv(oof_csv, index=False)
        oof_written = str(oof_csv)
    else:
        # also keep csv for easy inspection (same content; not overwriting existing EXP)
        oof_df.to_csv(oof_csv, index=False)

    after_fp = snapshot_protected()
    protected_check = assert_protected_unchanged(before_fp, after_fp)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": (
            "EXP-006 tuned CatBoost params improve raw BSS vs EXP-005 untuned "
            "on untouched 2022 and 2023 seasons under identical L features and weights."
        ),
        "changed_factor": "catboost_hyperparameters_only",
        "seed": SEED,
        "data_version": config.get("project", {}).get("data_version"),
        "n_features": len(used),
        "features_used": used,
        "numeric_cols": nums,
        "categorical_cols": cats,
        "dropped_ids": ID_DROP,
        "exclude_cols": exclude_cols,
        "sample_weight_schedule": {str(k): v for k, v in sorted(HOLD_WEIGHTS.items())},
        "models": {
            "untuned_exp005": UNTUNED_PARAMS,
            "tuned_exp006": TUNED_PARAMS,
        },
        "folds": FOLDS,
        "fold_metrics": fold_metric_rows,
        "paired_delta": paired_rows,
        "mean_delta_raw_bss_untouched": float(paired["delta_raw_bss_tuned_minus_untuned"].mean()),
        "decision": decision,
        "primary_2024_reference_only": primary_ref,
        "join_verification": join_verification,
        "protected_artifacts_unchanged": protected_check,
        "paths": {
            "fold_metrics": str(fold_path),
            "paired_delta": str(paired_path),
            "subgroup_metrics": str(subgroup_path),
            "oof_predictions": oof_written,
            "oof_predictions_csv": str(oof_csv),
            "join_verification": str(join_path),
            "report": str(report_path),
        },
        "elapsed_seconds_wall": float(time.perf_counter() - wall0),
        "prohibitions_honored": [
            "no_optuna",
            "no_feature_change",
            "no_we_li",
            "no_weight_schedule_change",
            "no_ensemble_stacking_submission",
            "no_overwrite_exp000_to_006",
            "no_2024_as_new_secondary_or_selection",
        ],
    }
    save_json(join_verification, join_path)
    save_json(summary, summary_path)

    # Report
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- changed factor: **CatBoost hyperparameters only**",
        f"- seed: `{SEED}`",
        f"- features: `{len(used)}` L (drop pitcher_id/batter_id, keep game_type)",
        f"- weights: `{HOLD_WEIGHTS}` applied only to train seasons in each fold",
        f"- verdict: **`{decision['verdict']}`** — {decision['reason']}",
        "",
        "## Untouched fold results (model × fold)",
        "",
        "| model | fold | n | target rate | Brier | raw BSS | official BSS | q10 ECE | pred mean±std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in fold_metric_rows:
        lines.append(
            f"| {r['model']} | {r['fold_id']} | {r['n']} | {r['target_rate']:.4f} | "
            f"{r['brier_score']:.6f} | {r['raw_brier_skill_score']:.3f} | "
            f"{r['official_brier_skill_score']:.3f} | {r['quantile_10_ece']:.6f} | "
            f"{r['prediction_mean']:.4f}±{r['prediction_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired delta (tuned − untuned)",
            "",
            "| fold | Δ raw BSS | Δ official BSS | Δ Brier | Δ q10 ECE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, r in paired.iterrows():
        lines.append(
            f"| {r['fold_id']} | {r['delta_raw_bss_tuned_minus_untuned']:.3f} | "
            f"{r['delta_official_bss_tuned_minus_untuned']:.3f} | "
            f"{r['delta_brier_tuned_minus_untuned']:.6f} | "
            f"{r['delta_quantile_10_ece_tuned_minus_untuned']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- mean Δ raw BSS (2022+2023): `{summary['mean_delta_raw_bss_untouched']:.3f}`",
            "",
            "## 2024 primary reference only (not a selection fold)",
            "",
            f"- untuned EXP-005 S1 keep-game_type raw BSS: `{primary_ref['untuned_exp005_primary_2024_raw_bss']:.6f}`",
            f"- tuned EXP-006 CatBoost raw BSS: `{primary_ref['tuned_exp006_primary_2024_raw_bss']:.6f}`",
            f"- delta: `{primary_ref['tuned_minus_untuned_primary_2024_raw_bss_delta']:.6f}`",
            f"- note: {primary_ref['note']}",
            "",
            "## Join verification",
            "",
            f"- oof rows: `{join_verification['oof_n_rows']}` (expected `{join_verification['oof_n_rows_expected']}`)",
            f"- (model, row_id) unique: `{join_verification['model_row_id_unique_ok']}`",
            f"- target aligned to train: `{join_verification['target_alignment_vs_train_control_success']}`",
            f"- protected EXP artifacts unchanged: `{protected_check['ok']}`",
            "",
            "## Subgroup families",
            "",
            "- game_type",
            "- pitcher_status (existing/new vs train fold IDs)",
            "- batter_status (existing/new vs train fold IDs)",
            "- pitcher_asof_n_bin: `0`, `1-100`, `101-500`, `501-2000`, `2000+`",
            "- batter_asof_n_bin: same bins",
            "",
            f"See `{subgroup_path.name}` for full rows.",
            "",
            "## Params",
            "",
            f"- untuned: `{json.dumps(UNTUNED_PARAMS)}`",
            f"- tuned: `{json.dumps(TUNED_PARAMS)}`",
            "",
            f"- wall time (s): `{summary['elapsed_seconds_wall']:.1f}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    if not protected_check["ok"]:
        raise RuntimeError(f"Protected artifacts changed: {protected_check['changed']}")

    print(json.dumps({"summary_path": str(summary_path), "decision": decision}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
