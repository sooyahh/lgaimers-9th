"""MODEL-TRANSFER-AUDIT-001: are the already-run FE deltas CatBoost-specific?

For each FE variant this re-measures the **paired** delta against the same
baseline under two backends (CatBoost / LightGBM) on the same locked folds and
the same seeds, then reports whether the two backends agree in sign and whether
either delta is larger than its own paired-delta noise.

Why the L feature set is the default: EXP-002 (LightGBM, raw `pitcher_id` /
`batter_id` as native cats) scored 8.2 while EXP-003 (CatBoost, same 44 columns)
scored 562.3. That gap is dominated by high-cardinality categorical *encoding*,
not by the model family, so a transfer audit on the 44-column set would measure
the encoding rather than the feature. Dropping the two ID columns removes the
confound; `--feature-set full44` is available only as a labelled contrast.

Scope guards:
- results only; no adopt/reject.
- does **not** append to `outputs/experiments.csv`.
- does **not** write into `outputs/predictions/EXP-*` or `outputs/predictions/FE-*`.
- no test.csv, no 2025 Trackman, no target encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fe002_prev1_shrinkage import SHRINKAGE_K, replace_prev1_with_shrinkage
from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from scripts.tree_exp_common import (
    build_feature_lists,
    positive_class_proba,
    prepare_native_frames,
)
from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.models import build_model
from src.trackman_profiles import (
    TM_EXPECTED_REL_SPEED,
    build_group_rel_speed_profile,
    compute_tm_expected_rel_speed,
    filter_trackman_history,
    profile_means,
)
from src.utils import ensure_dir, load_config, resolve_path, set_seed

AUDIT_ID = "MODEL-TRANSFER-AUDIT-001"
DEFAULT_CONFIG = "configs/baseline.yaml"
ID_DROP = ("pitcher_id", "batter_id")
TRACKMAN_PATH = ROOT / "data/raw/trackman_history.csv"
DEFAULT_WEIGHTS = {int(k): float(v) for k, v in SEASON_WEIGHTS.items()}
DEFAULT_SEEDS = (42, 1, 7)
DEFAULT_BACKENDS = ("catboost", "lightgbm")
BASE_VARIANT = "BASE"
# EXP-004 baseline params per backend; kept at the logged EXP-002/EXP-003 values so a
# variant delta is not confounded by a hyperparameter change.
BACKEND_PARAMS_KEY = {"catboost": "catboost", "lightgbm": "lightgbm"}
# Observed CatBoost cost on the 2024 holdout, 42 features (outputs/eda/seed_variance.md).
OBSERVED_CATBOOST_SECONDS_PER_FIT = (130.0, 190.0)


@dataclass(frozen=True)
class Variant:
    """One feature-engineering change, expressed model-agnostically."""

    name: str
    description: str
    source_experiment: str
    drop_cols: tuple[str, ...] = ()
    transform: str | None = None
    extra_numeric: tuple[str, ...] = ()
    needs_trackman: bool = False


VARIANTS: dict[str, Variant] = {
    BASE_VARIANT: Variant(
        name=BASE_VARIANT,
        description="EXP-004 feature set, unchanged (paired reference)",
        source_experiment="EXP-004",
    ),
    "FE-001A": Variant(
        name="FE-001A",
        description="drop asof_pitcher_ball_rate",
        source_experiment="FE-001A",
        drop_cols=("asof_pitcher_ball_rate",),
    ),
    "FE-002A": Variant(
        name="FE-002A",
        description="drop asof_pitcher_prev1_game_success_rate",
        source_experiment="FE-002A",
        drop_cols=("asof_pitcher_prev1_game_success_rate",),
    ),
    "FE-002": Variant(
        name="FE-002",
        description=f"row-local prev1 shrinkage toward long-term rate, k={SHRINKAGE_K}",
        source_experiment="FE-002",
        transform="fe002_prev1_shrinkage",
    ),
    "TM-002A": Variant(
        name="TM-002A",
        description="add tm_expected_rel_speed (prior-season Trackman group means x as-of mix)",
        source_experiment="TM-002A",
        extra_numeric=(TM_EXPECTED_REL_SPEED,),
        needs_trackman=True,
    ),
}
DEFAULT_VARIANTS = (BASE_VARIANT, "FE-001A", "FE-002A", "FE-002")


@dataclass
class FoldSpec:
    fold_id: str
    role: str
    train_seasons: list[int]
    valid_seasons: list[int]


# --------------------------------------------------------------------------- setup


def build_fold_specs(cv_cfg: dict[str, Any]) -> list[FoldSpec]:
    primary = cv_cfg["primary"]
    secondary = list(cv_cfg.get("secondary_folds") or [])
    if len(secondary) < 2:
        raise ValueError("Expected the two locked secondary folds in cv.secondary_folds")
    specs = [
        FoldSpec("secondary_2022", "secondary", list(secondary[0]["train_seasons"]), list(secondary[0]["valid_seasons"])),
        FoldSpec("secondary_2023", "secondary", list(secondary[1]["train_seasons"]), list(secondary[1]["valid_seasons"])),
        FoldSpec("primary_2024", "primary", list(primary["train_seasons"]), list(primary["valid_seasons"])),
    ]
    for spec in specs:
        if len(spec.valid_seasons) != 1:
            raise AssertionError(f"{spec.fold_id} must have exactly one valid season")
    return specs


def base_feature_lists(
    train_df: pd.DataFrame,
    *,
    target_col: str,
    exclude_cols: list[str],
    categorical_cols: list[str],
    feature_set: str,
) -> tuple[list[str], list[str], list[str]]:
    used, nums, cats = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols,
    )
    if feature_set == "L42":
        used = [c for c in used if c not in ID_DROP]
        cats = [c for c in cats if c not in ID_DROP]
        nums = [c for c in nums if c in used]
        leaked = [c for c in ID_DROP if c in used]
        if leaked:
            raise AssertionError(f"L42 feature set still contains {leaked}")
        if "game_type" not in used:
            raise AssertionError("L42 feature set is expected to keep game_type")
    elif feature_set != "full44":
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    return used, nums, cats


def subsample_by_season(
    df: pd.DataFrame, season_col: str, rows_per_season: int, seed: int = 42
) -> pd.DataFrame:
    """Smoke-mode only. Keeps every season so the fold structure survives."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby(season_col, sort=True):
        if len(group) <= rows_per_season:
            parts.append(group)
            continue
        picked = np.sort(rng.choice(len(group), size=rows_per_season, replace=False))
        parts.append(group.iloc[picked])
    return pd.concat(parts, axis=0).reset_index(drop=True)


# ----------------------------------------------------------------------- variants


def materialize_variant(base_df: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    """Apply a value-level transform. Column adds are fold-local (see fold_extra_frame)."""
    if variant.transform is None:
        return base_df
    if variant.transform == "fe002_prev1_shrinkage":
        return replace_prev1_with_shrinkage(base_df, k=SHRINKAGE_K)
    raise ValueError(f"Unknown transform: {variant.transform}")


def make_fold_extra_frame(
    variant: Variant, tm: pd.DataFrame | None
) -> Callable[[pd.DataFrame, FoldSpec], tuple[pd.DataFrame, dict[str, Any]]] | None:
    """Fold-local injector for variants that add a column derived from prior seasons."""
    if not variant.extra_numeric:
        return None
    if not variant.needs_trackman:
        raise ValueError(f"{variant.name} declares extra_numeric without a builder")
    if tm is None:
        raise ValueError(f"{variant.name} needs Trackman history but none was loaded")

    def fold_extra_frame(frame: pd.DataFrame, spec: FoldSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
        valid_season = int(spec.valid_seasons[0])
        hist = filter_trackman_history(tm, valid_season)
        used_seasons = sorted(int(s) for s in hist["season"].dropna().unique())
        if any(s >= valid_season for s in used_seasons):
            raise AssertionError(f"{spec.fold_id} Trackman profile includes valid/future seasons")
        profile = build_group_rel_speed_profile(hist, valid_season=valid_season)
        means = profile_means(profile)
        values = compute_tm_expected_rel_speed(frame, means)
        extra = pd.DataFrame({TM_EXPECTED_REL_SPEED: values}, index=frame.index)
        info = {
            "trackman_seasons_used": used_seasons,
            "group_means": {k: float(v) for k, v in means.items()},
            "missing_rate": float(extra[TM_EXPECTED_REL_SPEED].isna().mean()),
        }
        return extra, info

    return fold_extra_frame


def variant_feature_lists(
    variant: Variant,
    *,
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    train_columns: list[str],
) -> tuple[list[str], list[str], list[str]]:
    missing = [c for c in variant.drop_cols if c not in train_columns]
    if missing:
        raise AssertionError(f"{variant.name} drop_cols not present in train: {missing}")
    v_used = [c for c in used if c not in variant.drop_cols]
    v_nums = [c for c in numeric_cols if c not in variant.drop_cols]
    v_cats = [c for c in categorical_cols if c not in variant.drop_cols]
    for col in variant.extra_numeric:
        if col in v_used:
            raise AssertionError(f"{variant.name} extra column already present: {col}")
        if col in train_columns:
            raise AssertionError(f"{variant.name} extra column must be fold-injected: {col}")
        v_used.append(col)
        v_nums.append(col)
    return v_used, v_nums, v_cats


# ---------------------------------------------------------------------------- fit


def fit_backend(
    backend: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    categorical_cols: list[str],
    params: dict[str, Any],
    seed: int,
    sample_weight: np.ndarray | None,
) -> Any:
    if backend == "catboost":
        fit_params = {**params, "random_seed": seed, "verbose": False, "allow_writing_files": False}
        model = build_model("catboost", fit_params, seed)
        model.fit(X_tr, y_tr, cat_features=categorical_cols, sample_weight=sample_weight)
        return model
    if backend == "lightgbm":
        fit_params = {**params, "random_state": seed, "verbosity": -1}
        model = build_model("lightgbm", fit_params, seed)
        model.fit(
            X_tr,
            y_tr,
            categorical_feature=categorical_cols or "auto",
            sample_weight=sample_weight,
        )
        return model
    raise ValueError(f"Unsupported backend: {backend}")


def run_cell(
    *,
    train_df: pd.DataFrame,
    spec: FoldSpec,
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    id_col: str,
    backend: str,
    params: dict[str, Any],
    seed: int,
    scale: float,
    weight_map: dict[int, float],
    extra_numeric: tuple[str, ...],
    fold_extra_frame: Callable[[pd.DataFrame, FoldSpec], tuple[pd.DataFrame, dict[str, Any]]] | None,
    keep_pred: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    set_seed(seed)
    fold = build_season_holdout(
        train_df, season_col, spec.train_seasons, spec.valid_seasons, name=spec.fold_id
    )

    core_cols = [c for c in used if c not in extra_numeric]
    X_tr_raw = train_df.iloc[fold.train_idx][core_cols]
    X_va_raw = train_df.iloc[fold.valid_idx][core_cols]
    y_tr = train_df.iloc[fold.train_idx][target_col]
    y_va = train_df.iloc[fold.valid_idx][target_col]

    fold_extra_info: dict[str, Any] = {}
    if fold_extra_frame is not None:
        extra_df, fold_extra_info = fold_extra_frame(train_df, spec)
        if not extra_df.index.equals(train_df.index):
            raise AssertionError("fold_extra_frame index must match the training frame")
        leaked = [c for c in extra_numeric if c in train_df.columns]
        if leaked:
            raise AssertionError(f"fold_extra_frame mutated the shared frame with {leaked}")
        X_tr_raw = pd.concat([X_tr_raw, extra_df.iloc[fold.train_idx][list(extra_numeric)]], axis=1)
        X_va_raw = pd.concat([X_va_raw, extra_df.iloc[fold.valid_idx][list(extra_numeric)]], axis=1)
    X_tr_raw = X_tr_raw[used]
    X_va_raw = X_va_raw[used]

    X_tr, X_va = prepare_native_frames(
        X_tr_raw,
        X_va_raw,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        backend=backend,
    )

    seasons = train_df.iloc[fold.train_idx][season_col].to_numpy()
    missing = sorted({int(s) for s in seasons if int(s) not in weight_map})
    if missing:
        raise ValueError(f"weight_map missing train seasons {missing} in fold {spec.fold_id}")
    sample_weight = np.asarray([float(weight_map[int(s)]) for s in seasons], dtype=float)

    model = fit_backend(backend, X_tr, y_tr, categorical_cols, params, seed, sample_weight)
    pred, pos_index = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    elapsed = float(time.perf_counter() - t0)

    row: dict[str, Any] = {
        "fold_id": spec.fold_id,
        "role": spec.role,
        "backend": backend,
        "seed": int(seed),
        "n_features": len(used),
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
        "valid_target_rate": float(metrics["target_rate"]),
        "brier_score": float(metrics["brier_score"]),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "positive_class_index": int(pos_index),
        "categorical_encoding": f"native_{backend}",
        "elapsed_seconds": elapsed,
    }
    if fold_extra_info:
        row["fold_extra"] = fold_extra_info
    if keep_pred:
        row["_pred"] = np.asarray(pred, dtype=float)
        row["_y_true"] = np.asarray(y_va, dtype=float)
        row["_row_id"] = train_df.iloc[fold.valid_idx][id_col].to_numpy()
    return row


# -------------------------------------------------------------------------- cache


def cell_key(
    *, feature_set: str, variant: str, backend: str, fold_id: str, seed: int, params: dict[str, Any]
) -> str:
    payload = json.dumps(
        {
            "feature_set": feature_set,
            "variant": variant,
            "backend": backend,
            "fold_id": fold_id,
            "seed": int(seed),
            "params": params,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        cache[record["cell_key"]] = record
    return cache


def append_cache(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- aggregation


def paired_deltas(
    rows: list[dict[str, Any]], *, variant: str, backend: str
) -> list[dict[str, Any]]:
    """Delta vs BASE at identical (backend, fold, seed). Unpaired cells are dropped."""
    base = {
        (r["fold_id"], r["seed"]): r
        for r in rows
        if r["variant"] == BASE_VARIANT and r["backend"] == backend
    }
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["variant"] != variant or r["backend"] != backend:
            continue
        ref = base.get((r["fold_id"], r["seed"]))
        if ref is None:
            continue
        out.append(
            {
                "fold_id": r["fold_id"],
                "seed": r["seed"],
                "base_raw_bss": float(ref["raw_brier_skill_score"]),
                "variant_raw_bss": float(r["raw_brier_skill_score"]),
                "delta_raw_bss": float(r["raw_brier_skill_score"] - ref["raw_brier_skill_score"]),
            }
        )
    return out


def summarize_deltas(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold means are the unit of observation; seed spread is reported separately.

    Seeds are nested inside folds, so pooling folds x seeds as if they were
    independent understates the standard error — and it does so by a different
    amount per backend, because a backend can be deterministic in the seed
    (LightGBM is, under the configured params: no subsample / no colsample).
    Equal-weighting the folds also matches the `dmean` convention used by the
    existing FE-001A / FE-002A / TM-002A reports.
    """
    values = np.asarray([d["delta_raw_bss"] for d in deltas], dtype=float)
    n_cells = int(values.size)

    by_fold: dict[str, float] = {}
    within_fold_std: list[float] = []
    for fold_id in sorted({d["fold_id"] for d in deltas}):
        fold_values = np.asarray(
            [d["delta_raw_bss"] for d in deltas if d["fold_id"] == fold_id], dtype=float
        )
        by_fold[fold_id] = float(np.mean(fold_values))
        if fold_values.size > 1:
            within_fold_std.append(float(np.std(fold_values, ddof=1)))

    fold_means = np.asarray(list(by_fold.values()), dtype=float)
    n_folds = int(fold_means.size)
    mean = float(np.mean(fold_means)) if n_folds else float("nan")
    std = float(np.std(fold_means, ddof=1)) if n_folds > 1 else float("nan")
    sem = float(std / np.sqrt(n_folds)) if n_folds > 1 else float("nan")
    seed_spread = float(np.mean(within_fold_std)) if within_fold_std else float("nan")

    if n_folds < 2 or not np.isfinite(sem) or sem <= 0.0:
        materiality = "n/a"
    elif abs(mean) > 2.0 * sem:
        materiality = "MATERIAL"
    else:
        materiality = "WITHIN_NOISE"

    return {
        "n_paired_cells": n_cells,
        "n_folds": n_folds,
        "delta_mean": mean,
        "delta_std_across_folds": std,
        "delta_sem": sem,
        "seed_spread_within_fold": seed_spread,
        "seed_axis_informative": bool(np.isfinite(seed_spread) and seed_spread > 0.0),
        "delta_min_cell": float(np.min(values)) if n_cells else float("nan"),
        "delta_max_cell": float(np.max(values)) if n_cells else float("nan"),
        "materiality": materiality,
        "materiality_rule": (
            "|delta_mean| > 2 * sem, sem = std(fold means) / sqrt(n_folds) "
            "(descriptive heuristic, not a hypothesis test)"
        ),
        "delta_mean_by_fold": by_fold,
    }


def transfer_verdict(per_backend: dict[str, dict[str, Any]]) -> dict[str, Any]:
    backends = sorted(per_backend)
    means = {b: per_backend[b]["delta_mean"] for b in backends}
    materiality = {b: per_backend[b]["materiality"] for b in backends}
    finite = [m for m in means.values() if np.isfinite(m)]
    sign_agreement = (
        bool(len(finite) == len(means) and len({int(np.sign(m)) for m in finite}) <= 1)
        if finite
        else False
    )

    fold_ids = sorted({f for b in backends for f in per_backend[b]["delta_mean_by_fold"]})
    fold_agree = 0
    fold_compared = 0
    for fold_id in fold_ids:
        signs = {
            int(np.sign(per_backend[b]["delta_mean_by_fold"][fold_id]))
            for b in backends
            if fold_id in per_backend[b]["delta_mean_by_fold"]
        }
        if len(signs) == 0:
            continue
        fold_compared += 1
        if len(signs) == 1:
            fold_agree += 1

    material_backends = [b for b in backends if materiality[b] == "MATERIAL"]
    if len(material_backends) == 0:
        verdict = "INCONCLUSIVE_NOISE"
        reading = "No backend moved beyond its own paired-delta noise; this is a measurement result, not a feature result."
    elif len(material_backends) == len(backends):
        if sign_agreement:
            verdict = "TRANSFERS"
            reading = "Material in every backend and same sign; the FE effect is not CatBoost-specific."
        else:
            verdict = "MODEL_SPECIFIC"
            reading = "Material in every backend but signs disagree; the FE effect depends on the model family."
    else:
        verdict = "PARTIAL_" + "_".join(material_backends)
        reading = "Material in some backends only; treat as weak evidence and add seeds before deciding."

    return {
        "verdict": verdict,
        "reading": reading,
        "sign_agreement_on_overall_mean": sign_agreement,
        "fold_sign_agreement": f"{fold_agree}/{fold_compared}",
        "delta_mean_by_backend": means,
        "materiality_by_backend": materiality,
    }


def backend_gap_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """BASE raw BSS per backend per fold, averaged over seeds."""
    out: list[dict[str, Any]] = []
    base_rows = [r for r in rows if r["variant"] == BASE_VARIANT]
    for fold_id in sorted({r["fold_id"] for r in base_rows}):
        entry: dict[str, Any] = {"fold_id": fold_id}
        for backend in sorted({r["backend"] for r in base_rows}):
            values = [
                r["raw_brier_skill_score"]
                for r in base_rows
                if r["fold_id"] == fold_id and r["backend"] == backend
            ]
            entry[backend] = float(np.mean(values)) if values else float("nan")
            entry[f"{backend}_n_seeds"] = len(values)
        if "catboost" in entry and "lightgbm" in entry:
            entry["gap_catboost_minus_lightgbm"] = float(entry["catboost"] - entry["lightgbm"])
        out.append(entry)
    return out


def cross_backend_agreement(
    preds: dict[tuple[str, str, str, int], dict[str, Any]], scale: float
) -> dict[str, Any] | None:
    """Auxiliary: BASE CatBoost vs BASE LightGBM correlation and a naive 50/50 blend."""
    keys = [k for k in preds if k[0] == BASE_VARIANT]
    fold_ids = sorted({k[2] for k in keys})
    out: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        seeds = sorted(
            {k[3] for k in keys if k[2] == fold_id and k[1] == "catboost"}
            & {k[3] for k in keys if k[2] == fold_id and k[1] == "lightgbm"}
        )
        if not seeds:
            continue
        seed = seeds[0]
        cb = preds[(BASE_VARIANT, "catboost", fold_id, seed)]
        lgb = preds[(BASE_VARIANT, "lightgbm", fold_id, seed)]
        if not np.array_equal(cb["row_id"], lgb["row_id"]):
            raise AssertionError(f"{fold_id} row_id order differs between backends")
        y_true = cb["y_true"]
        blend = np.clip(0.5 * cb["pred"] + 0.5 * lgb["pred"], 0.0, 1.0)
        out.append(
            {
                "fold_id": fold_id,
                "seed": seed,
                "pearson_corr": float(np.corrcoef(cb["pred"], lgb["pred"])[0, 1]),
                "catboost_raw_bss": float(
                    evaluate_predictions(y_true, cb["pred"], scale=scale)["raw_brier_skill_score"]
                ),
                "lightgbm_raw_bss": float(
                    evaluate_predictions(y_true, lgb["pred"], scale=scale)["raw_brier_skill_score"]
                ),
                "blend_50_50_raw_bss": float(
                    evaluate_predictions(y_true, blend, scale=scale)["raw_brier_skill_score"]
                ),
            }
        )
    return {"note": "single seed per fold; ensemble input only, not an adopt signal", "folds": out} if out else None


# ------------------------------------------------------------------------- report


def build_markdown(payload: dict[str, Any]) -> str:
    setup = payload["setup"]
    lines = [
        f"# {AUDIT_ID} — FE transferability across model families",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- feature set: `{setup['feature_set']}` ({setup['n_features_base']} base features)",
        f"- backends: `{setup['backends']}`",
        f"- seeds: `{setup['seeds']}`",
        f"- folds: `{[f['fold_id'] for f in setup['folds']]}`",
        f"- season sample weights: `{setup['sample_weight_by_season']}`",
        f"- params per backend: `{json.dumps(setup['params'], ensure_ascii=False)}`",
        f"- fits executed / reused: `{payload['n_fits_executed']}` / `{payload['n_fits_reused']}`",
        "- **results only — no adopt/reject; no submission; not written to `outputs/experiments.csv`**",
        "",
    ]
    if setup["feature_set"] == "full44":
        lines.extend(
            [
                "> **Contrast run.** `full44` keeps `pitcher_id` / `batter_id` as native categoricals.",
                "> LightGBM is known to degrade on that encoding (EXP-002 = 8.2 vs EXP-003 = 562.3),",
                "> so a cross-backend delta here measures encoding, not the feature. Read `L42` for the audit.",
                "",
            ]
        )
    if setup["smoke"]:
        lines.extend(["> **Smoke run** — subsampled rows and reduced iterations. Not a result.", ""])

    lines.extend(
        [
            "## 1. Baseline model gap under this feature set",
            "",
            "Answers: how much of the CatBoost-over-LightGBM lead survives once the ID-encoding confound is removed?",
            "",
            "| fold | " + " | ".join(setup["backends"]) + " | gap (cb − lgb) |",
            "|---|" + "---:|" * (len(setup["backends"]) + 1),
        ]
    )
    for row in payload["baseline_gap"]:
        cells = [f"{row.get(b, float('nan')):.3f}" for b in setup["backends"]]
        gap = row.get("gap_catboost_minus_lightgbm")
        gap_text = f"{gap:+.3f}" if gap is not None else "—"
        lines.append(f"| {row['fold_id']} | " + " | ".join(cells) + f" | {gap_text} |")

    lines.extend(
        [
            "",
            "## 2. Transfer verdict per variant",
            "",
            "Paired Δ = variant raw BSS − BASE raw BSS at the **same** backend, fold and seed.",
            "",
            "| variant | change | Δ mean (catboost) | Δ mean (lightgbm) | sign agree | fold sign agree | verdict |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for name, block in payload["variants"].items():
        verdict = block["transfer"]
        means = verdict["delta_mean_by_backend"]
        cb = means.get("catboost", float("nan"))
        lgb = means.get("lightgbm", float("nan"))
        lines.append(
            f"| {name} | {block['description']} | {cb:+.3f} | {lgb:+.3f} | "
            f"{'yes' if verdict['sign_agreement_on_overall_mean'] else 'no'} | "
            f"{verdict['fold_sign_agreement']} | **{verdict['verdict']}** |"
        )

    lines.extend(["", "## 3. Noise reference and per-fold breakdown", ""])
    for name, block in payload["variants"].items():
        lines.extend([f"### {name} — {block['description']}", ""])
        lines.append(
            "| backend | cells | Δ mean | Δ std (folds) | 2·sem | seed spread | materiality | "
            + " | ".join(f"Δ {f['fold_id']}" for f in setup["folds"])
            + " |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|" + "---:|" * len(setup["folds"]))
        for backend in setup["backends"]:
            stats = block["per_backend"].get(backend)
            if stats is None:
                continue
            two_sem = 2.0 * stats["delta_sem"] if np.isfinite(stats["delta_sem"]) else float("nan")
            fold_cells = [
                f"{stats['delta_mean_by_fold'].get(f['fold_id'], float('nan')):+.3f}"
                for f in setup["folds"]
            ]
            seed_spread = stats["seed_spread_within_fold"]
            seed_text = (
                f"{seed_spread:.3f}" if np.isfinite(seed_spread) else "—"
            ) + ("" if stats["seed_axis_informative"] else " (deterministic)")
            lines.append(
                f"| {backend} | {stats['n_paired_cells']} | {stats['delta_mean']:+.3f} | "
                f"{stats['delta_std_across_folds']:.3f} | {two_sem:.3f} | {seed_text} | "
                f"{stats['materiality']} | "
                + " | ".join(fold_cells)
                + " |"
            )
        lines.extend(["", f"- reading: {block['transfer']['reading']}", ""])

    if payload.get("cross_backend"):
        lines.extend(
            [
                "## 4. Auxiliary — CatBoost vs LightGBM agreement on BASE",
                "",
                f"_{payload['cross_backend']['note']}_",
                "",
                "| fold | seed | Pearson corr | catboost raw BSS | lightgbm raw BSS | 50/50 blend raw BSS |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["cross_backend"]["folds"]:
            lines.append(
                f"| {row['fold_id']} | {row['seed']} | {row['pearson_corr']:.4f} | "
                f"{row['catboost_raw_bss']:.3f} | {row['lightgbm_raw_bss']:.3f} | "
                f"{row['blend_50_50_raw_bss']:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## How to read this",
            "",
            "- `TRANSFERS` — the FE moves both backends the same way beyond noise. The feature work is model-agnostic.",
            "- `MODEL_SPECIFIC` — both backends move materially but in opposite directions. That is the CatBoost-specialisation risk.",
            "- `PARTIAL_*` — material in one backend only. Weak; add seeds before acting.",
            "- `INCONCLUSIVE_NOISE` — the delta is inside its own paired-delta noise. This says the **measurement** was not sensitive enough; it is not evidence the feature is bad.",
            "- `materiality` is a descriptive heuristic (`|Δ mean| > 2·sem`, `sem = std(fold means)/√n_folds`), not a hypothesis test. Fold means are the unit of observation because seeds are nested inside folds.",
            "- `seed spread (deterministic)` means that backend produced identical predictions across seeds under the configured params. LightGBM is deterministic here because `configs/baseline.yaml` sets no `subsample` / `colsample_bytree`; add them if you want a seed-noise estimate for it.",
            "- Validation, metric and season weights are unchanged (`reports/validation_policy.md`, `src/metrics.py`).",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------- cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{AUDIT_ID}: FE transferability across models.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=f"comma list; BASE is always included. Available: {','.join(VARIANTS)}",
    )
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--folds", default="", help="comma list of fold_id; default all three")
    parser.add_argument("--feature-set", default="L42", choices=["L42", "full44"])
    parser.add_argument("--out-dir", default="outputs/analysis/model_transfer_audit_001")
    parser.add_argument("--eda-dir", default="outputs/eda")
    parser.add_argument("--no-resume", action="store_true", help="ignore the run cache")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-rows-per-season", type=int, default=800)
    parser.add_argument("--smoke-iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wall0 = time.perf_counter()

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variant_names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Available: {list(VARIANTS)}")
    if BASE_VARIANT not in variant_names:
        variant_names.insert(0, BASE_VARIANT)
    ordered_variants = [VARIANTS[n] for n in dict.fromkeys(variant_names)]

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    bad_backends = [b for b in backends if b not in BACKEND_PARAMS_KEY]
    if bad_backends:
        raise SystemExit(f"Unknown backends: {bad_backends}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    config = load_config(args.config)
    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    target_col = data_cfg["target_col"]
    season_col = data_cfg["season_col"]
    id_col = data_cfg["id_col"]
    scale = float(resolve_metric_settings(config.get("task", {}))["scale"])

    fold_specs = build_fold_specs(config.get("cv", {}))
    if args.folds:
        wanted = {f.strip() for f in args.folds.split(",") if f.strip()}
        fold_specs = [f for f in fold_specs if f.fold_id in wanted]
        if not fold_specs:
            raise SystemExit(f"No folds matched {sorted(wanted)}")

    params_by_backend: dict[str, dict[str, Any]] = {}
    for backend in backends:
        params = dict(config.get("models", {}).get("params", {}).get(BACKEND_PARAMS_KEY[backend], {}))
        if args.smoke:
            if backend == "catboost":
                params["iterations"] = int(args.smoke_iterations)
            else:
                params["n_estimators"] = int(args.smoke_iterations)
        params_by_backend[backend] = params

    n_fits = len(ordered_variants) * len(backends) * len(fold_specs) * len(seeds)
    print(f"=== {AUDIT_ID} ===", flush=True)
    print(
        f"variants={[v.name for v in ordered_variants]} backends={backends} "
        f"seeds={seeds} folds={[f.fold_id for f in fold_specs]} feature_set={args.feature_set}",
        flush=True,
    )
    print(
        f"planned fits: {n_fits} "
        f"(CatBoost fits observed at ~{OBSERVED_CATBOOST_SECONDS_PER_FIT[0]:.0f}-"
        f"{OBSERVED_CATBOOST_SECONDS_PER_FIT[1]:.0f}s each on the full 2024 holdout; "
        "results are cached per cell so the run is resumable)",
        flush=True,
    )
    if args.dry_run:
        return

    needs_tm = any(v.needs_trackman for v in ordered_variants)
    tm: pd.DataFrame | None = None
    if needs_tm:
        if not TRACKMAN_PATH.exists():
            raise SystemExit(f"Trackman history not found: {TRACKMAN_PATH}")
        tm = pd.read_csv(
            TRACKMAN_PATH,
            encoding="utf-8-sig",
            usecols=["season", "pitch_type_group", "rel_speed"],
            low_memory=False,
        )

    loaded = load_competition_data(config)
    base_df = loaded["train"].reset_index(drop=True)
    _ = loaded["test"]  # explicitly unused
    if args.smoke:
        base_df = subsample_by_season(base_df, season_col, args.smoke_rows_per_season)
    print(f"train rows: {len(base_df)}", flush=True)

    used, numeric_cols, categorical_cols = base_feature_lists(
        base_df,
        target_col=target_col,
        exclude_cols=list(feat_cfg.get("exclude_cols") or []),
        categorical_cols=list(feat_cfg.get("categorical_cols") or []),
        feature_set=args.feature_set,
    )
    print(f"base features: {len(used)} ({len(categorical_cols)} categorical)", flush=True)

    out_dir = ensure_dir(resolve_path(args.out_dir + ("_smoke" if args.smoke else "")))
    eda_dir = ensure_dir(resolve_path(args.eda_dir))
    cache_path = out_dir / "runs.jsonl"
    cache = {} if args.no_resume else load_cache(cache_path)
    if cache:
        print(f"resume: {len(cache)} cached cells in {cache_path}", flush=True)

    rows: list[dict[str, Any]] = []
    preds: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    n_executed = 0
    n_reused = 0
    # Predictions are held only for the one BASE cell per backend that feeds section 4,
    # so a resumed run refits at most `len(backends)` cells instead of all of BASE.
    pred_fold_id = next((f.fold_id for f in fold_specs if f.role == "primary"), fold_specs[0].fold_id)
    pred_seed = seeds[0]

    for variant in ordered_variants:
        frame = materialize_variant(base_df, variant)
        v_used, v_nums, v_cats = variant_feature_lists(
            variant,
            used=used,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            train_columns=list(base_df.columns),
        )
        fold_extra_frame = make_fold_extra_frame(variant, tm)
        for backend in backends:
            params = params_by_backend[backend]
            for spec in fold_specs:
                for seed in seeds:
                    key = cell_key(
                        feature_set=args.feature_set,
                        variant=variant.name,
                        backend=backend,
                        fold_id=spec.fold_id,
                        seed=seed,
                        params=params,
                    )
                    keep_pred = (
                        variant.name == BASE_VARIANT
                        and spec.fold_id == pred_fold_id
                        and seed == pred_seed
                    )
                    cached = cache.get(key)
                    if cached is not None and not keep_pred:
                        rows.append(cached["metrics"] | {"variant": variant.name})
                        n_reused += 1
                        continue
                    row = run_cell(
                        train_df=frame,
                        spec=spec,
                        used=v_used,
                        numeric_cols=v_nums,
                        categorical_cols=v_cats,
                        target_col=target_col,
                        season_col=season_col,
                        id_col=id_col,
                        backend=backend,
                        params=params,
                        seed=seed,
                        scale=scale,
                        weight_map=DEFAULT_WEIGHTS,
                        extra_numeric=variant.extra_numeric,
                        fold_extra_frame=fold_extra_frame,
                        keep_pred=keep_pred,
                    )
                    n_executed += 1
                    if keep_pred:
                        preds[(variant.name, backend, spec.fold_id, seed)] = {
                            "pred": row.pop("_pred"),
                            "y_true": row.pop("_y_true"),
                            "row_id": row.pop("_row_id"),
                        }
                    metrics_row = {k: v for k, v in row.items() if not k.startswith("_")}
                    rows.append(metrics_row | {"variant": variant.name})
                    append_cache(
                        cache_path,
                        {
                            "cell_key": key,
                            "audit_id": AUDIT_ID,
                            "generated_at": datetime.now().isoformat(timespec="seconds"),
                            "feature_set": args.feature_set,
                            "variant": variant.name,
                            "params": params,
                            "metrics": metrics_row,
                        },
                    )
                    print(
                        f"  [{variant.name}/{backend}/{spec.fold_id}/seed={seed}] "
                        f"raw_bss={row['raw_brier_skill_score']:.3f} "
                        f"elapsed_s={row['elapsed_seconds']:.1f}",
                        flush=True,
                    )

    variant_blocks: dict[str, Any] = {}
    for variant in ordered_variants:
        if variant.name == BASE_VARIANT:
            continue
        per_backend: dict[str, dict[str, Any]] = {}
        pairs: dict[str, list[dict[str, Any]]] = {}
        for backend in backends:
            deltas = paired_deltas(rows, variant=variant.name, backend=backend)
            if not deltas:
                continue
            pairs[backend] = deltas
            per_backend[backend] = summarize_deltas(deltas)
        if not per_backend:
            continue
        variant_blocks[variant.name] = {
            "description": variant.description,
            "source_experiment": variant.source_experiment,
            "per_backend": per_backend,
            "paired_runs": pairs,
            "transfer": transfer_verdict(per_backend),
        }

    payload = {
        "audit_id": AUDIT_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "setup": {
            "config": str(args.config),
            "feature_set": args.feature_set,
            "n_features_base": len(used),
            "base_features": used,
            "categorical_cols": categorical_cols,
            "backends": backends,
            "seeds": seeds,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "role": f.role,
                    "train_seasons": f.train_seasons,
                    "valid_seasons": f.valid_seasons,
                }
                for f in fold_specs
            ],
            "params": params_by_backend,
            "sample_weight_by_season": {str(k): v for k, v in sorted(DEFAULT_WEIGHTS.items())},
            "variants": {v.name: v.description for v in ordered_variants},
            "smoke": bool(args.smoke),
        },
        "n_fits_executed": n_executed,
        "n_fits_reused": n_reused,
        "runs": rows,
        "baseline_gap": backend_gap_table(rows),
        "variants": variant_blocks,
        "cross_backend": cross_backend_agreement(preds, scale) if preds else None,
        "elapsed_seconds_wall": float(time.perf_counter() - wall0),
        "decision": "results_only_no_adopt_reject",
    }

    stem = "model_transfer_audit_001" + ("_smoke" if args.smoke else "")
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    runs_csv = out_dir / f"{stem}_runs.csv"
    pd.DataFrame(rows).to_csv(runs_csv, index=False)
    md_path = eda_dir / f"{stem}.md"
    md_path.write_text(build_markdown(payload), encoding="utf-8")

    for path in (json_path, runs_csv, md_path):
        print(f"wrote {path}", flush=True)
    print(
        f"done: {n_executed} fits executed, {n_reused} reused, "
        f"wall={payload['elapsed_seconds_wall']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
