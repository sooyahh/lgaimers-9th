from __future__ import annotations

"""Shared helpers for EXP-002 LightGBM / EXP-003 CatBoost baselines."""

import inspect
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.utils import (
    append_experiment_log,
    ensure_dir,
    ensure_output_dirs,
    load_config,
    resolve_path,
    save_json,
    set_seed,
    to_json_text,
)

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None


def rss_mb() -> float | None:
    if psutil is None:
        return None
    return float(psutil.Process().memory_info().rss) / (1024 * 1024)


def build_feature_lists(
    columns: list[str],
    *,
    target_col: str,
    exclude_cols: list[str],
    categorical_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    excluded = set(exclude_cols) | {target_col}
    used = [c for c in columns if c not in excluded]
    cats = [c for c in categorical_cols if c in used]
    nums = [c for c in used if c not in cats]
    missing_cats = [c for c in categorical_cols if c not in columns and c not in excluded]
    if missing_cats:
        raise ValueError(f"Configured categorical columns missing from data: {missing_cats}")
    return used, nums, cats


def pred_stats(pred: np.ndarray) -> dict[str, float]:
    return {
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
    }


def prepare_native_frames(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    backend: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-local imputation + native categorical preparation (no OneHotEncoder)."""
    X_tr = X_tr.copy()
    X_va = X_va.copy()

    for col in numeric_cols:
        median = float(X_tr[col].median()) if X_tr[col].notna().any() else 0.0
        X_tr[col] = X_tr[col].fillna(median)
        X_va[col] = X_va[col].fillna(median)

    for col in categorical_cols:
        # String categories keep CatBoost/LightGBM native handling simple.
        tr = X_tr[col].astype("string").fillna("__NA__").astype(str)
        va = X_va[col].astype("string").fillna("__NA__").astype(str)
        if backend == "lightgbm":
            categories = pd.Index(pd.unique(tr))
            X_tr[col] = pd.Categorical(tr, categories=categories)
            # Unseen valid levels become NaN (LightGBM handles missing cats).
            X_va[col] = pd.Categorical(va, categories=categories)
        elif backend == "catboost":
            X_tr[col] = tr
            X_va[col] = va
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    return X_tr, X_va


def positive_class_proba(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, int]:
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        pos_index = classes.index(1)
    elif 1.0 in classes:
        pos_index = classes.index(1.0)
    else:
        raise ValueError(f"Positive class 1 not in model.classes_: {classes}")
    pred = np.asarray(model.predict_proba(X), dtype=float)[:, pos_index]
    return pred, int(pos_index)


def load_prior_primary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary = payload.get("primary_2024") or {}
    return {
        "official_brier_skill_score": primary.get(
            "official_brier_skill_score", payload.get("official_brier_skill_score")
        ),
        "raw_brier_skill_score": primary.get(
            "raw_brier_skill_score", payload.get("raw_brier_skill_score")
        ),
        "brier_score": primary.get("brier_score", payload.get("brier_score")),
    }


FitFn = Callable[..., Any]


def run_tree_experiment(
    *,
    experiment_id: str,
    model_name: str,
    backend: str,
    config_path: str | Path,
    params_key: str,
    fit_model: FitFn,
    model_filename: str,
    sample_weight_by_season: dict[int, float] | None = None,
    extra_exclude_cols: list[str] | None = None,
    extra_numeric_cols: list[str] | None = None,
    fold_extra_features: Callable[[pd.DataFrame, dict[str, Any]], tuple[pd.DataFrame, dict[str, Any]]]
    | None = None,
    log_overrides: dict[str, Any] | None = None,
    skip_experiment_log: bool = False,
) -> dict[str, Any]:
    wall0 = time.perf_counter()
    config = load_config(config_path)
    ensure_output_dirs(config)
    seed = int(config.get("runtime", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    cv_cfg = config.get("cv", {})
    model_params = dict(config.get("models", {}).get("params", {}).get(params_key, {}))
    metric_settings = resolve_metric_settings(config.get("task", {}))
    scale = float(metric_settings["scale"])

    target_col = data_cfg["target_col"]
    id_col = data_cfg["id_col"]
    season_col = data_cfg["season_col"]
    exclude_cols = list(feat_cfg.get("exclude_cols") or [])
    extra_exclude = list(extra_exclude_cols or [])
    extra_numeric = list(extra_numeric_cols or [])
    if extra_exclude or extra_numeric:
        if experiment_id in {"EXP-004", "EXP-003", "EXP-002", "EXP-001A"}:
            raise ValueError(
                f"Refusing extra feature hooks on protected experiment_id={experiment_id}."
            )
        for col in extra_exclude:
            if col not in exclude_cols:
                exclude_cols.append(col)
    if extra_numeric and fold_extra_features is None:
        raise ValueError("extra_numeric_cols requires fold_extra_features")
    if fold_extra_features is not None and not extra_numeric:
        raise ValueError("fold_extra_features requires extra_numeric_cols")
    categorical_cols_cfg = list(feat_cfg.get("categorical_cols") or [])

    loaded = load_competition_data(config)
    train_df = loaded["train"]
    # Explicitly ignore test for training/validation.
    _ = loaded["test"]

    used, numeric_cols, categorical_cols = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols_cfg,
    )
    if extra_exclude:
        missing_extra = [c for c in extra_exclude if c not in train_df.columns]
        if missing_extra:
            raise ValueError(f"extra_exclude_cols not in train: {missing_extra}")
        still_present = [c for c in extra_exclude if c in used]
        if still_present:
            raise AssertionError(f"extra_exclude_cols still in used features: {still_present}")
    if extra_numeric:
        overlap = [c for c in extra_numeric if c in used]
        if overlap:
            raise AssertionError(f"extra_numeric_cols already in used features: {overlap}")
        in_train = [c for c in extra_numeric if c in train_df.columns]
        if in_train:
            raise AssertionError(
                f"extra_numeric_cols already present on train_df (must be fold-injected): {in_train}"
            )
        used = list(used) + extra_numeric
        numeric_cols = list(numeric_cols) + extra_numeric
    asof_cols = [c for c in used if c.startswith("asof_")]

    primary = cv_cfg["primary"]
    secondary = list(cv_cfg.get("secondary_folds") or [])
    fold_specs = [
        {
            "fold_id": "secondary_2022",
            "train_seasons": list(secondary[0]["train_seasons"]),
            "valid_seasons": list(secondary[0]["valid_seasons"]),
            "role": "secondary",
        },
        {
            "fold_id": "secondary_2023",
            "train_seasons": list(secondary[1]["train_seasons"]),
            "valid_seasons": list(secondary[1]["valid_seasons"]),
            "role": "secondary",
        },
        {
            "fold_id": "primary_2024",
            "train_seasons": list(primary["train_seasons"]),
            "valid_seasons": list(primary["valid_seasons"]),
            "role": "primary",
        },
    ]

    out_dir = ensure_dir(resolve_path(config["output"]["predictions_dir"]) / experiment_id)
    model_dir = ensure_dir(resolve_path(config["output"]["models_dir"]) / experiment_id)

    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []
    primary_model: Any = None

    for spec in fold_specs:
        fold = build_season_holdout(
            train_df,
            season_col,
            spec["train_seasons"],
            spec["valid_seasons"],
            name=spec["fold_id"],
        )
        started = time.perf_counter()
        rss0 = rss_mb()

        X_tr_raw = train_df.iloc[fold.train_idx][[c for c in used if c not in extra_numeric]]
        y_tr = train_df.iloc[fold.train_idx][target_col]
        X_va_raw = train_df.iloc[fold.valid_idx][[c for c in used if c not in extra_numeric]]
        y_va = train_df.iloc[fold.valid_idx][target_col]
        fold_extra_info: dict[str, Any] = {}
        if fold_extra_features is not None:
            extra_df, fold_extra_info = fold_extra_features(train_df, spec)
            if extra_df is train_df:
                raise AssertionError("fold_extra_features must not return the shared train_df")
            leaked = [c for c in extra_numeric if c in train_df.columns]
            if leaked:
                raise AssertionError(f"fold_extra_features mutated train_df with {leaked}")
            missing_extra = [c for c in extra_numeric if c not in extra_df.columns]
            if missing_extra:
                raise AssertionError(f"fold_extra_features missing columns: {missing_extra}")
            if not extra_df.index.equals(train_df.index):
                raise AssertionError("fold_extra_features index must match train_df")
            X_tr_raw = pd.concat(
                [X_tr_raw, extra_df.iloc[fold.train_idx][extra_numeric]],
                axis=1,
            )
            X_va_raw = pd.concat(
                [X_va_raw, extra_df.iloc[fold.valid_idx][extra_numeric]],
                axis=1,
            )
        missing_used = [c for c in used if c not in X_tr_raw.columns]
        if missing_used:
            raise AssertionError(f"Training frame missing features: {missing_used}")
        X_tr_raw = X_tr_raw[used]
        X_va_raw = X_va_raw[used]

        X_tr, X_va = prepare_native_frames(
            X_tr_raw,
            X_va_raw,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            backend=backend,
        )
        sample_weight = None
        if sample_weight_by_season is not None:
            seasons = train_df.iloc[fold.train_idx][season_col].to_numpy()
            missing = sorted({int(s) for s in seasons if int(s) not in sample_weight_by_season})
            if missing:
                raise ValueError(
                    f"sample_weight_by_season missing seasons {missing} in fold {spec['fold_id']}"
                )
            sample_weight = np.asarray(
                [float(sample_weight_by_season[int(s)]) for s in seasons], dtype=float
            )
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            if "sample_weight" not in inspect.signature(fit_model).parameters:
                raise TypeError(
                    f"fit_model for {experiment_id} does not accept sample_weight, "
                    "but sample_weight_by_season was provided."
                )
            fit_kwargs["sample_weight"] = sample_weight
        model = fit_model(X_tr, y_tr, categorical_cols, model_params, seed, **fit_kwargs)
        pred, pos_index = positive_class_proba(model, X_va)
        metrics = evaluate_predictions(y_va, pred, scale=scale)

        elapsed = float(time.perf_counter() - started)
        rss1 = rss_mb()
        peak = None if rss0 is None or rss1 is None else float(max(rss0, rss1))

        pred_frame = train_df.iloc[fold.valid_idx][[id_col, season_col, target_col]].copy()
        pred_frame["prediction"] = pred
        pred_frame["fold_id"] = spec["fold_id"]
        pred_frame["split_type"] = "expanding_or_primary"
        pred_parts.append(pred_frame)

        row = {
            "fold_id": spec["fold_id"],
            "role": spec["role"],
            "train_seasons": spec["train_seasons"],
            "valid_seasons": spec["valid_seasons"],
            "n_train": int(len(fold.train_idx)),
            "n_valid": int(len(fold.valid_idx)),
            "train_target_mean": float(y_tr.mean()),
            "valid_target_rate": float(metrics["target_rate"]),
            "brier_score": float(metrics["brier_score"]),
            "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
            "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
            **pred_stats(pred),
            "elapsed_seconds": elapsed,
            "rss_mb_before": rss0,
            "rss_mb_after": rss1,
            "rss_mb_peak_observed": peak,
            "positive_class_index": pos_index,
            "categorical_encoding": f"native_{backend}",
            "sample_weight_by_season": (
                {str(k): float(v) for k, v in sorted(sample_weight_by_season.items())}
                if sample_weight_by_season is not None
                else None
            ),
        }
        if fold_extra_info:
            row["fold_extra"] = fold_extra_info
        fold_rows.append(row)
        if spec["role"] == "primary":
            primary_model = model
            joblib.dump(model, model_dir / model_filename)
        print(
            f"{spec['fold_id']}: official_bss={row['official_brier_skill_score']:.3f} "
            f"raw_bss={row['raw_brier_skill_score']:.3f} brier={row['brier_score']:.6f} "
            f"elapsed={row['elapsed_seconds']:.1f}s"
        )

    preds = pd.concat(pred_parts, ignore_index=True)
    preds_path = out_dir / "oof_holdout_predictions.csv"
    preds.to_csv(preds_path, index=False)

    oof_metrics = evaluate_predictions(
        preds[target_col].to_numpy(),
        preds["prediction"].to_numpy(),
        scale=scale,
    )
    fold_raw = np.array([r["raw_brier_skill_score"] for r in fold_rows], dtype=float)
    primary_row = next(r for r in fold_rows if r["role"] == "primary")

    fold_csv = pd.DataFrame(fold_rows).copy()
    fold_csv["train_seasons"] = fold_csv["train_seasons"].map(json.dumps)
    fold_csv["valid_seasons"] = fold_csv["valid_seasons"].map(json.dumps)
    if "sample_weight_by_season" in fold_csv.columns:
        fold_csv["sample_weight_by_season"] = fold_csv["sample_weight_by_season"].map(
            lambda v: json.dumps(v, ensure_ascii=False) if v is not None else ""
        )
    if "fold_extra" in fold_csv.columns:
        fold_csv["fold_extra"] = fold_csv["fold_extra"].map(
            lambda v: json.dumps(v, ensure_ascii=False) if v else ""
        )
    fold_metrics_path = out_dir / "fold_metrics.csv"
    fold_csv.to_csv(fold_metrics_path, index=False)

    exp000 = load_prior_primary(resolve_path("outputs/predictions/EXP-000/metrics.json"))
    exp001a = load_prior_primary(resolve_path("outputs/predictions/EXP-001A/summary.json"))

    def _cmp(label: str, prior: dict[str, Any] | None) -> dict[str, Any] | None:
        if prior is None:
            return None
        return {
            "baseline": label,
            "baseline_official_bss": prior.get("official_brier_skill_score"),
            "baseline_raw_bss": prior.get("raw_brier_skill_score"),
            "baseline_brier": prior.get("brier_score"),
            "this_official_bss": primary_row["official_brier_skill_score"],
            "this_raw_bss": primary_row["raw_brier_skill_score"],
            "this_brier": primary_row["brier_score"],
            "delta_official_bss": float(
                primary_row["official_brier_skill_score"]
                - float(prior.get("official_brier_skill_score") or 0.0)
            ),
            "delta_raw_bss": float(
                primary_row["raw_brier_skill_score"]
                - float(prior.get("raw_brier_skill_score") or 0.0)
            ),
        }

    comparison = {
        "exp000": _cmp("EXP-000", exp000),
        "exp001a": _cmp("EXP-001A", exp001a),
    }

    summary = {
        "experiment_id": experiment_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "seed": seed,
        "features_used": used,
        "n_features_used": len(used),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "excluded_cols": exclude_cols,
        "asof_cols_included": asof_cols,
        "categorical_encoding": f"native_{backend}",
        "params": model_params,
        "sample_weight_by_season": (
            {str(k): float(v) for k, v in sorted(sample_weight_by_season.items())}
            if sample_weight_by_season is not None
            else None
        ),
        "fold_metrics": fold_rows,
        "fold_raw_bss_mean": float(np.mean(fold_raw)),
        "fold_raw_bss_std": float(np.std(fold_raw, ddof=0)),
        "oof": {
            "n_samples": int(oof_metrics["n_samples"]),
            "target_rate": float(oof_metrics["target_rate"]),
            "brier_score": float(oof_metrics["brier_score"]),
            "raw_brier_skill_score": float(oof_metrics["raw_brier_skill_score"]),
            "official_brier_skill_score": float(oof_metrics["official_brier_skill_score"]),
            **pred_stats(preds["prediction"].to_numpy()),
        },
        "primary_2024": {
            "brier_score": primary_row["brier_score"],
            "raw_brier_skill_score": primary_row["raw_brier_skill_score"],
            "official_brier_skill_score": primary_row["official_brier_skill_score"],
            "valid_target_rate": primary_row["valid_target_rate"],
            **{
                k: primary_row[k]
                for k in ("prediction_mean", "prediction_std", "prediction_min", "prediction_max")
            },
        },
        "comparison_primary_2024": comparison,
        "prediction_path": str(preds_path),
        "fold_metrics_path": str(fold_metrics_path),
        "model_path": str(model_dir / model_filename) if primary_model is not None else None,
        "elapsed_seconds_wall": float(time.perf_counter() - wall0),
        "decision": "results_only_no_adopt_reject",
    }

    summary_path = out_dir / "summary.json"
    save_json(summary, summary_path)

    lines = [
        f"# {experiment_id} {model_name} Baseline",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- features used: `{len(used)}` (same exclude/categorical lists as EXP-001A)",
        f"- categorical encoding: `native_{backend}` (no OneHotEncoder)",
        f"- params: `{json.dumps(model_params, ensure_ascii=False)}`",
        f"- seed: `{seed}`",
        "",
        "## Fold metrics",
        "",
        "| fold | role | valid rate | Brier | raw BSS | official BSS | pred mean±std | elapsed_s |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in fold_rows:
        lines.append(
            f"| {r['fold_id']} | {r['role']} | {r['valid_target_rate']:.4f} | {r['brier_score']:.6f} | "
            f"{r['raw_brier_skill_score']:.3f} | {r['official_brier_skill_score']:.3f} | "
            f"{r['prediction_mean']:.4f}±{r['prediction_std']:.4f} | {r['elapsed_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- fold raw BSS mean±std: `{summary['fold_raw_bss_mean']:.3f} ± {summary['fold_raw_bss_std']:.3f}`",
            f"- OOF Brier: `{summary['oof']['brier_score']:.6f}`",
            f"- OOF raw BSS: `{summary['oof']['raw_brier_skill_score']:.3f}`",
            f"- OOF official BSS: `{summary['oof']['official_brier_skill_score']:.3f}`",
            f"- primary 2024 raw BSS: `{summary['primary_2024']['raw_brier_skill_score']:.3f}`",
            f"- primary 2024 official BSS: `{summary['primary_2024']['official_brier_skill_score']:.3f}`",
            f"- wall time (s): `{summary['elapsed_seconds_wall']:.1f}`",
            "",
            "## Comparison on primary 2024 (raw BSS is the discriminative metric)",
            "",
            "| baseline | Brier | raw BSS | official BSS | Δ raw BSS | Δ official BSS |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {experiment_id} | {primary_row['brier_score']:.6f} | "
            f"{primary_row['raw_brier_skill_score']:.3f} | "
            f"{primary_row['official_brier_skill_score']:.3f} | 0.000 | 0.000 |",
        ]
    )
    for key in ("exp000", "exp001a"):
        cmp_ = comparison.get(key)
        if not cmp_:
            lines.append(f"| {key} | (missing) | | | | |")
            continue
        lines.append(
            f"| {cmp_['baseline']} | {cmp_['baseline_brier']} | {cmp_['baseline_raw_bss']} | "
            f"{cmp_['baseline_official_bss']} | {cmp_['delta_raw_bss']:.3f} | "
            f"{cmp_['delta_official_bss']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Official BSS is clipped with `max(0, ·)`; near-zero values have low ranking power.",
            "- Compare primarily on **raw BSS** and Brier.",
            "- `decision: results_only_no_adopt_reject` (no adopt/reject).",
            "",
            f"- summary: `{summary_path}`",
            f"- predictions: `{preds_path}`",
            "",
        ]
    )
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    log_overrides = dict(log_overrides or {})
    purpose = log_overrides.get(
        "purpose",
        (
            f"{model_name}_season_sample_weight_native_cats"
            if sample_weight_by_season is not None
            else f"{model_name}_provisionally_safe_features_native_cats"
        ),
    )
    notes_payload: dict[str, Any] = {
        "primary_raw_bss": primary_row["raw_brier_skill_score"],
        "primary_official_bss": primary_row["official_brier_skill_score"],
        "oof_raw_bss": summary["oof"]["raw_brier_skill_score"],
        "oof_official_bss": summary["oof"]["official_brier_skill_score"],
        "comparison_primary_2024": comparison,
        "sample_weight_by_season": (
            {str(k): float(v) for k, v in sorted(sample_weight_by_season.items())}
            if sample_weight_by_season is not None
            else None
        ),
        "promoted_from": (
            "regime_shift_experiment_A" if sample_weight_by_season is not None else None
        ),
        "decision": "results_only_no_adopt_reject",
        "extra_exclude_cols": extra_exclude or None,
        "extra_numeric_cols": extra_numeric or None,
    }
    if log_overrides.get("notes"):
        notes_payload.update(log_overrides["notes"])
    next_action = log_overrides.get(
        "next_action",
        (
            "Compare EXP-004 fold raw BSS to EXP-003; no adopt/reject yet."
            if sample_weight_by_season is not None
            else "Compare raw BSS across EXP-000/001A/002/003; no adopt/reject yet."
        ),
    )

    if not skip_experiment_log:
        append_experiment_log(
            config["output"]["experiments_path"],
            {
                "experiment_id": experiment_id,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "purpose": purpose,
                "data_version": config.get("project", {}).get("data_version", ""),
                "features": to_json_text(
                    {
                        "n": len(used),
                        "exclude": exclude_cols,
                        "categorical": categorical_cols,
                        "encoding": f"native_{backend}",
                        "extra_exclude_cols": extra_exclude or None,
                        "extra_numeric_cols": extra_numeric or None,
                    }
                ),
                "model": model_name,
                "params": to_json_text(
                    {
                        **model_params,
                        **(
                            {
                                "sample_weight_by_season": {
                                    str(k): float(v)
                                    for k, v in sorted(sample_weight_by_season.items())
                                }
                            }
                            if sample_weight_by_season is not None
                            else {}
                        ),
                    }
                ),
                "cv_strategy": to_json_text({"primary": primary, "secondary_folds": secondary}),
                "metric": "brier_skill_score",
                "metric_direction": "maximize",
                "cv_score": summary["fold_raw_bss_mean"],
                "cv_std": summary["fold_raw_bss_std"],
                "holdout_score": primary_row["official_brier_skill_score"],
                "public_lb_score": "",
                "cv_lb_gap": "",
                "seed": seed,
                "notes": to_json_text(notes_payload),
                "next_action": next_action,
                "status": "success",
                "error": "",
            },
        )

    summary["report_path"] = str(report_path)
    summary["summary_path"] = str(summary_path)
    save_json(summary, summary_path)
    return summary
