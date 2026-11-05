from __future__ import annotations

"""EXP-001A: Logistic Regression on provisionally safe features (sparse OHE)."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

EXPERIMENT_ID = "EXP-001A"

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None


def _rss_mb() -> float | None:
    if psutil is None:
        return None
    return float(psutil.Process().memory_info().rss) / (1024 * 1024)


def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


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


def make_pipeline(numeric_cols: list[str], categorical_cols: list[str], params: dict[str, Any], seed: int) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric_cols,
            )
        )
    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _one_hot()),
            ]
        )
        transformers.append(("categorical", cat_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No feature columns configured for EXP-001A.")

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,  # keep sparse; dense OHE would be ~20GB
    )
    clf = LogisticRegression(
        C=float(params.get("C", 1.0)),
        solver=str(params.get("solver", "liblinear")),
        max_iter=int(params.get("max_iter", 1000)),
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("model", clf)])


def _pred_stats(pred: np.ndarray) -> dict[str, float]:
    return {
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
    }


def run_fold(
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    id_col: str,
    season_col: str,
    train_seasons: list[int],
    valid_seasons: list[int],
    fold_id: str,
    params: dict[str, Any],
    seed: int,
    scale: float,
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline]:
    fold = build_season_holdout(
        train_df, season_col, train_seasons, valid_seasons, name=fold_id
    )
    started = time.perf_counter()
    rss0 = _rss_mb()

    X_tr = train_df.iloc[fold.train_idx][feature_cols]
    y_tr = train_df.iloc[fold.train_idx][target_col]
    X_va = train_df.iloc[fold.valid_idx][feature_cols]
    y_va = train_df.iloc[fold.valid_idx][target_col]

    pipe = make_pipeline(numeric_cols, categorical_cols, params, seed)
    pipe.fit(X_tr, y_tr)
    # positive class column for control_success=1
    classes = list(pipe.named_steps["model"].classes_)
    if 1 not in classes and 1.0 not in classes:
        raise ValueError(f"Positive class 1 not in model.classes_: {classes}")
    pos_index = classes.index(1) if 1 in classes else classes.index(1.0)
    pred = pipe.predict_proba(X_va)[:, pos_index]

    metrics = evaluate_predictions(y_va, pred, scale=scale)
    elapsed = float(time.perf_counter() - started)
    rss1 = _rss_mb()
    peak = None if rss0 is None or rss1 is None else float(max(rss0, rss1))

    # Inspect matrix sparsity after transform (no copy of full dense)
    Xt = pipe.named_steps["pre"].transform(X_tr.head(1000))
    matrix_kind = type(Xt).__name__
    is_sparse = hasattr(Xt, "nnz")

    pred_frame = train_df.iloc[fold.valid_idx][[id_col, season_col, target_col]].copy()
    pred_frame["prediction"] = pred
    pred_frame["fold_id"] = fold_id
    pred_frame["split_type"] = "expanding_or_primary"

    row = {
        "fold_id": fold_id,
        "train_seasons": train_seasons,
        "valid_seasons": valid_seasons,
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
        "train_target_mean": float(y_tr.mean()),
        "valid_target_rate": float(metrics["target_rate"]),
        "brier_score": float(metrics["brier_score"]),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        **_pred_stats(pred),
        "elapsed_seconds": elapsed,
        "rss_mb_before": rss0,
        "rss_mb_after": rss1,
        "rss_mb_peak_observed": peak,
        "matrix_kind_sample": matrix_kind,
        "transform_is_sparse_sample": bool(is_sparse),
        "positive_class_index": int(pos_index),
        "n_iter_": int(getattr(pipe.named_steps["model"], "n_iter_", [0])[0])
        if np.ndim(getattr(pipe.named_steps["model"], "n_iter_", 0))
        else int(getattr(pipe.named_steps["model"], "n_iter_", 0)),
    }
    return row, pred_frame, pipe


def run_exp001a(config_path: str | Path) -> dict[str, Any]:
    wall0 = time.perf_counter()
    config = load_config(config_path)
    ensure_output_dirs(config)
    seed = int(config.get("runtime", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    cv_cfg = config.get("cv", {})
    model_params = config.get("models", {}).get("params", {}).get("logistic_regression", {})
    metric_settings = resolve_metric_settings(config.get("task", {}))
    scale = float(metric_settings["scale"])

    target_col = data_cfg["target_col"]
    id_col = data_cfg["id_col"]
    season_col = data_cfg["season_col"]
    exclude_cols = list(feat_cfg.get("exclude_cols") or [])
    categorical_cols_cfg = list(feat_cfg.get("categorical_cols") or [])

    loaded = load_competition_data(config)
    train_df = loaded["train"]

    used, numeric_cols, categorical_cols = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols_cfg,
    )

    # asof re-check note: all asof_* remain in numeric_cols (officially pre-pitch)
    asof_cols = [c for c in used if c.startswith("asof_")]
    unclear_asof: list[str] = []  # none after official re-check

    primary = cv_cfg["primary"]
    secondary = list(cv_cfg.get("secondary_folds") or [])
    # Ensure primary is evaluated; secondary includes the same 2024 fold for OOF coverage
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

    out_dir = ensure_dir(resolve_path(config["output"]["predictions_dir"]) / EXPERIMENT_ID)
    model_dir = ensure_dir(resolve_path(config["output"]["models_dir"]) / EXPERIMENT_ID)

    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []
    primary_pipe: Pipeline | None = None

    for spec in fold_specs:
        row, pred_frame, pipe = run_fold(
            train_df=train_df,
            feature_cols=used,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_col=target_col,
            id_col=id_col,
            season_col=season_col,
            train_seasons=spec["train_seasons"],
            valid_seasons=spec["valid_seasons"],
            fold_id=spec["fold_id"],
            params=model_params,
            seed=seed,
            scale=scale,
        )
        row["role"] = spec["role"]
        fold_rows.append(row)
        pred_parts.append(pred_frame)
        if spec["role"] == "primary":
            primary_pipe = pipe
            joblib.dump(pipe, model_dir / "logistic_regression_primary.joblib")
        print(
            f"{spec['fold_id']}: official_bss={row['official_brier_skill_score']:.3f} "
            f"raw_bss={row['raw_brier_skill_score']:.3f} brier={row['brier_score']:.6f} "
            f"sparse={row['transform_is_sparse_sample']} elapsed={row['elapsed_seconds']:.1f}s"
        )

    preds = pd.concat(pred_parts, ignore_index=True)
    preds_path = out_dir / "oof_holdout_predictions.csv"
    preds.to_csv(preds_path, index=False)

    # OOF = concatenation of secondary folds' validation rows (2022+2023+2024)
    # primary_2024 is the same rows as secondary's last fold conceptually; we use all three secondary specs
    # Here fold_specs uses primary_2024 instead of duplicating secondary_2024 — OOF uses all pred_parts (2022,2023,2024 once).
    oof_metrics = evaluate_predictions(
        preds[target_col].to_numpy(),
        preds["prediction"].to_numpy(),
        scale=scale,
    )
    secondary_raw = np.array(
        [r["raw_brier_skill_score"] for r in fold_rows], dtype=float
    )
    primary_row = next(r for r in fold_rows if r["role"] == "primary")

    fold_csv = pd.DataFrame(fold_rows).copy()
    fold_csv["train_seasons"] = fold_csv["train_seasons"].map(json.dumps)
    fold_csv["valid_seasons"] = fold_csv["valid_seasons"].map(json.dumps)
    fold_metrics_path = out_dir / "fold_metrics.csv"
    fold_csv.to_csv(fold_metrics_path, index=False)

    # EXP-000 / VAL-AUDIT comparison (read-only)
    exp000_path = resolve_path("outputs/predictions/EXP-000/metrics.json")
    val_audit_path = resolve_path("outputs/predictions/VAL-AUDIT-001/summary.json")
    exp000 = json.loads(exp000_path.read_text(encoding="utf-8")) if exp000_path.exists() else None
    val_audit = json.loads(val_audit_path.read_text(encoding="utf-8")) if val_audit_path.exists() else None

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "logistic_regression",
        "seed": seed,
        "features_used": used,
        "n_features_used": len(used),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "excluded_cols": exclude_cols,
        "asof_cols_included": asof_cols,
        "asof_unclear_excluded": unclear_asof,
        "fold_metrics": fold_rows,
        "fold_raw_bss_mean": float(np.mean(secondary_raw)),
        "fold_raw_bss_std": float(np.std(secondary_raw, ddof=0)),
        "oof": {
            "n_samples": int(oof_metrics["n_samples"]),
            "target_rate": float(oof_metrics["target_rate"]),
            "brier_score": float(oof_metrics["brier_score"]),
            "raw_brier_skill_score": float(oof_metrics["raw_brier_skill_score"]),
            "official_brier_skill_score": float(oof_metrics["official_brier_skill_score"]),
            **_pred_stats(preds["prediction"].to_numpy()),
        },
        "primary_2024": {
            "brier_score": primary_row["brier_score"],
            "raw_brier_skill_score": primary_row["raw_brier_skill_score"],
            "official_brier_skill_score": primary_row["official_brier_skill_score"],
            "valid_target_rate": primary_row["valid_target_rate"],
            **{k: primary_row[k] for k in ("prediction_mean", "prediction_std", "prediction_min", "prediction_max")},
        },
        "comparison_to_exp000_primary": None
        if exp000 is None
        else {
            "exp000_official_bss": exp000.get("official_brier_skill_score"),
            "exp000_raw_bss": exp000.get("raw_brier_skill_score"),
            "exp000_brier": exp000.get("brier_score"),
            "exp001a_official_bss": primary_row["official_brier_skill_score"],
            "exp001a_raw_bss": primary_row["raw_brier_skill_score"],
            "exp001a_brier": primary_row["brier_score"],
            "delta_official_bss": float(
                primary_row["official_brier_skill_score"] - float(exp000.get("official_brier_skill_score", 0))
            ),
            "delta_raw_bss": float(
                primary_row["raw_brier_skill_score"] - float(exp000.get("raw_brier_skill_score", 0))
            ),
        },
        "val_audit_target_mean_raw_bss_by_fold": None
        if val_audit is None
        else {
            f["fold_id"]: f["raw_brier_skill_score"] for f in val_audit.get("folds", [])
        },
        "prediction_path": str(preds_path),
        "fold_metrics_path": str(fold_metrics_path),
        "model_path": str(model_dir / "logistic_regression_primary.joblib") if primary_pipe else None,
        "elapsed_seconds_wall": float(time.perf_counter() - wall0),
        "decision": "results_only_no_adopt_reject",
    }
    summary_path = out_dir / "summary.json"
    save_json(summary, summary_path)

    # Markdown report
    lines = [
        f"# {EXPERIMENT_ID} Logistic Regression Baseline",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- features used: `{len(used)}` (excluded WE/LI/row_id)",
        f"- asof included: `{len(asof_cols)}`; unclear excluded: `{unclear_asof}`",
        f"- categorical: `{categorical_cols}`",
        f"- seed: `{seed}`",
        "",
        "## Fold metrics",
        "",
        "| fold | role | valid rate | Brier | raw BSS | official BSS | pred mean±std | elapsed_s | sparse? |",
        "|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for r in fold_rows:
        lines.append(
            f"| {r['fold_id']} | {r['role']} | {r['valid_target_rate']:.4f} | {r['brier_score']:.6f} | "
            f"{r['raw_brier_skill_score']:.3f} | {r['official_brier_skill_score']:.3f} | "
            f"{r['prediction_mean']:.4f}±{r['prediction_std']:.4f} | {r['elapsed_seconds']:.1f} | "
            f"{r['transform_is_sparse_sample']} |"
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
            f"- primary 2024 official BSS: `{summary['primary_2024']['official_brier_skill_score']:.3f}`",
            f"- primary 2024 raw BSS: `{summary['primary_2024']['raw_brier_skill_score']:.3f}`",
            f"- wall time (s): `{summary['elapsed_seconds_wall']:.1f}`",
            "",
            "## Comparison to EXP-000 (primary 2024, target-mean)",
            "",
        ]
    )
    cmp_ = summary["comparison_to_exp000_primary"]
    if cmp_:
        lines.extend(
            [
                f"- EXP-000 official BSS: `{cmp_['exp000_official_bss']}`",
                f"- EXP-001A official BSS: `{cmp_['exp001a_official_bss']}`",
                f"- Δ official BSS: `{cmp_['delta_official_bss']:.3f}`",
                f"- EXP-000 raw BSS: `{cmp_['exp000_raw_bss']}`",
                f"- EXP-001A raw BSS: `{cmp_['exp001a_raw_bss']}`",
                f"- Δ raw BSS: `{cmp_['delta_raw_bss']:.3f}`",
                f"- EXP-000 Brier: `{cmp_['exp000_brier']}` → EXP-001A Brier: `{cmp_['exp001a_brier']}`",
                "",
                "채택/보류/폐기 판단은 수행하지 않음 (결과 보고만).",
            ]
        )
    else:
        lines.append("- EXP-000 metrics.json not found.")
    lines.extend(["", f"- summary: `{summary_path}`", f"- predictions: `{preds_path}`", ""])
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    append_experiment_log(
        config["output"]["experiments_path"],
        {
            "experiment_id": EXPERIMENT_ID,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "logistic_regression_provisionally_safe_features",
            "data_version": config.get("project", {}).get("data_version", ""),
            "features": to_json_text(
                {
                    "n": len(used),
                    "exclude": exclude_cols,
                    "categorical": categorical_cols,
                    "asof_included": asof_cols,
                }
            ),
            "model": "logistic_regression",
            "params": to_json_text(model_params),
            "cv_strategy": to_json_text(
                {"primary": primary, "secondary_folds": secondary}
            ),
            "metric": "brier_skill_score",
            "metric_direction": "maximize",
            "cv_score": summary["fold_raw_bss_mean"],
            "cv_std": summary["fold_raw_bss_std"],
            "holdout_score": primary_row["official_brier_skill_score"],
            "public_lb_score": "",
            "cv_lb_gap": "",
            "seed": seed,
            "notes": to_json_text(
                {
                    "primary_raw_bss": primary_row["raw_brier_skill_score"],
                    "oof_raw_bss": summary["oof"]["raw_brier_skill_score"],
                    "comparison_to_exp000": cmp_,
                    "decision": "results_only_no_adopt_reject",
                }
            ),
            "next_action": "Review EXP-001A vs EXP-000; no adopt/reject yet.",
            "status": "success",
            "error": "",
        },
    )
    summary["report_path"] = str(report_path)
    summary["summary_path"] = str(summary_path)
    save_json(summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    result = run_exp001a(args.config)
    print(json.dumps(
        {
            "experiment_id": result["experiment_id"],
            "primary_2024": result["primary_2024"],
            "oof": result["oof"],
            "fold_raw_bss_mean": result["fold_raw_bss_mean"],
            "fold_raw_bss_std": result["fold_raw_bss_std"],
            "comparison_to_exp000_primary": result["comparison_to_exp000_primary"],
            "elapsed_seconds_wall": result["elapsed_seconds_wall"],
            "summary_path": result.get("summary_path"),
            "report_path": result.get("report_path"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
