from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import load_competition_data, sort_temporal_frame
from src.diagnostics import save_diagnostics
from src.evaluate import (
    build_season_holdout,
    build_temporal_folds,
    metric_direction,
    resolve_metric_settings,
    score_metric,
    validate_metric,
)
from src.features import FeatureBuilder
from src.models import build_model, positive_probability
from src.utils import (
    append_experiment_log,
    assert_not_placeholder,
    ensure_dir,
    ensure_output_dirs,
    load_config,
    make_experiment_id,
    resolve_path,
    set_seed,
    to_json_text,
)


def _configured_models(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    models_cfg = config.get("models", {})
    enabled = models_cfg.get("enabled", [])
    if not enabled:
        raise ValueError("models.enabled must contain at least one baseline name.")
    params_by_model = models_cfg.get("params", {})
    return [(str(name), dict(params_by_model.get(name, {}))) for name in enabled]


def _make_pipeline(
    name: str,
    params: dict[str, Any],
    seed: int,
    data_cfg: dict[str, Any],
    features_cfg: dict[str, Any],
) -> Pipeline:
    if name.lower() == "global_mean":
        return Pipeline([("model", build_model(name, params, seed))])
    return Pipeline(
        [
            (
                "features",
                FeatureBuilder(
                    target_col=data_cfg["target_col"],
                    id_col=data_cfg["id_col"],
                    parse_datetime=bool(features_cfg.get("parse_datetime", True)),
                    add_missing_indicators=bool(features_cfg.get("add_missing_indicators", True)),
                    datetime_cols=[data_cfg["time_col"]] if data_cfg.get("time_col") else [],
                ),
            ),
            ("model", build_model(name, params, seed)),
        ]
    )


def _validate_binary_target(y: pd.Series) -> None:
    values = set(y.dropna().unique().tolist())
    if y.isna().any() or not values.issubset({0, 1}) or len(values) != 2:
        raise ValueError(f"Target must be non-null binary 0/1; found values={sorted(values)}")


def _validate_chronology(frame: pd.DataFrame, folds: list[Any], time_col: str | None) -> None:
    if not time_col:
        return
    parsed = pd.to_datetime(frame[time_col], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"time_col '{time_col}' contains unparseable or missing values.")
    for fold in folds:
        if parsed.iloc[fold.train_idx].max() >= parsed.iloc[fold.valid_idx].min():
            raise ValueError(f"Time leakage detected in fold '{fold.name}' using '{time_col}'.")


def train(config_path: str | Path) -> str:
    config = load_config(config_path)
    ensure_output_dirs(config)
    seed = int(config.get("runtime", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = config.get("data", {})
    task_cfg = config.get("task", {})
    cv_cfg = config.get("cv", {})
    features_cfg = config.get("features", {})
    target_col = data_cfg.get("target_col")
    id_col = data_cfg.get("id_col")
    season_col = data_cfg.get("season_col")
    time_col = data_cfg.get("time_col")
    entity_col = data_cfg.get("entity_id_col")
    problem_type = task_cfg.get("problem_type")
    metric_settings = resolve_metric_settings(task_cfg)
    metric = metric_settings["name"]
    metric_callable = metric_settings["metric_callable"]
    configured_metric_direction = metric_settings["direction"]
    metric_clip = bool(metric_settings["clip_at_zero"])
    metric_scale = float(metric_settings["scale"])
    for key, value in [
        ("data.target_col", target_col), ("data.id_col", id_col),
        ("data.season_col", season_col), ("task.problem_type", problem_type),
        ("task.metric", metric),
    ]:
        assert_not_placeholder(value, key)
    validate_metric(problem_type, metric, metric_callable)
    if metric_callable and not configured_metric_direction:
        raise ValueError("task.metric_direction is required when task.metric_callable is configured.")
    resolved_metric_direction = metric_direction(metric, configured_metric_direction)

    loaded = load_competition_data(config)
    train_df, source_positions = sort_temporal_frame(
        loaded["train"],
        list(data_cfg.get("order_cols", [])),
        season_col,
        time_col,
    )
    if entity_col and entity_col not in train_df:
        raise ValueError(f"entity_id_col '{entity_col}' not found in train data.")
    _validate_binary_target(train_df[target_col])
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    strategy = cv_cfg.get("strategy")
    if strategy == "season_holdout":
        primary = cv_cfg.get("primary") or {}
        folds = [
            build_season_holdout(
                train_df,
                season_col,
                list(primary.get("train_seasons") or []),
                list(primary.get("valid_seasons") or []),
                name="season_holdout",
            )
        ]
    elif strategy == "expanding_window":
        folds = build_temporal_folds(
            train_df,
            season_col=season_col,
            min_train_seasons=int(cv_cfg.get("min_train_seasons", 2)),
            validation_seasons=int(cv_cfg.get("validation_seasons", 1)),
            holdout_seasons=int(cv_cfg.get("holdout_seasons", 1)),
            gap_seasons=int(cv_cfg.get("gap_seasons", 0)),
        )
    else:
        raise ValueError(
            "cv.strategy must be 'season_holdout' or 'expanding_window'; "
            "random KFold is intentionally disabled."
        )
    _validate_chronology(train_df, folds, time_col)

    experiment_id = make_experiment_id("temporal")
    models_root = ensure_dir(resolve_path(config["output"]["models_dir"]) / experiment_id)
    pred_root = ensure_dir(resolve_path(config["output"]["predictions_dir"]) / experiment_id)
    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    completed_models: list[str] = []
    missing_policy = config.get("models", {}).get("missing_dependency", "error")

    for model_no, (model_name, params) in enumerate(_configured_models(config), start=1):
        try:
            build_model(model_name, params, seed)
        except ImportError as exc:
            if missing_policy == "skip":
                print(f"SKIP model={model_name}: {exc}")
                append_experiment_log(
                    config["output"]["experiments_path"],
                    {
                        "experiment_id": experiment_id,
                        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "purpose": config.get("project", {}).get("purpose", ""),
                        "data_version": config.get("project", {}).get("data_version", ""),
                        "features": to_json_text(features_cfg),
                        "model": model_name,
                        "params": to_json_text(params),
                        "cv_strategy": to_json_text(cv_cfg),
                        "metric": metric,
                        "metric_direction": resolved_metric_direction,
                        "status": "skipped",
                        "error": str(exc),
                        "seed": seed,
                        "notes": config.get("project", {}).get("notes", ""),
                        "next_action": "Install the optional dependency before comparing this model.",
                    },
                )
                continue
            raise
        model_dir = ensure_dir(models_root / model_name)
        model_predictions: list[pd.DataFrame] = []
        model_fold_scores: list[float] = []
        holdout_score = np.nan
        for fold_no, fold in enumerate(folds, start=1):
            pipeline = _make_pipeline(model_name, params, seed + model_no * 100 + fold_no, data_cfg, features_cfg)
            pipeline.fit(X.iloc[fold.train_idx], y.iloc[fold.train_idx])
            pred = positive_probability(pipeline, X.iloc[fold.valid_idx])
            fold_score = score_metric(
                y.iloc[fold.valid_idx],
                pred,
                metric,
                problem_type,
                metric_callable,
                clip_at_zero=metric_clip,
                scale=metric_scale,
            )
            if fold.split_type in {"expanding_window", "season_holdout"}:
                model_fold_scores.append(fold_score)
            else:
                holdout_score = fold_score
            joblib.dump(pipeline, model_dir / f"{fold.name}.joblib")

            part = train_df.iloc[fold.valid_idx][[id_col, target_col, season_col]].copy()
            if entity_col:
                part[entity_col] = train_df.iloc[fold.valid_idx][entity_col].to_numpy()
                known = set(train_df.iloc[fold.train_idx][entity_col].dropna())
                part["entity_status"] = np.where(part[entity_col].isin(known), "seen", "new")
            part["row_position"] = source_positions[fold.valid_idx]
            part["model"] = model_name
            part["fold"] = fold.name
            part["split_type"] = fold.split_type
            part["prediction"] = pred
            part.to_csv(pred_root / f"{model_name}__{fold.name}.csv", index=False)
            model_predictions.append(part)
            fold_metrics.append(
                {
                    "model": model_name, "fold": fold.name, "split_type": fold.split_type,
                    "train_seasons": "|".join(map(str, fold.train_seasons)),
                    "valid_seasons": "|".join(map(str, fold.valid_seasons)),
                    "n_train": len(fold.train_idx), "n_valid": len(fold.valid_idx), "score": fold_score,
                }
            )
            print(f"model={model_name} fold={fold.name} score={fold_score:.6f}")

        final_model = _make_pipeline(model_name, params, seed + model_no * 1000, data_cfg, features_cfg)
        final_model.fit(X, y)
        joblib.dump(final_model, model_dir / "final_model.joblib")
        joined = pd.concat(model_predictions, ignore_index=True)
        all_predictions.append(joined)
        completed_models.append(model_name)
        cv_score = float(np.mean(model_fold_scores)) if model_fold_scores else np.nan
        append_experiment_log(
            config["output"]["experiments_path"],
            {
                "experiment_id": experiment_id, "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "purpose": config.get("project", {}).get("purpose", ""),
                "data_version": config.get("project", {}).get("data_version", ""),
                "features": to_json_text(features_cfg), "model": model_name,
                "params": to_json_text(params), "cv_strategy": to_json_text(cv_cfg),
                "metric": metric, "metric_direction": resolved_metric_direction,
                "cv_score": cv_score, "cv_std": float(np.std(model_fold_scores)) if model_fold_scores else np.nan,
                "holdout_score": holdout_score, "seed": seed,
                "status": "success", "error": "",
                "notes": config.get("project", {}).get("notes", ""),
                "next_action": config.get("project", {}).get("next_action", ""),
            },
        )

    if not completed_models:
        raise RuntimeError("No model completed. Install optional dependencies or enable a core baseline.")
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(pred_root / "all_validation_predictions.csv", index=False)
    oof_mask = predictions["split_type"].isin(["expanding_window", "season_holdout"])
    predictions.loc[oof_mask].sort_values(
        ["model", "row_position"], kind="stable"
    ).to_csv(pred_root / "oof_predictions.csv", index=False)
    predictions.loc[predictions["split_type"] == "recent_holdout"].sort_values(
        ["model", "row_position"], kind="stable"
    ).to_csv(pred_root / "holdout_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(pred_root / "fold_metrics.csv", index=False)
    save_diagnostics(
        predictions, pred_root, target_col, metric, season_col, entity_col,
        int(config.get("diagnostics", {}).get("calibration_bins", 10)),
        int(config.get("diagnostics", {}).get("min_group_rows", 30)),
        metric_callable,
    )
    print(f"experiment_id={experiment_id}")
    print(f"completed_models={','.join(completed_models)}")
    print(f"predictions={pred_root}")
    return experiment_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    try:
        train(args.config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
