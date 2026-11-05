from __future__ import annotations

"""EXP-000: training-split target-mean constant baseline on season holdout."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import configured_submission_key_cols, load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.submission import prediction_columns, save_and_recheck_submission
from src.utils import (
    append_experiment_log,
    ensure_dir,
    ensure_output_dirs,
    load_config,
    resolve_path,
    save_json,
    to_json_text,
)


EXPERIMENT_ID = "EXP-000"


def run_exp000(config_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    ensure_output_dirs(config)

    data_cfg = config["data"]
    task_cfg = config.get("task", {})
    cv_cfg = config.get("cv", {})
    metric_settings = resolve_metric_settings(task_cfg)
    if metric_settings["name"] != "brier_skill_score":
        raise ValueError("EXP-000 expects task.metric.name = brier_skill_score.")
    if not metric_settings["clip_at_zero"]:
        raise ValueError("Official model selection requires clip_at_zero=true.")

    primary = cv_cfg.get("primary") or {}
    train_seasons = list(primary.get("train_seasons") or [])
    valid_seasons = list(primary.get("valid_seasons") or [])
    if train_seasons != [2019, 2020, 2021, 2022, 2023] or valid_seasons != [2024]:
        raise ValueError(
            "EXP-000 primary split must be train=[2019..2023], valid=[2024]; "
            f"got train={train_seasons}, valid={valid_seasons}."
        )
    if data_cfg.get("order_cols"):
        raise ValueError("EXP-000 keeps order_cols empty; do not invent pitch order.")

    target_col = data_cfg["target_col"]
    id_col = data_cfg["id_col"]
    season_col = data_cfg["season_col"]
    submission_keys = configured_submission_key_cols(data_cfg)

    loaded = load_competition_data(config)
    train_df = loaded["train"]
    test_df = loaded["test"]
    sample = loaded["sample_submission"]

    fold = build_season_holdout(
        train_df,
        season_col,
        train_seasons,
        valid_seasons,
        name="season_holdout_2024",
    )
    y_train = train_df.iloc[fold.train_idx][target_col]
    y_valid = train_df.iloc[fold.valid_idx][target_col]
    train_target_mean = float(y_train.mean())
    valid_target_mean = float(y_valid.mean())

    # Valid predictions use ONLY the training-split mean (not full-data mean).
    valid_pred = np.full(len(fold.valid_idx), train_target_mean, dtype=float)
    metrics = evaluate_predictions(
        y_valid,
        valid_pred,
        scale=float(metric_settings["scale"]),
    )
    selection_score = float(metrics["official_brier_skill_score"])

    pred_dir = ensure_dir(
        resolve_path(config["output"]["predictions_dir"]) / EXPERIMENT_ID
    )
    model_dir = ensure_dir(resolve_path(config["output"]["models_dir"]) / EXPERIMENT_ID)
    sub_dir = ensure_dir(resolve_path(config["output"]["submissions_dir"]))

    holdout_predictions = train_df.iloc[fold.valid_idx][[id_col, season_col, target_col]].copy()
    holdout_predictions["prediction"] = valid_pred
    holdout_predictions["split"] = fold.name
    holdout_predictions["train_target_mean"] = train_target_mean
    holdout_path = pred_dir / "holdout_predictions.csv"
    holdout_predictions.to_csv(holdout_path, index=False)

    # Constant baseline: store metadata only (no model weights file required).
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "model": "target_mean",
        "constant_probability": train_target_mean,
        "fitted_on": {
            "split": fold.name,
            "train_seasons": list(fold.train_seasons),
            "valid_seasons": list(fold.valid_seasons),
            "n_train": int(len(fold.train_idx)),
            "n_valid": int(len(fold.valid_idx)),
        },
        "notes": [
            "Valid predictions use training-split target mean only.",
            "Full-data target mean is not used for validation predictions.",
            "order_cols left empty; row_id not treated as a time key.",
            "No game_id; season holdout still used as primary validation.",
            "Seed unused: deterministic constant baseline.",
        ],
        "metric_settings": metric_settings,
    }
    metadata_path = save_json(metadata, model_dir / "target_mean_metadata.json")

    # Sample test (5 rows): use the same fitted training-split mean.
    pred_col = prediction_columns(sample, submission_keys)[0]
    submission = sample.copy()
    submission[pred_col] = train_target_mean
    submission_path = sub_dir / f"submission_{EXPERIMENT_ID}_target_mean.csv"
    submission_validation = {"ok": False, "error": None}
    try:
        save_and_recheck_submission(
            sample, submission, submission_keys, submission_path, probability=True
        )
        submission_validation = {"ok": True, "error": None}
    except Exception as exc:  # noqa: BLE001 - record validator failure for the report
        submission_validation = {"ok": False, "error": str(exc)}
        raise

    elapsed = float(time.perf_counter() - started)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "model": "target_mean",
        "train_target_mean": train_target_mean,
        "valid_target_mean": valid_target_mean,
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
        "train_seasons": list(fold.train_seasons),
        "valid_seasons": list(fold.valid_seasons),
        "brier_score": float(metrics["brier_score"]),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": selection_score,
        "target_rate_valid": float(metrics["target_rate"]),
        "n_samples_valid": int(metrics["n_samples"]),
        "prediction_path": str(holdout_path),
        "submission_path": str(submission_path),
        "metadata_path": str(metadata_path),
        "submission_validation": submission_validation,
        "elapsed_seconds": elapsed,
        "seed": None,
        "seed_reason": "unused: deterministic constant baseline (no RNG)",
        "order_cols": list(data_cfg.get("order_cols") or []),
    }
    result_path = save_json(result, pred_dir / "metrics.json")

    report_lines = [
        f"# {EXPERIMENT_ID} Target Mean Baseline",
        "",
        f"- date: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- split: train seasons `{fold.train_seasons}` / valid seasons `{fold.valid_seasons}`",
        f"- n_train / n_valid: `{len(fold.train_idx)}` / `{len(fold.valid_idx)}`",
        f"- train target mean (fitted constant): `{train_target_mean:.10f}`",
        f"- valid target mean: `{valid_target_mean:.10f}`",
        f"- Brier Score: `{metrics['brier_score']:.10f}`",
        f"- raw BSS: `{metrics['raw_brier_skill_score']:.10f}`",
        f"- official BSS (max(0, raw)): `{selection_score:.10f}`",
        f"- prediction path: `{holdout_path}`",
        f"- submission path: `{submission_path}`",
        f"- submission validation: `{submission_validation}`",
        f"- metadata path: `{metadata_path}`",
        f"- metrics json: `{result_path}`",
        f"- elapsed_seconds: `{elapsed:.3f}`",
        f"- seed: unused ({result['seed_reason']})",
        "",
        "## Notes",
        "",
        "- Valid predictions use training-split mean only (not full-data mean).",
        "- order_cols remained empty; row_id not confirmed as a time key.",
        "- Season holdout used despite missing game id.",
        "- No Logistic Regression / GBDT / Trackman / FE / Public LB submit.",
        "",
    ]
    report_path = pred_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    append_experiment_log(
        config["output"]["experiments_path"],
        {
            "experiment_id": EXPERIMENT_ID,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "target_mean_season_holdout_baseline",
            "data_version": config.get("project", {}).get("data_version", ""),
            "features": to_json_text({"used": "none", "constant": "train_split_target_mean"}),
            "model": "target_mean",
            "params": to_json_text({"constant_probability": train_target_mean}),
            "cv_strategy": to_json_text(
                {
                    "strategy": "season_holdout",
                    "train_seasons": list(fold.train_seasons),
                    "valid_seasons": list(fold.valid_seasons),
                }
            ),
            "metric": "brier_skill_score",
            "metric_direction": "maximize",
            "cv_score": "",
            "cv_std": "",
            "holdout_score": selection_score,
            "public_lb_score": "",
            "cv_lb_gap": "",
            "seed": "unused",
            "notes": to_json_text(
                {
                    "train_target_mean": train_target_mean,
                    "valid_target_mean": valid_target_mean,
                    "brier_score": float(metrics["brier_score"]),
                    "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
                    "official_brier_skill_score": selection_score,
                    "prediction_path": str(holdout_path),
                    "submission_path": str(submission_path),
                    "submission_validation": submission_validation,
                    "elapsed_seconds": elapsed,
                }
            ),
            "next_action": "Optional: Logistic Regression on same season holdout (no Trackman/FE yet).",
            "status": "success",
            "error": "",
        },
    )
    result["report_path"] = str(report_path)
    result["metrics_json_path"] = str(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP-000 target-mean baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    result = run_exp000(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
