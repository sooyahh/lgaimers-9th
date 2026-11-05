from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import configured_submission_key_cols, load_prediction_data
from src.models import positive_probability
from src.submission import align_predictions_to_sample, prediction_columns, save_and_recheck_submission
from src.utils import assert_not_placeholder, ensure_dir, load_config, resolve_path


def _latest_experiment_id(models_dir: Path) -> str | None:
    candidates = [p for p in models_dir.iterdir() if p.is_dir()] if models_dir.exists() else []
    return max(candidates, key=lambda p: p.stat().st_mtime).name if candidates else None


def predict(config_path: str | Path, experiment_id: str | None = None) -> Path:
    config = load_config(config_path)
    id_col = config.get("data", {}).get("id_col")
    assert_not_placeholder(id_col, "data.id_col")
    loaded = load_prediction_data(config)
    test, sample = loaded["test"], loaded["sample_submission"]
    submission_keys = configured_submission_key_cols(config.get("data", {}))
    pred_cols = prediction_columns(sample, submission_keys)
    if len(pred_cols) != 1:
        raise ValueError("Binary probability task requires exactly one submission prediction column.")

    models_root = resolve_path(config["output"]["models_dir"])
    experiment_id = experiment_id or _latest_experiment_id(models_root)
    if not experiment_id:
        raise FileNotFoundError("No trained experiment found. Run training first.")
    experiment_dir = models_root / experiment_id
    requested = config.get("submission", {}).get("models")
    model_dirs = [experiment_dir / name for name in requested] if requested else [p for p in experiment_dir.iterdir() if p.is_dir()]
    rows: dict[str, np.ndarray] = {}
    for model_dir in sorted(model_dirs):
        path = model_dir / "final_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Final model not found: {path}")
        rows[model_dir.name] = positive_probability(joblib.load(path), test)
    if not rows:
        raise FileNotFoundError(f"No final models found under {experiment_dir}")

    test_predictions = test[submission_keys].copy()
    for name, values in rows.items():
        test_predictions[name] = values
    test_predictions["ensemble_mean"] = np.mean(np.column_stack(list(rows.values())), axis=1)
    test_predictions = align_predictions_to_sample(test_predictions, sample, submission_keys)
    pred_dir = ensure_dir(resolve_path(config["output"]["predictions_dir"]) / experiment_id)
    test_predictions.to_csv(pred_dir / "test_predictions.csv", index=False)

    submission = sample.copy()
    submission[pred_cols[0]] = test_predictions["ensemble_mean"].to_numpy()
    output_dir = ensure_dir(resolve_path(config["output"]["submissions_dir"]))
    return save_and_recheck_submission(
        sample, submission, submission_keys,
        output_dir / f"submission_{experiment_id}.csv", probability=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    try:
        print(f"submission={predict(args.config, args.experiment_id)}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
