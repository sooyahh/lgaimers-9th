from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dir


def normalize_key_cols(key_cols: str | list[str]) -> list[str]:
    normalized = [key_cols] if isinstance(key_cols, str) else list(key_cols)
    if not normalized:
        raise ValueError("At least one submission key column must be configured.")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Configured submission key columns must be unique.")
    return normalized


def prediction_columns(
    sample_submission: pd.DataFrame, key_cols: str | list[str]
) -> list[str]:
    keys = normalize_key_cols(key_cols)
    cols = [col for col in sample_submission.columns if col not in keys]
    if not cols:
        raise ValueError("sample_submission has no prediction columns.")
    return cols


def validate_key_correspondence(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
    key_cols: str | list[str],
) -> None:
    keys = normalize_key_cols(key_cols)
    for label, frame in [("predictions", predictions), ("sample_submission", sample_submission)]:
        missing_columns = [col for col in keys if col not in frame.columns]
        if missing_columns:
            raise ValueError(f"{label} is missing configured key columns: {missing_columns}")
        null_counts = frame[keys].isna().sum()
        null_columns = {col: int(count) for col, count in null_counts.items() if count}
        if null_columns:
            raise ValueError(f"{label} has missing submission key values: {null_columns}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} has duplicate submission keys for {keys}.")

    correspondence = predictions[keys].merge(
        sample_submission[keys], on=keys, how="outer", indicator=True, validate="one_to_one"
    )
    unexpected = int((correspondence["_merge"] == "left_only").sum())
    missing = int((correspondence["_merge"] == "right_only").sum())
    if missing or unexpected:
        raise ValueError(
            "Submission key mismatch: "
            f"missing_prediction_keys={missing}, unexpected_prediction_keys={unexpected}."
        )


def align_predictions_to_sample(
    test_predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
    key_cols: str | list[str],
) -> pd.DataFrame:
    """Return one-to-one predictions in exact sample-submission key order."""
    keys = normalize_key_cols(key_cols)
    validate_key_correspondence(test_predictions, sample_submission, keys)
    order = sample_submission[keys].copy()
    order["__sample_order"] = np.arange(len(order))
    aligned = order.merge(
        test_predictions, on=keys, how="left", validate="one_to_one", sort=False
    )
    value_cols = [col for col in test_predictions.columns if col not in keys]
    if aligned[value_cols].isna().any().any():
        raise ValueError("Some sample_submission keys have no test prediction.")
    aligned = aligned.sort_values("__sample_order", kind="stable").drop(columns="__sample_order")
    return aligned.reset_index(drop=True)


def validate_submission(
    sample_submission: pd.DataFrame,
    submission: pd.DataFrame,
    key_cols: str | list[str],
    probability: bool = True,
) -> None:
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError(
            "Submission columns do not match sample_submission.\n"
            f"expected={list(sample_submission.columns)}\n"
            f"actual={list(submission.columns)}"
        )
    if len(submission) != len(sample_submission):
        raise ValueError(
            f"Submission row count mismatch: expected {len(sample_submission)}, got {len(submission)}"
        )
    keys = normalize_key_cols(key_cols)
    validate_key_correspondence(submission, sample_submission, keys)
    if not submission[keys].reset_index(drop=True).equals(
        sample_submission[keys].reset_index(drop=True)
    ):
        raise ValueError("Submission key order does not match sample_submission.")

    pred_cols = prediction_columns(sample_submission, keys)
    values = submission[pred_cols]
    if values.isna().any().any():
        raise ValueError("Submission contains NaN values.")
    numeric_values = values.select_dtypes(include=[np.number])
    if len(numeric_values.columns) != len(pred_cols):
        raise ValueError("All prediction columns must be numeric.")
    if not np.isfinite(numeric_values.to_numpy()).all():
        raise ValueError("Submission contains inf or -inf values.")
    if probability and ((numeric_values < 0) | (numeric_values > 1)).any().any():
        raise ValueError("Probability submission values must be in [0, 1].")


def save_and_recheck_submission(
    sample_submission: pd.DataFrame,
    submission: pd.DataFrame,
    key_cols: str | list[str],
    path: str | Path,
    probability: bool = True,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    validate_submission(sample_submission, submission, key_cols, probability=probability)
    submission.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    validate_submission(sample_submission, reloaded, key_cols, probability=probability)
    return path
