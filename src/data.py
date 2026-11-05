from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.submission import normalize_key_cols, validate_key_correspondence
from src.utils import assert_not_placeholder, resolve_path


def _read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path}\n"
            "Place the official competition file under data/raw/ or update configs/baseline.yaml."
        )
    return pd.read_csv(path)


def summarize_dataframe(df: pd.DataFrame, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isna().sum().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
    }


def validate_required_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    target_col: str,
    id_col: str,
    season_col: str | None = None,
    time_col: str | None = None,
    order_cols: list[str] | None = None,
    submission_key_cols: list[str] | None = None,
) -> None:
    if target_col not in train.columns:
        raise ValueError(f"target_col '{target_col}' not found in train columns.")
    if target_col in test.columns:
        raise ValueError(
            f"target_col '{target_col}' exists in test data. Check for target leakage."
        )
    if id_col not in train.columns:
        raise ValueError(f"id_col '{id_col}' not found in train columns.")
    if id_col not in test.columns:
        raise ValueError(f"id_col '{id_col}' not found in test columns.")
    for label, col in [("season_col", season_col), ("time_col", time_col)]:
        if col and col not in train.columns:
            raise ValueError(f"{label} '{col}' not found in train columns.")
    validate_prediction_ids(test, sample_submission, submission_key_cols or [id_col])
    missing_order_cols = [col for col in (order_cols or []) if col not in train.columns]
    if missing_order_cols:
        raise ValueError(f"order_cols not found in train data: {missing_order_cols}")
    missing_test_features = sorted(set(train.columns) - {target_col} - set(test.columns))
    if missing_test_features:
        raise ValueError(f"test is missing train feature columns: {missing_test_features}")


def load_competition_data(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data_cfg = config.get("data", {})
    target_col = data_cfg.get("target_col")
    id_col = data_cfg.get("id_col")
    assert_not_placeholder(target_col, "data.target_col")
    assert_not_placeholder(id_col, "data.id_col")

    train_path = resolve_path(data_cfg.get("train_path", "data/raw/train.csv"))
    test_path = resolve_path(data_cfg.get("test_path", "data/raw/test.csv"))
    sample_path = resolve_path(
        data_cfg.get("sample_submission_path", "data/raw/sample_submission.csv")
    )

    train = _read_csv_required(train_path, "train")
    test = _read_csv_required(test_path, "test")
    sample_submission = _read_csv_required(sample_path, "sample_submission")
    validate_required_columns(
        train,
        test,
        sample_submission,
        target_col,
        id_col,
        data_cfg.get("season_col"),
        data_cfg.get("time_col"),
        data_cfg.get("order_cols", []),
        configured_submission_key_cols(data_cfg),
    )
    return {
        "train": train,
        "test": test,
        "sample_submission": sample_submission,
    }


def load_prediction_data(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data_cfg = config.get("data", {})
    id_col = data_cfg.get("id_col")
    assert_not_placeholder(id_col, "data.id_col")

    test_path = resolve_path(data_cfg.get("test_path", "data/raw/test.csv"))
    sample_path = resolve_path(
        data_cfg.get("sample_submission_path", "data/raw/sample_submission.csv")
    )
    test = _read_csv_required(test_path, "test")
    sample_submission = _read_csv_required(sample_path, "sample_submission")
    if id_col not in test.columns:
        raise ValueError(f"id_col '{id_col}' not found in test columns.")
    validate_prediction_ids(test, sample_submission, configured_submission_key_cols(data_cfg))
    return {"test": test, "sample_submission": sample_submission}


def validate_prediction_ids(
    test: pd.DataFrame, sample_submission: pd.DataFrame, key_cols: str | list[str]
) -> None:
    validate_key_correspondence(test, sample_submission, key_cols)


def configured_submission_key_cols(data_cfg: dict[str, Any]) -> list[str]:
    id_col = data_cfg.get("id_col")
    configured = data_cfg.get("submission_key_cols")
    keys = normalize_key_cols(configured if configured is not None else id_col)
    for index, key in enumerate(keys):
        assert_not_placeholder(key, f"data.submission_key_cols[{index}]")
    return keys


def sort_temporal_frame(
    frame: pd.DataFrame,
    order_cols: list[str],
    season_col: str,
    time_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Stable lexicographic ordering with original row positions retained for OOF export."""
    if not order_cols:
        return frame.reset_index(drop=True), np.arange(len(frame))
    if season_col not in order_cols:
        raise ValueError("data.order_cols must include data.season_col.")
    missing = [col for col in order_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"order_cols not found in train data: {missing}")
    null_counts = frame[order_cols].isna().sum()
    null_columns = {col: int(count) for col, count in null_counts.items() if count}
    if null_columns:
        raise ValueError(f"data.order_cols contain missing values: {null_columns}")
    if frame.duplicated(order_cols).any():
        raise ValueError(
            "data.order_cols do not uniquely order pitches. Stable sort cannot recover the actual "
            "time order of tied rows; configure an official sequence or tie-breaker column."
        )
    working = frame.copy()
    working["__source_row_position"] = np.arange(len(working))
    sort_cols = list(order_cols)
    temporary_time_col = None
    if time_col and time_col in order_cols:
        parsed = pd.to_datetime(working[time_col], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"time_col '{time_col}' contains unparseable or missing values.")
        temporary_time_col = "__parsed_time_order"
        working[temporary_time_col] = parsed
        sort_cols[sort_cols.index(time_col)] = temporary_time_col
    working = working.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    source_positions = working.pop("__source_row_position").to_numpy()
    if temporary_time_col:
        working = working.drop(columns=[temporary_time_col])
    return working, source_positions


def build_data_summary(config: dict[str, Any]) -> dict[str, Any]:
    data = load_competition_data(config)
    return {
        name: summarize_dataframe(df, name)
        for name, df in data.items()
    }
