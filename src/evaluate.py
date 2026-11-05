from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.metrics import OFFICIAL_BSS_SCALE, compute_brier_metrics


CLASSIFICATION_METRICS = {"roc_auc", "log_loss", "brier", "brier_skill_score"}


def resolve_metric_settings(task_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize flat or nested task.metric config."""
    task_cfg = task_cfg or {}
    metric = task_cfg.get("metric")
    if isinstance(metric, dict):
        name = metric.get("name")
        direction = metric.get("direction", task_cfg.get("metric_direction"))
        clip_at_zero = bool(metric.get("clip_at_zero", True))
        scale = float(metric.get("scale", OFFICIAL_BSS_SCALE))
        metric_callable = metric.get("metric_callable", task_cfg.get("metric_callable"))
    else:
        name = metric
        direction = task_cfg.get("metric_direction")
        clip_at_zero = True
        scale = float(OFFICIAL_BSS_SCALE)
        metric_callable = task_cfg.get("metric_callable")
    return {
        "name": name,
        "direction": direction,
        "clip_at_zero": clip_at_zero,
        "scale": scale,
        "metric_callable": metric_callable,
    }


def load_metric_callable(path: str | None) -> Callable[[Any, Any], float] | None:
    if not path:
        return None
    if ":" not in path:
        raise ValueError("task.metric_callable must use 'module:function' syntax.")
    module_name, function_name = path.split(":", 1)
    function = getattr(import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"Custom metric callable not found: {path}")
    return function


def validate_metric(problem_type: str, metric: str, metric_callable: str | None = None) -> None:
    if problem_type != "binary_classification":
        raise ValueError("This scaffold intentionally supports only binary_classification.")
    if metric not in CLASSIFICATION_METRICS and not metric_callable:
        raise ValueError(f"Unsupported binary probability metric: {metric}")
    load_metric_callable(metric_callable)


def metric_direction(metric: str, configured_direction: str | None = None) -> str:
    default = "maximize" if metric in {"roc_auc", "brier_skill_score"} else "minimize"
    direction = configured_direction or default
    if direction not in {"maximize", "minimize"}:
        raise ValueError("task.metric_direction must be 'maximize' or 'minimize'.")
    return direction


def _positive_probs(y_pred: Any) -> np.ndarray:
    p = np.asarray(y_pred, dtype=float)
    if p.ndim == 2:
        if p.shape[1] != 2:
            raise ValueError("Binary probability prediction must have exactly two columns.")
        p = p[:, 1]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probability predictions must be finite and in [0, 1].")
    return p


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    *,
    scale: float = OFFICIAL_BSS_SCALE,
) -> dict[str, float | int]:
    """Official evaluation bundle for model selection and diagnostics.

    Returns:
        n_samples, target_rate, brier_score,
        raw_brier_skill_score, official_brier_skill_score

    Model selection uses official_brier_skill_score = max(0, raw_bss).
    raw_brier_skill_score is retained for diagnostics when predictions are
    worse than climatology.
    """
    p = _positive_probs(y_pred)
    return compute_brier_metrics(y_true, p, scale=scale)


def score_metric(
    y_true: Any,
    y_pred: Any,
    metric: str,
    problem_type: str,
    metric_callable: str | None = None,
    *,
    clip_at_zero: bool = True,
    scale: float = OFFICIAL_BSS_SCALE,
) -> float:
    validate_metric(problem_type, metric, metric_callable)
    y = np.asarray(y_true)
    p = _positive_probs(y_pred)
    if metric_callable:
        value = float(load_metric_callable(metric_callable)(y, p))
        if not np.isfinite(value):
            raise ValueError("Custom metric returned a non-finite value.")
        return value
    if metric == "roc_auc":
        if np.unique(y).size < 2:
            raise ValueError("roc_auc is undefined when y_true has one class.")
        return float(roc_auc_score(y, p))
    if metric == "log_loss":
        return float(log_loss(y, p, labels=[0, 1]))
    if metric == "brier":
        return float(brier_score_loss(y, p))
    if metric == "brier_skill_score":
        bundle = compute_brier_metrics(y, p, scale=scale)
        if clip_at_zero:
            return float(bundle["official_brier_skill_score"])
        return float(bundle["raw_brier_skill_score"])
    raise ValueError(f"Unsupported metric: {metric}")


@dataclass(frozen=True)
class TemporalFold:
    name: str
    split_type: str
    train_seasons: tuple[Any, ...]
    valid_seasons: tuple[Any, ...]
    train_idx: np.ndarray
    valid_idx: np.ndarray


def build_season_holdout(
    frame: pd.DataFrame,
    season_col: str,
    train_seasons: list[Any] | tuple[Any, ...],
    valid_seasons: list[Any] | tuple[Any, ...],
    name: str = "season_holdout",
) -> TemporalFold:
    """Explicit season holdout (e.g. 2019-2023 train / 2024 valid)."""
    if season_col not in frame:
        raise ValueError(f"season_col '{season_col}' not found.")
    train_seasons_t = tuple(train_seasons)
    valid_seasons_t = tuple(valid_seasons)
    if not train_seasons_t or not valid_seasons_t:
        raise ValueError("train_seasons and valid_seasons must be non-empty.")
    if set(train_seasons_t) & set(valid_seasons_t):
        raise ValueError("train_seasons and valid_seasons must be disjoint.")
    if max(train_seasons_t) >= min(valid_seasons_t):
        raise ValueError(
            "train seasons must be strictly earlier than valid seasons "
            f"(train_max={max(train_seasons_t)}, valid_min={min(valid_seasons_t)})."
        )
    return _make_fold(
        frame, season_col, name, "season_holdout", train_seasons_t, valid_seasons_t
    )


def build_temporal_folds(
    frame: pd.DataFrame,
    season_col: str,
    min_train_seasons: int,
    validation_seasons: int = 1,
    holdout_seasons: int = 1,
    gap_seasons: int = 0,
) -> list[TemporalFold]:
    """Create expanding folds on development seasons plus a latest-season holdout."""
    if season_col not in frame:
        raise ValueError(f"season_col '{season_col}' not found.")
    if frame[season_col].isna().any():
        raise ValueError("season_col contains missing values; temporal ordering is ambiguous.")
    seasons = tuple(sorted(frame[season_col].unique().tolist()))
    required = min_train_seasons + validation_seasons + holdout_seasons + gap_seasons
    if len(seasons) < required:
        raise ValueError(
            f"Need at least {required} seasons, found {len(seasons)}: {list(seasons)}"
        )
    development = seasons[:-holdout_seasons]
    folds: list[TemporalFold] = []
    fold_no = 1
    max_valid_start = len(development) - validation_seasons
    for valid_start in range(min_train_seasons + gap_seasons, max_valid_start + 1):
        train_end = valid_start - gap_seasons
        train_seasons = development[:train_end]
        valid_season_values = development[valid_start : valid_start + validation_seasons]
        folds.append(
            _make_fold(
                frame,
                season_col,
                f"expanding_{fold_no}",
                "expanding_window",
                train_seasons,
                valid_season_values,
            )
        )
        fold_no += 1

    holdout_valid = seasons[-holdout_seasons:]
    holdout_train_end = len(seasons) - holdout_seasons - gap_seasons
    holdout_train = seasons[:holdout_train_end]
    folds.append(
        _make_fold(frame, season_col, "recent_holdout", "recent_holdout", holdout_train, holdout_valid)
    )
    return folds


def _make_fold(
    frame: pd.DataFrame,
    season_col: str,
    name: str,
    split_type: str,
    train_seasons: tuple[Any, ...],
    valid_seasons: tuple[Any, ...],
) -> TemporalFold:
    train_idx = np.flatnonzero(frame[season_col].isin(train_seasons).to_numpy())
    valid_idx = np.flatnonzero(frame[season_col].isin(valid_seasons).to_numpy())
    if not len(train_idx) or not len(valid_idx):
        raise ValueError(f"Temporal fold '{name}' has an empty train or validation partition.")
    if max(train_seasons) >= min(valid_seasons):
        raise AssertionError(f"Temporal leakage in fold '{name}'.")
    return TemporalFold(name, split_type, tuple(train_seasons), tuple(valid_seasons), train_idx, valid_idx)


def calibration_table(y_true: Any, y_pred: Any, n_bins: int = 10) -> pd.DataFrame:
    work = pd.DataFrame({"target": np.asarray(y_true), "prediction": np.asarray(y_pred)})
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    work["bin"] = pd.cut(work["prediction"], edges, include_lowest=True, duplicates="drop")
    result = work.groupby("bin", observed=False).agg(
        count=("target", "size"), mean_prediction=("prediction", "mean"), observed_rate=("target", "mean")
    )
    return result.reset_index().assign(bin=lambda x: x["bin"].astype(str))


def safe_group_scores(
    frame: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    pred_col: str,
    metric: str,
    min_rows: int = 1,
    metric_callable: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: Any = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, part in frame.groupby(grouper, dropna=False, sort=False):
        keys = (keys,) if len(group_cols) == 1 else tuple(keys)
        row = dict(zip(group_cols, keys))
        row["n_rows"] = len(part)
        row["n_positive"] = int((part[target_col] == 1).sum())
        row["n_negative"] = int((part[target_col] == 0).sum())
        row["positive_rate"] = float(part[target_col].mean())
        if len(part) < min_rows:
            row["eligible_for_score"] = False
            row["score"] = np.nan
            row["score_reason"] = f"below_min_rows:{min_rows}"
            rows.append(row)
            continue
        try:
            row["score"] = score_metric(
                part[target_col], part[pred_col], metric, "binary_classification", metric_callable
            )
            row["eligible_for_score"] = True
            row["score_reason"] = "ok"
        except ValueError as exc:
            row["eligible_for_score"] = False
            row["score"] = np.nan
            row["score_reason"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)
