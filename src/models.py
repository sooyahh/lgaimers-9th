from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class GlobalMeanClassifier(BaseEstimator, ClassifierMixin):
    """Leakage-safe constant baseline; the mean is learned within each training fold."""

    def fit(self, X: Any, y: Any) -> "GlobalMeanClassifier":
        self.positive_rate_ = float(np.asarray(y).mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.full(len(X), self.positive_rate_, dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_model(name: str, params: dict[str, Any], seed: int) -> Any:
    name = name.lower()
    model_params = dict(params)
    if name == "global_mean":
        return GlobalMeanClassifier()
    if name == "logistic_regression":
        model_params.setdefault("max_iter", 1000)
        model_params.setdefault("random_state", seed)
        return LogisticRegression(**model_params)
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("catboost is not installed; install the optional dependency.") from exc
        model_params.setdefault("random_seed", seed)
        model_params.setdefault("verbose", False)
        return CatBoostClassifier(**model_params)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is not installed; install the optional dependency.") from exc
        model_params.setdefault("random_state", seed)
        model_params.setdefault("verbosity", -1)
        return LGBMClassifier(**model_params)
    raise ValueError(f"Unsupported model '{name}'.")


def positive_probability(model: Any, X: Any) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{type(model).__name__} does not provide predict_proba.")
    pred = np.asarray(model.predict_proba(X), dtype=float)
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"Expected binary predict_proba output, got {pred.shape}.")
    return pred[:, 1]
