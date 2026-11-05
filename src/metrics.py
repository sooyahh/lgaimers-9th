from __future__ import annotations

"""Official Phase 2 probability metrics: Brier Score and Brier Skill Score.

Official formula (DACON evaluation page / Competition Spec §2):

    Brier Score = mean((p_i - y_i)^2)
    r = mean(y)   # evaluation-set (or local fold) positive rate
    Score = max(0, 100000 * (1 - BrierScore / (r * (1 - r))))

Single-class policy
-------------------
When r * (1 - r) == 0 (all labels identical), the official denominator is zero.
This module **raises ValueError** and does not invent a score. Callers must
exclude or specially handle such folds; do not silently return 0 or 100000.
"""

from typing import Any

import numpy as np

OFFICIAL_BSS_SCALE = 100_000


def _as_1d_float(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array; got shape={arr.shape}.")
    return arr


def _validate_binary_targets(y_true: np.ndarray) -> np.ndarray:
    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or inf.")
    unique = np.unique(y_true)
    if not np.isin(unique, [0.0, 1.0]).all():
        raise ValueError(
            f"y_true must be binary {{0, 1}}; found unique values={unique.tolist()}."
        )
    return y_true


def _validate_probabilities(y_prob: np.ndarray) -> np.ndarray:
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or inf.")
    if ((y_prob < 0.0) | (y_prob > 1.0)).any():
        raise ValueError("y_prob values must be in [0, 1].")
    return y_prob


def _validate_aligned(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    y = _validate_binary_targets(_as_1d_float(y_true, "y_true"))
    p = _validate_probabilities(_as_1d_float(y_prob, "y_prob"))
    if len(y) != len(p):
        raise ValueError(
            f"y_true and y_prob length mismatch: len(y_true)={len(y)}, len(y_prob)={len(p)}."
        )
    if len(y) == 0:
        raise ValueError("y_true and y_prob must be non-empty.")
    return y, p


def brier_score(y_true: Any, y_prob: Any) -> float:
    """Mean squared error between binary labels and predicted probabilities."""
    y, p = _validate_aligned(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def reference_brier_score(y_true: Any) -> float:
    """Climatology Brier score r * (1 - r) for the given label vector."""
    y = _validate_binary_targets(_as_1d_float(y_true, "y_true"))
    if len(y) == 0:
        raise ValueError("y_true must be non-empty.")
    r = float(np.mean(y))
    return float(r * (1.0 - r))


def brier_skill_score(
    y_true: Any,
    y_prob: Any,
    *,
    scale: float = OFFICIAL_BSS_SCALE,
    clip_at_zero: bool = True,
) -> float:
    """Scaled Brier Skill Score. Official model-selection uses clip_at_zero=True."""
    y, p = _validate_aligned(y_true, y_prob)
    bs = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    denom = r * (1.0 - r)
    if denom == 0.0:
        raise ValueError(
            "Brier Skill Score is undefined when r*(1-r)==0 "
            "(single-class y_true). Exclude or specially handle this fold; "
            "do not invent a score."
        )
    raw = scale * (1.0 - bs / denom)
    if clip_at_zero:
        return float(max(0.0, raw))
    return float(raw)


def compute_brier_metrics(
    y_true: Any,
    y_prob: Any,
    *,
    scale: float = OFFICIAL_BSS_SCALE,
) -> dict[str, float | int]:
    """Return sample count, target rate, Brier, raw BSS, and official (clipped) BSS."""
    y, p = _validate_aligned(y_true, y_prob)
    bs = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    denom = r * (1.0 - r)
    if denom == 0.0:
        raise ValueError(
            "Brier Skill Score is undefined when r*(1-r)==0 "
            "(single-class y_true). Exclude or specially handle this fold; "
            "do not invent a score."
        )
    raw = float(scale * (1.0 - bs / denom))
    official = float(max(0.0, raw))
    return {
        "n_samples": int(len(y)),
        "target_rate": r,
        "brier_score": bs,
        "raw_brier_skill_score": raw,
        "official_brier_skill_score": official,
    }
