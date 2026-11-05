"""RECENCY-AUDIT-001: uniform-weight training-window comparison on CatBoost L42.

Question: under sample_weight=1.0 for every training row, does keeping the oldest
season in an expanding window help next-season generalization, or does it hurt?

This is NOT an ablation of the current weighted canonical L42. Season recency
weights are intentionally disabled so the window effect is isolated.

Does not write submissions. Does not append to outputs/experiments.csv
(audit convention: VAL-AUDIT / MODEL-TRANSFER also stay out of that registry).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exp004_catboost_weighted import fit_catboost
from scripts.run_full_train_exp005 import ID_DROP
from scripts.run_l42_2019_2023_lb_probe import assert_l42_contract, l42_feature_lists
from scripts.tree_exp_common import positive_class_proba, prepare_native_frames
from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.metrics import reference_brier_score
from src.utils import ensure_dir, load_config, resolve_path, set_seed

AUDIT_ID = "RECENCY-AUDIT-001"
DEFAULT_CONFIG = "configs/baseline.yaml"
WEIGHT_POLICY = "uniform_1.0"
BACKEND = "catboost"
CONDITIONS = ("A_gap", "B_expanding", "C_shifted")
VALID_SEASONS = (2022, 2023, 2024)
EXPECTED_CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
}
EXPECTED_SEED = 42
CONTRAST_IDENTITY_ATOL = 1e-9
VERDICT_SHORTEN = "SHORTEN_WINDOW_CANDIDATE"
VERDICT_KEEP = "KEEP_FULL_HISTORY_CANDIDATE"
VERDICT_HOLD = "HOLD_MIXED_RECENCY_EFFECT"

# Locked experimental contract. Do not edit after seeing results.
LOCKED_WINDOWS: dict[int, dict[str, list[int]]] = {
    2022: {
        "A_gap": [2019, 2020],
        "B_expanding": [2019, 2020, 2021],
        "C_shifted": [2020, 2021],
    },
    2023: {
        "A_gap": [2019, 2020, 2021],
        "B_expanding": [2019, 2020, 2021, 2022],
        "C_shifted": [2020, 2021, 2022],
    },
    2024: {
        "A_gap": [2019, 2020, 2021, 2022],
        "B_expanding": [2019, 2020, 2021, 2022, 2023],
        "C_shifted": [2020, 2021, 2022, 2023],
    },
}


@dataclass(frozen=True)
class CellSpec:
    valid_season: int
    condition: str
    train_seasons: tuple[int, ...]

    @property
    def cell_id(self) -> str:
        return f"{self.valid_season}__{self.condition}"


def iter_cell_specs() -> list[CellSpec]:
    cells: list[CellSpec] = []
    for valid in VALID_SEASONS:
        for condition in CONDITIONS:
            seasons = LOCKED_WINDOWS[valid][condition]
            cells.append(
                CellSpec(
                    valid_season=int(valid),
                    condition=str(condition),
                    train_seasons=tuple(int(s) for s in seasons),
                )
            )
    if len(cells) != 9:
        raise AssertionError(f"expected 9 cells, got {len(cells)}")
    return cells


def assert_locked_window_contract(cells: list[CellSpec] | None = None) -> None:
    cells = cells if cells is not None else iter_cell_specs()
    by_valid: dict[int, dict[str, CellSpec]] = {}
    for cell in cells:
        by_valid.setdefault(cell.valid_season, {})[cell.condition] = cell
        if cell.valid_season in cell.train_seasons:
            raise AssertionError(f"{cell.cell_id}: valid season leaked into train")
        if any(s >= cell.valid_season for s in cell.train_seasons):
            raise AssertionError(f"{cell.cell_id}: future or valid season in train")
        if list(cell.train_seasons) != LOCKED_WINDOWS[cell.valid_season][cell.condition]:
            raise AssertionError(f"{cell.cell_id}: train seasons drifted from lock")

    for valid in VALID_SEASONS:
        a = by_valid[valid]["A_gap"]
        b = by_valid[valid]["B_expanding"]
        c = by_valid[valid]["C_shifted"]
        if len(a.train_seasons) != len(c.train_seasons):
            raise AssertionError(f"valid={valid}: A and C must have the same season count")
        oldest = min(b.train_seasons)
        if tuple(s for s in b.train_seasons if s != oldest) != c.train_seasons:
            raise AssertionError(
                f"valid={valid}: B must equal C plus the single oldest season {oldest}"
            )
        newest = max(b.train_seasons)
        if newest in a.train_seasons:
            raise AssertionError(f"valid={valid}: A must omit the newest train season {newest}")
        if newest not in c.train_seasons:
            raise AssertionError(f"valid={valid}: C must include the newest train season {newest}")
        if newest != valid - 1:
            raise AssertionError(f"valid={valid}: newest train season must be {valid - 1}")


def uniform_sample_weight(n_train: int) -> np.ndarray:
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    weights = np.ones(int(n_train), dtype=float)
    assert_uniform_sample_weight(weights)
    return weights


def assert_uniform_sample_weight(sample_weight: np.ndarray) -> None:
    if sample_weight.ndim != 1 or len(sample_weight) == 0:
        raise AssertionError("sample_weight must be a non-empty 1-D array")
    if not np.all(sample_weight == 1.0):
        uniq = sorted({float(v) for v in np.unique(sample_weight)})
        raise AssertionError(
            f"{WEIGHT_POLICY} violated: unique weights={uniq}"
        )


def assert_canonical_controls(params: dict[str, Any], seed: int) -> None:
    if int(seed) != EXPECTED_SEED:
        raise AssertionError(f"seed must remain {EXPECTED_SEED}, got {seed}")
    for key, expected in EXPECTED_CATBOOST_PARAMS.items():
        got = params.get(key)
        if got != expected:
            raise AssertionError(
                f"canonical CatBoost param mismatch: {key} expected {expected!r}, got {got!r}"
            )


def subsample_by_season(
    df: pd.DataFrame, season_col: str, rows_per_season: int, seed: int = 42
) -> pd.DataFrame:
    """Smoke-mode only. Keeps every season so A/B/C windows survive."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby(season_col, sort=True):
        if len(group) <= rows_per_season:
            parts.append(group)
            continue
        picked = np.sort(rng.choice(len(group), size=rows_per_season, replace=False))
        parts.append(group.iloc[picked])
    return pd.concat(parts, axis=0).reset_index(drop=True)


def cell_cache_key(
    *,
    spec: CellSpec,
    seed: int,
    params: dict[str, Any],
    n_features: int,
    smoke: bool,
) -> str:
    payload = json.dumps(
        {
            "audit_id": AUDIT_ID,
            "valid_season": spec.valid_season,
            "condition": spec.condition,
            "train_seasons": list(spec.train_seasons),
            "seed": int(seed),
            "params": params,
            "n_features": int(n_features),
            "weight_policy": WEIGHT_POLICY,
            "smoke": bool(smoke),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_fit_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        cache[record["cell_key"]] = record
    return cache


def append_fit_cache(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_cell(
    *,
    train_df: pd.DataFrame,
    spec: CellSpec,
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    params: dict[str, Any],
    seed: int,
    scale: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    set_seed(seed)
    fold = build_season_holdout(
        train_df,
        season_col,
        list(spec.train_seasons),
        [spec.valid_season],
        name=spec.cell_id,
    )
    train_seasons_present = sorted(
        {int(s) for s in train_df.iloc[fold.train_idx][season_col].to_numpy()}
    )
    valid_seasons_present = sorted(
        {int(s) for s in train_df.iloc[fold.valid_idx][season_col].to_numpy()}
    )
    if train_seasons_present != list(spec.train_seasons):
        raise AssertionError(
            f"{spec.cell_id}: train rows seasons {train_seasons_present} "
            f"!= locked {list(spec.train_seasons)}"
        )
    if valid_seasons_present != [spec.valid_season]:
        raise AssertionError(
            f"{spec.cell_id}: valid rows seasons {valid_seasons_present} "
            f"!= [{spec.valid_season}]"
        )
    if spec.valid_season in train_seasons_present:
        raise AssertionError(f"{spec.cell_id}: valid season leaked into training rows")

    cats = [c for c in categorical_cols if c in used]
    nums = [c for c in numeric_cols if c in used]
    assert_l42_contract(used, cats)
    X_tr_raw = train_df.iloc[fold.train_idx][used]
    y_tr = train_df.iloc[fold.train_idx][target_col]
    X_va_raw = train_df.iloc[fold.valid_idx][used]
    y_va = train_df.iloc[fold.valid_idx][target_col]

    X_tr, X_va = prepare_native_frames(
        X_tr_raw,
        X_va_raw,
        numeric_cols=nums,
        categorical_cols=cats,
        backend=BACKEND,
    )
    sample_weight = uniform_sample_weight(len(X_tr))
    model = fit_catboost(
        X_tr, y_tr, cats, params, seed, sample_weight=sample_weight
    )
    pred, pos_index = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    ref_brier = float(reference_brier_score(y_va))
    elapsed = float(time.perf_counter() - t0)
    return {
        "valid_season": int(spec.valid_season),
        "condition": spec.condition,
        "train_seasons": ",".join(str(s) for s in spec.train_seasons),
        "train_rows": int(len(fold.train_idx)),
        "valid_rows": int(len(fold.valid_idx)),
        "feature_count": int(len(used)),
        "seed": int(seed),
        "catboost_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "weight_policy": WEIGHT_POLICY,
        "sample_weight_unique": [1.0],
        "raw_bss": float(metrics["raw_brier_skill_score"]),
        "official_bss": float(metrics["official_brier_skill_score"]),
        "brier": float(metrics["brier_score"]),
        "reference_brier": ref_brier,
        "target_rate": float(metrics["target_rate"]),
        "prediction_mean": float(np.mean(pred)),
        "positive_class_index": int(pos_index),
        "fit_seconds": elapsed,
    }


def _delta_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "worst": float(np.min(arr)),
        "best": float(np.max(arr)),
    }


def compute_contrasts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_valid: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_valid.setdefault(int(row["valid_season"]), {})[row["condition"]] = row
    missing = []
    for valid in VALID_SEASONS:
        for condition in CONDITIONS:
            if condition not in by_valid.get(valid, {}):
                missing.append(f"{valid}/{condition}")
    if missing:
        raise RuntimeError(
            "refusing to aggregate incomplete results; missing fits: " + ", ".join(missing)
        )

    fold_rows: list[dict[str, Any]] = []
    b_minus_a: list[float] = []
    c_minus_a: list[float] = []
    b_minus_c: list[float] = []
    b_raw: dict[int, float] = {}
    c_raw: dict[int, float] = {}
    for valid in VALID_SEASONS:
        a = float(by_valid[valid]["A_gap"]["raw_bss"])
        b = float(by_valid[valid]["B_expanding"]["raw_bss"])
        c = float(by_valid[valid]["C_shifted"]["raw_bss"])
        ba = b - a
        ca = c - a
        bc = b - c
        residual = ba - (ca + bc)
        if abs(residual) > CONTRAST_IDENTITY_ATOL:
            raise AssertionError(
                f"contrast identity failed for valid={valid}: "
                f"B-A={ba} vs (C-A)+(B-C)={ca + bc} residual={residual}"
            )
        b_raw[valid] = b
        c_raw[valid] = c
        b_minus_a.append(ba)
        c_minus_a.append(ca)
        b_minus_c.append(bc)
        fold_rows.append(
            {
                "valid_season": valid,
                "A_raw_bss": a,
                "B_raw_bss": b,
                "C_raw_bss": c,
                "B_minus_A": ba,
                "C_minus_A": ca,
                "B_minus_C": bc,
                "identity_residual": residual,
                "B_minus_C_sign": (
                    "positive" if bc > 0 else "negative" if bc < 0 else "zero"
                ),
            }
        )

    aggregates = {
        "B_minus_A": _delta_stats(b_minus_a),
        "C_minus_A": _delta_stats(c_minus_a),
        "B_minus_C": _delta_stats(b_minus_c),
    }
    without_2023 = {
        "B_minus_A": _delta_stats([b_minus_a[0], b_minus_a[2]]),
        "C_minus_A": _delta_stats([c_minus_a[0], c_minus_a[2]]),
        "B_minus_C": _delta_stats([b_minus_c[0], b_minus_c[2]]),
    }
    verdict = predeclared_verdict(
        b_minus_c={y: fold_rows[i]["B_minus_C"] for i, y in enumerate(VALID_SEASONS)},
        b_raw=b_raw,
        c_raw=c_raw,
    )
    return {
        "fold_rows": fold_rows,
        "aggregates": aggregates,
        "aggregates_excluding_2023": without_2023,
        "verdict": verdict,
    }


def predeclared_verdict(
    *,
    b_minus_c: dict[int, float],
    b_raw: dict[int, float],
    c_raw: dict[int, float],
) -> dict[str, Any]:
    deltas = [float(b_minus_c[y]) for y in VALID_SEASONS]
    mean_d = float(np.mean(deltas))
    n_neg = int(sum(d < 0 for d in deltas))
    n_pos = int(sum(d > 0 for d in deltas))
    b_worst = float(min(b_raw[y] for y in VALID_SEASONS))
    c_worst = float(min(c_raw[y] for y in VALID_SEASONS))
    signs = {
        str(y): ("positive" if b_minus_c[y] > 0 else "negative" if b_minus_c[y] < 0 else "zero")
        for y in VALID_SEASONS
    }
    shorten_checks = {
        "mean_B_minus_C_lt_0": mean_d < 0,
        "at_least_two_folds_negative": n_neg >= 2,
        "C_worst_not_below_B_worst": c_worst >= b_worst,
    }
    keep_checks = {
        "mean_B_minus_C_gt_0": mean_d > 0,
        "at_least_two_folds_positive": n_pos >= 2,
        "B_worst_not_below_C_worst": b_worst >= c_worst,
    }
    shorten = all(shorten_checks.values())
    keep = all(keep_checks.values())
    if shorten and keep:
        raise AssertionError("shorten and keep rules cannot both fire")
    if shorten:
        bucket = VERDICT_SHORTEN
    elif keep:
        bucket = VERDICT_KEEP
    else:
        bucket = VERDICT_HOLD
    return {
        "bucket": bucket,
        "mean_B_minus_C": mean_d,
        "n_negative_folds": n_neg,
        "n_positive_folds": n_pos,
        "B_worst_fold_raw_bss": b_worst,
        "C_worst_fold_raw_bss": c_worst,
        "fold_signs_B_minus_C": signs,
        "shorten_checks": shorten_checks,
        "keep_checks": keep_checks,
        "note": (
            "training-window policy candidate under uniform weights; "
            "not a canonical model ADOPT/REJECT"
        ),
    }


def contrasts_table(contrasts: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "valid_season": str(row["valid_season"]),
            "B_minus_A": row["B_minus_A"],
            "C_minus_A": row["C_minus_A"],
            "B_minus_C": row["B_minus_C"],
        }
        for row in contrasts["fold_rows"]
    ]
    for stat_name in ("mean", "std", "worst", "best"):
        rows.append(
            {
                "valid_season": stat_name,
                "B_minus_A": contrasts["aggregates"]["B_minus_A"][stat_name],
                "C_minus_A": contrasts["aggregates"]["C_minus_A"][stat_name],
                "B_minus_C": contrasts["aggregates"]["B_minus_C"][stat_name],
            }
        )
    return pd.DataFrame(rows)


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build_markdown(
    *,
    setup: dict[str, Any],
    rows: list[dict[str, Any]],
    contrasts: dict[str, Any],
    commands: list[str],
    smoke: bool,
) -> str:
    verdict = contrasts["verdict"]
    by_valid: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_valid.setdefault(int(row["valid_season"]), {})[row["condition"]] = row
    fold_map = {int(r["valid_season"]): r for r in contrasts["fold_rows"]}
    agg = contrasts["aggregates"]
    agg_no_2023 = contrasts["aggregates_excluding_2023"]
    d2023 = fold_map[2023]["B_minus_C"]
    same_as_others = (
        fold_map[2022]["B_minus_C_sign"] == fold_map[2024]["B_minus_C_sign"]
        and fold_map[2023]["B_minus_C_sign"] == fold_map[2022]["B_minus_C_sign"]
    )
    lines = [
        f"# {AUDIT_ID}",
        "",
        f"- generated_at: `{setup['generated_at']}`",
        f"- smoke: `{str(smoke).lower()}`",
        f"- seed: `{setup['seed']}`",
        f"- weight_policy: `{WEIGHT_POLICY}`",
        f"- feature_set: L42 (`{setup['n_features']}` features; drop `{list(ID_DROP)}`; keep `game_type`)",
        f"- CatBoost params: `{json.dumps(setup['catboost_params'], ensure_ascii=False)}`",
        f"- decision: `results_only_no_adopt_reject` (window-policy candidate only)",
        "",
        "## Objective",
        "",
        "오래된 시즌 데이터를 expanding window에 계속 포함하는 것이 다음 시즌 일반화에",
        "도움이 되는지, 아니면 방해가 되는지 **uniform sample weight (1.0)** 조건에서 측정한다.",
        "",
        "## Experimental contract",
        "",
        "- 3 valid seasons × 3 windows = 9 CatBoost L42 fits",
        "- 바뀌는 것: training season window, 그리고 모든 학습 행 weight = 1.0",
        "- 바뀌지 않는 것: L42 feature spec, CatBoost hyperparameters, seed 42, metric, preprocessing",
        "- Public LB / 신규 submission 없음",
        "- WEIGHT-LAG-001은 이번 작업 범위 밖",
        "",
        "## Canonical L42 controls",
        "",
        f"- backend: `{BACKEND}`",
        f"- params lock: `{json.dumps(EXPECTED_CATBOOST_PARAMS, ensure_ascii=False)}`",
        f"- seed lock: `{EXPECTED_SEED}`",
        f"- dropped IDs: `{list(ID_DROP)}`",
        "- `game_type` kept as categorical",
        "- missing numeric: fold-local median (`prepare_native_frames`)",
        "- categorical: CatBoost native string/`__NA__`",
        "- metric: `src.evaluate.evaluate_predictions` → `src.metrics.compute_brier_metrics`",
        "- comparison metric: **raw BSS** (official clipped BSS is recorded only)",
        "",
        "## Uniform-weight scope",
        "",
        "> 이 결과는 uniform weighting 조건의 training-window 비교이며,",
        "> 현재 weighted canonical L42의 직접 ablation으로 과대해석하지 않는다.",
        "",
        "기존 `SEASON_WEIGHTS = {2019: 1.0, 2020: 1.2, 2021: 1.4, 2022: 1.6, 2023: 2.0}` 는",
        "이번 9 fit에 사용하지 않는다. 구현 자체는 삭제하지 않았다.",
        "",
        "## Fold definitions",
        "",
        "| valid | A_gap | B_expanding | C_shifted |",
        "|---|---|---|---|",
    ]
    for valid in VALID_SEASONS:
        windows = LOCKED_WINDOWS[valid]
        lines.append(
            f"| {valid} | {windows['A_gap']} | {windows['B_expanding']} | {windows['C_shifted']} |"
        )
    lines.extend(
        [
            "",
            "## Fit-level results",
            "",
            "| valid | condition | train seasons | train rows | valid rows | raw BSS | Brier | ref Brier | fit_s |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for valid in VALID_SEASONS:
        for condition in CONDITIONS:
            row = by_valid[valid][condition]
            lines.append(
                f"| {valid} | {condition} | `{row['train_seasons']}` | {row['train_rows']} | "
                f"{row['valid_rows']} | {_fmt(row['raw_bss'])} | {_fmt(row['brier'], 8)} | "
                f"{_fmt(row['reference_brier'], 8)} | {_fmt(row['fit_seconds'], 1)} |"
            )
    lines.extend(
        [
            "",
            "### Compact raw BSS",
            "",
            "| valid | A_gap | B_expanding | C_shifted |",
            "|---|---|---|---|",
        ]
    )
    for valid in VALID_SEASONS:
        a = by_valid[valid]["A_gap"]["raw_bss"]
        b = by_valid[valid]["B_expanding"]["raw_bss"]
        c = by_valid[valid]["C_shifted"]["raw_bss"]
        lines.append(f"| {valid} | {_fmt(a)} | {_fmt(b)} | {_fmt(c)} |")
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "정의:",
            "",
            "- `B-A` = expanding_history_effect (가장 최근 train season 하나를 기존 history에 추가)",
            "- `C-A` = window_shift_effect (같은 시즌 수에서 window를 한 해 앞으로 이동).",
            "  **순수 recency 효과라고 부르지 않는다.**",
            "- `B-C` = old_history_inclusion_effect (최신 window에 가장 오래된 시즌 하나를 추가). **PRIMARY**",
            "",
            "| valid | B-A | C-A | B-C | B-C sign |",
            "|---|---|---|---|---|",
        ]
    )
    for valid in VALID_SEASONS:
        row = fold_map[valid]
        lines.append(
            f"| {valid} | {_fmt(row['B_minus_A'])} | {_fmt(row['C_minus_A'])} | "
            f"{_fmt(row['B_minus_C'])} | {row['B_minus_C_sign']} |"
        )
    lines.append(
        f"| mean | {_fmt(agg['B_minus_A']['mean'])} | {_fmt(agg['C_minus_A']['mean'])} | "
        f"{_fmt(agg['B_minus_C']['mean'])} | |"
    )
    lines.append(
        f"| std (ddof=1) | {_fmt(agg['B_minus_A']['std'])} | {_fmt(agg['C_minus_A']['std'])} | "
        f"{_fmt(agg['B_minus_C']['std'])} | |"
    )
    lines.append(
        f"| worst (min) | {_fmt(agg['B_minus_A']['worst'])} | {_fmt(agg['C_minus_A']['worst'])} | "
        f"{_fmt(agg['B_minus_C']['worst'])} | |"
    )
    lines.append(
        f"| best (max) | {_fmt(agg['B_minus_A']['best'])} | {_fmt(agg['C_minus_A']['best'])} | "
        f"{_fmt(agg['B_minus_C']['best'])} | |"
    )
    lines.extend(
        [
            "",
            "항등식 `B-A = (C-A) + (B-C)` 는 세 fold 모두 floating-point tolerance "
            f"`{CONTRAST_IDENTITY_ATOL}` 안에서 성립했다.",
            "",
            "## Primary contrast",
            "",
            "`old_history_inclusion_effect = B - C`",
            "",
            "- `D > 0`: 오래된 시즌 포함이 유리",
            "- `D < 0`: 오래된 시즌 제외가 유리",
            "",
            f"- 2022: `{_fmt(fold_map[2022]['B_minus_C'])}` ({fold_map[2022]['B_minus_C_sign']})",
            f"- 2023: `{_fmt(fold_map[2023]['B_minus_C'])}` ({fold_map[2023]['B_minus_C_sign']})",
            f"- 2024: `{_fmt(fold_map[2024]['B_minus_C'])}` ({fold_map[2024]['B_minus_C_sign']})",
            f"- mean: `{_fmt(agg['B_minus_C']['mean'])}`",
            "",
            "## Fold stability",
            "",
            f"- B-C mean: `{_fmt(agg['B_minus_C']['mean'])}`",
            f"- B-C std: `{_fmt(agg['B_minus_C']['std'])}`",
            f"- B-C worst: `{_fmt(agg['B_minus_C']['worst'])}`",
            f"- B-C best: `{_fmt(agg['B_minus_C']['best'])}`",
            "",
            "### 2023 behavior",
            "",
            f"- 2023 B-C: `{_fmt(d2023)}` ({fold_map[2023]['B_minus_C_sign']})",
            f"- 2022/2024와 부호가 같은가: `{str(same_as_others).lower()}` "
            f"(2022={fold_map[2022]['B_minus_C_sign']}, "
            f"2024={fold_map[2024]['B_minus_C_sign']})",
            f"- B-C mean including 2023: `{_fmt(agg['B_minus_C']['mean'])}`",
            f"- B-C mean excluding 2023: `{_fmt(agg_no_2023['B_minus_C']['mean'])}`",
            f"- mean shift from including 2023: "
            f"`{_fmt(agg['B_minus_C']['mean'] - agg_no_2023['B_minus_C']['mean'])}`",
            f"- B-C std including 2023: `{_fmt(agg['B_minus_C']['std'])}`",
            f"- B-C std excluding 2023: `{_fmt(agg_no_2023['B_minus_C']['std'])}`",
            "",
            "2023 fold는 제외하지 않았다. 위 숫자는 해석용 보조 통계다.",
            "",
            "## Predeclared-rule result",
            "",
            f"**`{verdict['bucket']}`**",
            "",
            "이 판정은 최종 canonical 모델 ADOPT/REJECT가 아니다.",
            "uniform-weight 조건에서의 training-window 정책 후보 판정이다.",
            "",
            "### SHORTEN_WINDOW_CANDIDATE checks",
            "",
        ]
    )
    for key, ok in verdict["shorten_checks"].items():
        lines.append(f"- `{key}`: `{ok}`")
    lines.extend(["", "### KEEP_FULL_HISTORY_CANDIDATE checks", ""])
    for key, ok in verdict["keep_checks"].items():
        lines.append(f"- `{key}`: `{ok}`")
    lines.extend(
        [
            "",
            f"- mean(B-C): `{_fmt(verdict['mean_B_minus_C'])}`",
            f"- negative folds: `{verdict['n_negative_folds']}/3`",
            f"- positive folds: `{verdict['n_positive_folds']}/3`",
            f"- B worst-fold raw BSS: `{_fmt(verdict['B_worst_fold_raw_bss'])}`",
            f"- C worst-fold raw BSS: `{_fmt(verdict['C_worst_fold_raw_bss'])}`",
            f"- B-C signs: `{json.dumps(verdict['fold_signs_B_minus_C'], ensure_ascii=False)}`",
            "",
        ]
    )
    if verdict["bucket"] == VERDICT_KEEP and all(
        sign == "positive" for sign in verdict["fold_signs_B_minus_C"].values()
    ):
        lines.extend(
            [
                "핵심 결론 (이 결과의 해석 상한):",
                "",
                "> Uniform-weight L42 조건에서는 오래된 시즌을 제거해 recent window로 단축할 근거가 없으며, "
                "B-C가 2022/2023/2024에서 모두 양수이므로 full-history 유지가 training-window policy candidate이다.",
                "",
                "이 문장은 canonical model ADOPT가 아니고, weighted L42 우열 비교도 아니며, "
                "2023 shift 해소나 '오래된 시즌은 항상 유익하다'는 일반화도 아니다.",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "- single seed 42",
            "- uniform weighting experiment; current weighted canonical L42와 직접 같지 않음",
            "- 2024 uniform B와 weighted canonical L42 raw BSS 차이는 이번 audit의 인과 비교가 아니다",
            "- Public LB를 사용하지 않음",
            "- CatBoost L42에서의 결과",
            "- 다른 backend / feature-set / Trackman feature에 자동 일반화 금지",
            "- 2023 regime shift를 포함한 3-fold이며, fold 효과를 평균으로만 읽으면 안 됨",
            "- `load_competition_data`가 test.csv를 같이 열지만, 이 audit은 train만 사용한다. test로 aggregate/encoding/통계를 만들지 않았다",
            "",
            "## Next step",
            "",
        ]
    )
    bucket = verdict["bucket"]
    if bucket == VERDICT_SHORTEN:
        lines.extend(
            [
                "SHORTEN이므로 `WEIGHT-LAG-001`의 window 범위를 이번 결과에 맞게 먼저 재설계한 뒤",
                "가중치 스케줄을 측정한다. WEIGHT-LAG-001은 이번 작업에서 구현하지 않았다.",
            ]
        )
    else:
        lines.extend(
            [
                "RECENCY-AUDIT-001이 main에 반영된 뒤, 최신 origin/main에서 새 브랜치를 만들어 WEIGHT-LAG-001을 시작한다.",
                "WEIGHT-LAG-001은 이번 작업에서 구현하지 않았다.",
            ]
        )
    lines.extend(
        [
            "",
            "## Commands actually run",
            "",
        ]
    )
    for cmd in commands:
        lines.append(f"- `{cmd}`")
    lines.extend(
        [
            "",
            "## experiments.csv",
            "",
            "이번 audit는 `outputs/experiments.csv`에 append하지 않았다.",
            "VAL-AUDIT-001 / MODEL-TRANSFER-AUDIT-001과 같은 진단 convention이다.",
        ]
    )
    if smoke:
        lines.extend(
            [
                "> SMOKE RUN. 이 숫자는 canonical 결과가 아니다.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{AUDIT_ID}: uniform-weight training-window audit on CatBoost L42."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", default="outputs/analysis/recency_audit_001")
    parser.add_argument("--report-path", default="reports/recency_audit_001.md")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-rows-per-season", type=int, default=800)
    parser.add_argument("--smoke-iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wall0 = time.perf_counter()
    assert_locked_window_contract()
    cells = iter_cell_specs()

    config = load_config(args.config)
    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    target_col = data_cfg["target_col"]
    season_col = data_cfg["season_col"]
    seed = int(config.get("runtime", {}).get("seed", 42))
    scale = float(resolve_metric_settings(config.get("task", {}))["scale"])
    params = dict(config.get("models", {}).get("params", {}).get("catboost", {}))
    if int(seed) != EXPECTED_SEED:
        raise AssertionError(f"seed must remain {EXPECTED_SEED}, got {seed}")
    if args.smoke:
        params["iterations"] = int(args.smoke_iterations)
    else:
        assert_canonical_controls(params, seed)

    print(f"=== {AUDIT_ID} ===", flush=True)
    print(
        f"cells={len(cells)} weight_policy={WEIGHT_POLICY} seed={seed} smoke={args.smoke}",
        flush=True,
    )
    if args.dry_run:
        for cell in cells:
            print(f"  {cell.cell_id}: train={list(cell.train_seasons)}", flush=True)
        return

    loaded = load_competition_data(config)
    train_df = loaded["train"].reset_index(drop=True)
    _ = loaded["test"]  # loaded by the shared helper; never used in this audit
    if args.smoke:
        train_df = subsample_by_season(
            train_df, season_col, args.smoke_rows_per_season, seed=seed
        )
    print(f"train rows: {len(train_df)}", flush=True)

    used, numeric_cols, categorical_cols = l42_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=list(feat_cfg.get("exclude_cols") or []),
        categorical_cols=list(feat_cfg.get("categorical_cols") or []),
    )
    assert_l42_contract(used, categorical_cols)
    print(
        f"L42 features: {len(used)} (cats={len(categorical_cols)}; drop={list(ID_DROP)})",
        flush=True,
    )

    out_dir = ensure_dir(resolve_path(args.out_dir + ("_smoke" if args.smoke else "")))
    cache_path = out_dir / "fits.jsonl"
    cache = {} if args.no_resume else load_fit_cache(cache_path)
    if cache:
        print(f"resume: {len(cache)} cached fits in {cache_path}", flush=True)

    rows: list[dict[str, Any]] = []
    n_executed = 0
    n_reused = 0
    current_cell = "before_fits"
    try:
        for spec in cells:
            current_cell = spec.cell_id
            key = cell_cache_key(
                spec=spec,
                seed=seed,
                params=params,
                n_features=len(used),
                smoke=bool(args.smoke),
            )
            cached = cache.get(key)
            if cached is not None:
                rows.append(cached["metrics"])
                n_reused += 1
                print(
                    f"  [{spec.cell_id}] reused raw_bss={cached['metrics']['raw_bss']:.6f}",
                    flush=True,
                )
                continue
            print(f"  [{spec.cell_id}] fitting...", flush=True)
            row = run_cell(
                train_df=train_df,
                spec=spec,
                used=used,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                target_col=target_col,
                season_col=season_col,
                params=params,
                seed=seed,
                scale=scale,
            )
            rows.append(row)
            n_executed += 1
            append_fit_cache(
                cache_path,
                {
                    "cell_key": key,
                    "audit_id": AUDIT_ID,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "metrics": row,
                },
            )
            print(
                f"  [{spec.cell_id}] raw_bss={row['raw_bss']:.6f} "
                f"brier={row['brier']:.8f} fit_s={row['fit_seconds']:.1f}",
                flush=True,
            )
    except Exception:
        completed = [f"{r['valid_season']}/{r['condition']}" for r in rows]
        print(
            f"FAILED at {current_cell}. completed={completed} "
            f"cache={cache_path} (resume with the same command)",
            flush=True,
        )
        raise

    contrasts = compute_contrasts(rows)
    generated_at = datetime.now().isoformat(timespec="seconds")
    command = "python -m scripts.run_recency_audit_001" + (" --smoke" if args.smoke else "")
    setup = {
        "audit_id": AUDIT_ID,
        "generated_at": generated_at,
        "config": str(args.config),
        "seed": seed,
        "catboost_params": params,
        "n_features": len(used),
        "features": used,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dropped_ids": list(ID_DROP),
        "weight_policy": WEIGHT_POLICY,
        "smoke": bool(args.smoke),
        "locked_windows": {
            str(valid): {cond: seasons for cond, seasons in windows.items()}
            for valid, windows in LOCKED_WINDOWS.items()
        },
    }
    results_df = pd.DataFrame(rows)
    results_path = out_dir / "results.csv"
    contrasts_df = contrasts_table(contrasts)
    contrasts_path = out_dir / "contrasts.csv"
    metadata = {
        "audit_id": AUDIT_ID,
        "generated_at": generated_at,
        "setup": setup,
        "n_fits_executed": n_executed,
        "n_fits_reused": n_reused,
        "failed_cell": None,
        "contrasts": contrasts,
        "elapsed_seconds_wall": float(time.perf_counter() - wall0),
        "decision": "results_only_no_adopt_reject",
        "experiments_csv": "not_appended_audit_convention",
        "command": command,
    }
    metadata_path = out_dir / "metadata.json"
    results_df.to_csv(results_path, index=False)
    contrasts_df.to_csv(contrasts_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_text = build_markdown(
        setup=setup,
        rows=rows,
        contrasts=contrasts,
        commands=[command],
        smoke=bool(args.smoke),
    )
    smoke_report = out_dir / "report.md"
    smoke_report.write_text(report_text, encoding="utf-8")
    report_path: Path | None = None
    if not args.smoke:
        report_path = resolve_path(args.report_path)
        ensure_dir(report_path.parent)
        report_path.write_text(report_text, encoding="utf-8")

    for path in (results_path, contrasts_path, metadata_path, smoke_report):
        print(f"wrote {path}", flush=True)
    if report_path is not None:
        print(f"wrote {report_path}", flush=True)
    print(
        json.dumps(
            {
                "audit_id": AUDIT_ID,
                "verdict": contrasts["verdict"]["bucket"],
                "mean_B_minus_C": contrasts["verdict"]["mean_B_minus_C"],
                "n_executed": n_executed,
                "n_reused": n_reused,
                "elapsed_seconds_wall": metadata["elapsed_seconds_wall"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
