"""WEIGHT-LAG-001: validation-relative season-lag sample weights on CatBoost L42.

Question: given the locked full-history training windows from RECENCY-AUDIT-001,
does up-weighting more recent training seasons improve next-season raw BSS?

The only experimental factor is the geometric lag multiplier m. Features, model,
hyperparameters, seed, metric, and preprocessing stay locked.

Does not write submissions. Does not append to outputs/experiments.csv.
Does not overwrite RECENCY-AUDIT-001 artifacts.
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
from scripts.run_l42_2019_2023_lb_probe import (
    assert_l42_contract,
    feature_list_hash,
    l42_feature_lists,
)
from scripts.run_recency_audit_001 import (
    EXPECTED_CATBOOST_PARAMS,
    EXPECTED_SEED,
    LOCKED_WINDOWS,
    VALID_SEASONS,
    append_fit_cache,
    assert_canonical_controls,
    load_fit_cache,
    subsample_by_season,
)
from scripts.tree_exp_common import positive_class_proba, prepare_native_frames
from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.metrics import reference_brier_score
from src.utils import ensure_dir, load_config, resolve_path, set_seed

EXPERIMENT_ID = "WEIGHT-LAG-001"
DEFAULT_CONFIG = "configs/baseline.yaml"
BACKEND = "catboost"
WEIGHT_POLICY = "season_lag_geometric"
LAG_DEFINITION = "lag = valid_season - train_season; w(lag) = m ** (1 - (lag - 1) / (L - 1)); L = max lag in fold; L==1 -> w=m"
MULTIPLIERS = (1.0, 1.5, 2.0, 3.0, 4.0)
CONTROL_M = 1.0
EXPECTED_FIT_COUNT = 15
CONTRAST_ATOL = 1e-9
REPRODUCTION_ATOL = 1e-9
REPRODUCTION_BRIER_ATOL = 1e-12
DEFAULT_RECENCY_RESULTS = "outputs/analysis/recency_audit_001/results.csv"
PROTECTED_PATHS = (
    "outputs/analysis/recency_audit_001",
    "outputs/analysis/recency_audit_001_smoke",
    "outputs/analysis/val_shift_2023_001",
    "outputs/analysis/val_shift_2023_001_smoke",
    "reports/recency_audit_001.md",
    "outputs/experiments.csv",
)
FOLD_NAMES = {
    2022: "secondary_2022",
    2023: "secondary_2023",
    2024: "primary_2024",
}
FULL_HISTORY_WINDOWS: dict[int, tuple[int, ...]] = {
    2022: (2019, 2020, 2021),
    2023: (2019, 2020, 2021, 2022),
    2024: (2019, 2020, 2021, 2022, 2023),
}
LOCKED_RECENCY_B_EXPANDING_RAW_BSS = {
    2022: 2322.4597083638996,
    2023: -1261.5399758588496,
    2024: 734.8190445647051,
}
LOCKED_RECENCY_B_EXPANDING_ROWS = {
    2022: (728588, 247472),
    2023: (976060, 245525),
    2024: (1221585, 253507),
}


@dataclass(frozen=True)
class CellSpec:
    valid_season: int
    multiplier: float
    train_seasons: tuple[int, ...]

    @property
    def cell_id(self) -> str:
        return f"{self.valid_season}__m{self.multiplier:.1f}"

    @property
    def fold_name(self) -> str:
        return FOLD_NAMES[self.valid_season]

    @property
    def is_control(self) -> bool:
        return float(self.multiplier) == CONTROL_M


def assert_full_history_matches_recency() -> None:
    for valid, seasons in FULL_HISTORY_WINDOWS.items():
        locked = LOCKED_WINDOWS[valid]["B_expanding"]
        if list(seasons) != list(locked):
            raise AssertionError(
                f"full-history window for {valid} drifted from RECENCY B_expanding: "
                f"{list(seasons)} vs {locked}"
            )
        if any(s >= valid for s in seasons):
            raise AssertionError(f"valid={valid}: future/valid season in train {seasons}")
        if max(seasons) != valid - 1:
            raise AssertionError(
                f"valid={valid}: newest train season must be {valid - 1}, got {max(seasons)}"
            )


def iter_cell_specs(multipliers: tuple[float, ...] = MULTIPLIERS) -> list[CellSpec]:
    assert_full_history_matches_recency()
    cells: list[CellSpec] = []
    for multiplier in multipliers:
        for valid in VALID_SEASONS:
            cells.append(
                CellSpec(
                    valid_season=int(valid),
                    multiplier=float(multiplier),
                    train_seasons=FULL_HISTORY_WINDOWS[int(valid)],
                )
            )
    expected = len(multipliers) * len(VALID_SEASONS)
    if len(cells) != expected:
        raise AssertionError(f"expected {expected} cells, got {len(cells)}")
    return cells


def season_lag(train_season: int, valid_season: int) -> int:
    lag = int(valid_season) - int(train_season)
    if lag < 1:
        raise AssertionError(
            f"train_season {train_season} is not strictly before valid_season {valid_season}"
        )
    return lag


def fold_max_lag(train_seasons: list[int] | tuple[int, ...], valid_season: int) -> int:
    if not train_seasons:
        raise ValueError("train_seasons must be non-empty")
    return max(season_lag(int(s), int(valid_season)) for s in train_seasons)


def lag_weight(lag: int, max_lag: int, multiplier: float) -> float:
    """Validation-relative geometric weight. Independent of calendar year labels."""
    if int(lag) < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if int(max_lag) < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")
    if int(lag) > int(max_lag):
        raise ValueError(f"lag {lag} exceeds max_lag {max_lag}")
    m = float(multiplier)
    if not np.isfinite(m) or m <= 0:
        raise ValueError(f"multiplier must be a positive finite number, got {multiplier}")
    if int(max_lag) == 1:
        return m
    exponent = 1.0 - (int(lag) - 1) / (int(max_lag) - 1)
    return float(m ** exponent)


def season_weight_map(
    train_seasons: list[int] | tuple[int, ...],
    valid_season: int,
    multiplier: float,
) -> dict[int, float]:
    seasons = tuple(int(s) for s in train_seasons)
    if not seasons:
        raise ValueError("train_seasons must be non-empty")
    if any(s >= int(valid_season) for s in seasons):
        raise AssertionError(
            f"train seasons {seasons} are not strictly before valid_season {valid_season}"
        )
    L = fold_max_lag(seasons, valid_season)
    mapping = {
        int(s): lag_weight(season_lag(int(s), valid_season), L, multiplier)
        for s in seasons
    }
    assert_weight_endpoints(mapping, seasons, valid_season, multiplier)
    return mapping


def sample_weights_for_seasons(
    seasons: np.ndarray | pd.Series,
    valid_season: int,
    multiplier: float,
) -> tuple[np.ndarray, dict[int, float]]:
    season_arr = np.asarray(seasons)
    if season_arr.ndim != 1 or len(season_arr) == 0:
        raise AssertionError("seasons must be a non-empty 1-D array")
    present = tuple(sorted({int(s) for s in season_arr}))
    mapping = season_weight_map(present, valid_season, multiplier)
    weights = np.asarray([mapping[int(s)] for s in season_arr], dtype=float)
    if len(weights) != len(season_arr):
        raise AssertionError("sample_weight length drifted from training rows")
    if float(multiplier) == CONTROL_M and not np.all(weights == 1.0):
        raise AssertionError(f"m=1.0 must yield all weights 1.0; unique={np.unique(weights)}")
    return weights, mapping


def assert_weight_endpoints(
    mapping: dict[int, float],
    train_seasons: list[int] | tuple[int, ...],
    valid_season: int,
    multiplier: float,
) -> None:
    seasons = [int(s) for s in train_seasons]
    newest = max(seasons)
    oldest = min(seasons)
    m = float(multiplier)
    if abs(mapping[newest] - m) > CONTRAST_ATOL:
        raise AssertionError(
            f"most recent season {newest} weight {mapping[newest]} != m={m}"
        )
    expected_oldest = m if (newest == oldest) else 1.0
    if abs(mapping[oldest] - expected_oldest) > CONTRAST_ATOL:
        raise AssertionError(
            f"oldest season {oldest} weight {mapping[oldest]} != {expected_oldest}"
        )
    ordered = sorted(seasons, reverse=True)
    weights_newest_to_oldest = [mapping[s] for s in ordered]
    if m > 1.0:
        for earlier, later in zip(weights_newest_to_oldest, weights_newest_to_oldest[1:]):
            if later > earlier + CONTRAST_ATOL:
                raise AssertionError(
                    f"weights must be monotonically non-increasing toward older seasons: "
                    f"{weights_newest_to_oldest}"
                )
    if m == CONTROL_M:
        if any(abs(v - 1.0) > CONTRAST_ATOL for v in mapping.values()):
            raise AssertionError(f"m=1.0 mapping must be all 1.0, got {mapping}")


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
            "experiment_id": EXPERIMENT_ID,
            "valid_season": spec.valid_season,
            "multiplier": spec.multiplier,
            "train_seasons": list(spec.train_seasons),
            "seed": int(seed),
            "params": params,
            "n_features": int(n_features),
            "weight_policy": WEIGHT_POLICY,
            "lag_definition": LAG_DEFINITION,
            "smoke": bool(smoke),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assert_protected_paths(out_dir: Path, report_path: Path | None) -> None:
    resolved = [str(out_dir.resolve())]
    if report_path is not None:
        resolved.append(str(report_path.resolve()))
    for raw in PROTECTED_PATHS:
        protected = str((ROOT / raw).resolve())
        for candidate in resolved:
            if candidate == protected or candidate.startswith(protected + "\\") or candidate.startswith(protected + "/"):
                raise AssertionError(
                    f"{EXPERIMENT_ID} refused to write into protected path {raw}"
                )


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
    feature_fp: str,
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
    if max(train_seasons_present) >= spec.valid_season:
        raise AssertionError(
            f"{spec.cell_id}: max(train season)={max(train_seasons_present)} "
            f">= valid {spec.valid_season}"
        )

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
    sample_weight, weight_map = sample_weights_for_seasons(
        train_df.iloc[fold.train_idx][season_col].to_numpy(),
        spec.valid_season,
        spec.multiplier,
    )
    if len(sample_weight) != len(X_tr):
        raise AssertionError(f"{spec.cell_id}: sample_weight length != n_train")
    model = fit_catboost(
        X_tr, y_tr, cats, params, seed, sample_weight=sample_weight
    )
    pred, pos_index = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    ref_brier = float(reference_brier_score(y_va))
    elapsed = float(time.perf_counter() - t0)
    unique_weights = sorted({float(v) for v in np.unique(sample_weight)})
    return {
        "experiment_id": EXPERIMENT_ID,
        "cell_id": spec.cell_id,
        "fold_name": spec.fold_name,
        "valid_season": int(spec.valid_season),
        "train_seasons": ",".join(str(s) for s in spec.train_seasons),
        "multiplier_m": float(spec.multiplier),
        "lag_definition": LAG_DEFINITION,
        "max_lag": int(fold_max_lag(spec.train_seasons, spec.valid_season)),
        "season_to_weight": json.dumps(
            {str(k): float(v) for k, v in sorted(weight_map.items())},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "train_rows": int(len(fold.train_idx)),
        "valid_rows": int(len(fold.valid_idx)),
        "feature_count": int(len(used)),
        "feature_fingerprint": feature_fp,
        "seed": int(seed),
        "model": BACKEND,
        "catboost_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "weight_policy": WEIGHT_POLICY,
        "sample_weight_unique": json.dumps(unique_weights, ensure_ascii=False),
        "raw_bss": float(metrics["raw_brier_skill_score"]),
        "official_bss": float(metrics["official_brier_skill_score"]),
        "brier": float(metrics["brier_score"]),
        "reference_brier": ref_brier,
        "target_rate": float(metrics["target_rate"]),
        "prediction_mean": float(np.mean(pred)),
        "positive_class_index": int(pos_index),
        "fit_seconds": elapsed,
        "status": "ok",
    }


def _delta_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_contrasts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[tuple[int, float], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["valid_season"]), float(row["multiplier_m"]))
        if key in by_cell:
            raise AssertionError(f"duplicate cell {key}")
        by_cell[key] = row
    missing = [
        f"{valid}/m{m:.1f}"
        for m in MULTIPLIERS
        for valid in VALID_SEASONS
        if (valid, float(m)) not in by_cell
    ]
    if missing:
        raise RuntimeError(
            "refusing to aggregate incomplete results; missing fits: " + ", ".join(missing)
        )
    if len(by_cell) != EXPECTED_FIT_COUNT:
        raise AssertionError(f"expected {EXPECTED_FIT_COUNT} cells, got {len(by_cell)}")

    control = {valid: by_cell[(valid, CONTROL_M)] for valid in VALID_SEASONS}
    fold_rows: list[dict[str, Any]] = []
    for m in MULTIPLIERS:
        raws = [float(by_cell[(valid, float(m))]["raw_bss"]) for valid in VALID_SEASONS]
        deltas = [
            float(by_cell[(valid, float(m))]["raw_bss"]) - float(control[valid]["raw_bss"])
            for valid in VALID_SEASONS
        ]
        for valid, raw, delta in zip(VALID_SEASONS, raws, deltas):
            reconstructed = (
                float(by_cell[(valid, float(m))]["raw_bss"])
                - float(control[valid]["raw_bss"])
            )
            residual = delta - reconstructed
            if abs(residual) > CONTRAST_ATOL:
                raise AssertionError(
                    f"delta identity failed valid={valid} m={m}: residual={residual}"
                )
        fold_rows.append(
            {
                "multiplier_m": float(m),
                "raw_bss_2022": raws[0],
                "raw_bss_2023": raws[1],
                "raw_bss_2024": raws[2],
                "mean_raw_bss": _delta_stats(raws)["mean"],
                "std_raw_bss": _delta_stats(raws)["std"],
                "min_raw_bss": _delta_stats(raws)["min"],
                "delta_2022": deltas[0],
                "delta_2023": deltas[1],
                "delta_2024": deltas[2],
                "mean_delta": _delta_stats(deltas)["mean"],
                "std_delta": _delta_stats(deltas)["std"],
                "min_delta": _delta_stats(deltas)["min"],
                "delta_identity_max_abs_residual": 0.0,
            }
        )
        for valid, delta in zip(VALID_SEASONS, deltas):
            treatment = float(by_cell[(valid, float(m))]["raw_bss"])
            expected = treatment - float(control[valid]["raw_bss"])
            residual = delta - expected
            fold_rows[-1]["delta_identity_max_abs_residual"] = max(
                float(fold_rows[-1]["delta_identity_max_abs_residual"]),
                abs(residual),
            )
    return {"fold_rows": fold_rows, "control_raw_bss": {
        str(valid): float(control[valid]["raw_bss"]) for valid in VALID_SEASONS
    }}


def contrasts_table(contrasts: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(contrasts["fold_rows"])


def load_recency_b_expanding(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"RECENCY-AUDIT-001 reference missing: {path}. "
            "WEIGHT-LAG-001 cannot run the m=1 reproduction gate."
        )
    frame = pd.read_csv(path)
    needed = frame[frame["condition"] == "B_expanding"].copy()
    if len(needed) != 3:
        raise AssertionError(
            f"expected 3 B_expanding rows in {path}, got {len(needed)}"
        )
    out: dict[int, dict[str, Any]] = {}
    for _, row in needed.iterrows():
        valid = int(row["valid_season"])
        out[valid] = {
            "raw_bss": float(row["raw_bss"]),
            "brier": float(row["brier"]),
            "train_rows": int(row["train_rows"]),
            "valid_rows": int(row["valid_rows"]),
            "feature_count": int(row["feature_count"]),
            "seed": int(row["seed"]),
            "train_seasons": str(row["train_seasons"]),
        }
    for valid in VALID_SEASONS:
        if valid not in out:
            raise AssertionError(f"RECENCY B_expanding missing valid={valid}")
        locked_bss = LOCKED_RECENCY_B_EXPANDING_RAW_BSS[valid]
        if abs(out[valid]["raw_bss"] - locked_bss) > REPRODUCTION_ATOL:
            raise AssertionError(
                f"RECENCY artifact drifted for valid={valid}: "
                f"{out[valid]['raw_bss']} vs locked {locked_bss}"
            )
        locked_rows = LOCKED_RECENCY_B_EXPANDING_ROWS[valid]
        if (out[valid]["train_rows"], out[valid]["valid_rows"]) != locked_rows:
            raise AssertionError(
                f"RECENCY row counts drifted for valid={valid}: "
                f"{out[valid]['train_rows']}/{out[valid]['valid_rows']} vs {locked_rows}"
            )
        expected_seasons = ",".join(str(s) for s in FULL_HISTORY_WINDOWS[valid])
        if out[valid]["train_seasons"] != expected_seasons:
            raise AssertionError(
                f"RECENCY train_seasons drifted for valid={valid}: "
                f"{out[valid]['train_seasons']} vs {expected_seasons}"
            )
    return out


def reproduction_gate(
    control_rows: list[dict[str, Any]],
    recency: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if len(control_rows) != len(VALID_SEASONS):
        raise AssertionError(
            f"reproduction gate needs {len(VALID_SEASONS)} control fits, got {len(control_rows)}"
        )
    by_valid = {int(r["valid_season"]): r for r in control_rows}
    fold_reports: dict[str, Any] = {}
    failures: list[str] = []
    for valid in VALID_SEASONS:
        got = by_valid[valid]
        exp = recency[valid]
        raw_diff = float(got["raw_bss"]) - float(exp["raw_bss"])
        brier_diff = float(got["brier"]) - float(exp["brier"])
        checks = {
            "train_rows": int(got["train_rows"]) == int(exp["train_rows"]),
            "valid_rows": int(got["valid_rows"]) == int(exp["valid_rows"]),
            "feature_count": int(got["feature_count"]) == int(exp["feature_count"]),
            "seed": int(got["seed"]) == int(exp["seed"]),
            "train_seasons": str(got["train_seasons"]) == str(exp["train_seasons"]),
            "raw_bss": abs(raw_diff) <= REPRODUCTION_ATOL,
            "brier": abs(brier_diff) <= REPRODUCTION_BRIER_ATOL,
            "weights_all_one": json.loads(got["sample_weight_unique"]) == [1.0],
        }
        if not all(checks.values()):
            failures.append(
                f"valid={valid} checks={json.dumps(checks, ensure_ascii=False)} "
                f"raw_diff={raw_diff} brier_diff={brier_diff}"
            )
        fold_reports[str(valid)] = {
            "recency_raw_bss": float(exp["raw_bss"]),
            "control_raw_bss": float(got["raw_bss"]),
            "raw_bss_diff": raw_diff,
            "recency_brier": float(exp["brier"]),
            "control_brier": float(got["brier"]),
            "brier_diff": brier_diff,
            "recency_train_rows": int(exp["train_rows"]),
            "control_train_rows": int(got["train_rows"]),
            "recency_valid_rows": int(exp["valid_rows"]),
            "control_valid_rows": int(got["valid_rows"]),
            "checks": checks,
        }
    status = "PASS" if not failures else "FAIL"
    if failures:
        raise AssertionError(
            "WEIGHT-LAG-001 INVALID: m=1.0 CONTROL did not reproduce "
            "RECENCY-AUDIT-001 B_expanding uniform control. "
            + " | ".join(failures)
        )
    return {
        "status": status,
        "atol_raw_bss": REPRODUCTION_ATOL,
        "atol_brier": REPRODUCTION_BRIER_ATOL,
        "folds": fold_reports,
    }


def assert_rowcount_and_feature_lock(rows: list[dict[str, Any]]) -> None:
    by_valid: dict[int, list[dict[str, Any]]] = {}
    fingerprints: set[str] = set()
    params: set[str] = set()
    seeds: set[int] = set()
    for row in rows:
        by_valid.setdefault(int(row["valid_season"]), []).append(row)
        fingerprints.add(str(row["feature_fingerprint"]))
        params.add(str(row["catboost_params"]))
        seeds.add(int(row["seed"]))
        if int(row["feature_count"]) != 42:
            raise AssertionError(f"feature_count lock failed: {row['feature_count']}")
    if len(fingerprints) != 1:
        raise AssertionError(f"feature fingerprint changed across cells: {fingerprints}")
    if len(params) != 1:
        raise AssertionError(f"CatBoost params changed across cells: {params}")
    if seeds != {EXPECTED_SEED}:
        raise AssertionError(f"seed lock failed: {seeds}")
    for valid, group in by_valid.items():
        train_rows = {int(r["train_rows"]) for r in group}
        valid_rows = {int(r["valid_rows"]) for r in group}
        if len(train_rows) != 1 or len(valid_rows) != 1:
            raise AssertionError(
                f"row-count lock failed for valid={valid}: "
                f"train={train_rows} valid={valid_rows}"
            )


def assert_artifact_integrity(
    rows: list[dict[str, Any]],
    contrasts: dict[str, Any],
    results_df: pd.DataFrame,
    contrasts_df: pd.DataFrame,
) -> None:
    if len(rows) != EXPECTED_FIT_COUNT:
        raise AssertionError(f"fit count {len(rows)} != {EXPECTED_FIT_COUNT}")
    if len(results_df) != EXPECTED_FIT_COUNT:
        raise AssertionError("results.csv row count mismatch")
    cells = {(int(r["valid_season"]), float(r["multiplier_m"])) for r in rows}
    expected = {(int(y), float(m)) for y in VALID_SEASONS for m in MULTIPLIERS}
    if cells != expected:
        raise AssertionError(f"cell set mismatch: extra={cells-expected} missing={expected-cells}")
    assert_rowcount_and_feature_lock(rows)
    by_cell = {(int(r["valid_season"]), float(r["multiplier_m"])): r for r in rows}
    for table_row in contrasts["fold_rows"]:
        m = float(table_row["multiplier_m"])
        for valid, key in ((2022, "raw_bss_2022"), (2023, "raw_bss_2023"), (2024, "raw_bss_2024")):
            if abs(float(table_row[key]) - float(by_cell[(valid, m)]["raw_bss"])) > CONTRAST_ATOL:
                raise AssertionError(f"contrasts {key} mismatch for m={m}")
        for valid, dkey in ((2022, "delta_2022"), (2023, "delta_2023"), (2024, "delta_2024")):
            expected_delta = (
                float(by_cell[(valid, m)]["raw_bss"])
                - float(by_cell[(valid, CONTROL_M)]["raw_bss"])
            )
            residual = float(table_row[dkey]) - expected_delta
            if abs(residual) > CONTRAST_ATOL:
                raise AssertionError(
                    f"delta residual for m={m} valid={valid}: {residual}"
                )
    if len(contrasts_df) != len(MULTIPLIERS):
        raise AssertionError("contrasts.csv multiplier count mismatch")


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build_markdown(
    *,
    setup: dict[str, Any],
    rows: list[dict[str, Any]],
    contrasts: dict[str, Any],
    gate: dict[str, Any] | None,
    commands: list[str],
    smoke: bool,
) -> str:
    by_cell = {(int(r["valid_season"]), float(r["multiplier_m"])): r for r in rows}
    contrast_map = {float(r["multiplier_m"]): r for r in contrasts["fold_rows"]}
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        f"- generated_at: `{setup['generated_at']}`",
        f"- smoke: `{str(smoke).lower()}`",
        f"- seed: `{setup['seed']}`",
        f"- weight_policy: `{WEIGHT_POLICY}`",
        f"- lag_definition: `{LAG_DEFINITION}`",
        f"- feature_set: L42 (`{setup['n_features']}` features; drop `{list(ID_DROP)}`; keep `game_type`)",
        f"- CatBoost params: `{json.dumps(setup['catboost_params'], ensure_ascii=False)}`",
        "- decision: `results_only_no_adopt_reject`",
        "",
        "## Objective",
        "",
        "RECENCY-AUDIT-001 full-history training window를 유지한 채, validation 시즌에",
        "가까운 학습 시즌에 더 큰 sample weight를 주는 것이 다음 시즌 raw BSS를",
        "바꾸는지 측정한다. 바뀌는 핵심 변수는 multiplier `m` 하나뿐이다.",
        "",
        "## Experimental contract",
        "",
        "- 5 multipliers × 3 folds = 15 CatBoost L42 fits",
        f"- multipliers: `{list(MULTIPLIERS)}`",
        "- `m = 1.0` is CONTROL (all training weights = 1.0)",
        "- 바뀌는 것: validation-relative season-lag sample weight",
        "- 바뀌지 않는 것: full-history window, L42, CatBoost hyperparameters, seed 42, metric, preprocessing",
        "- Public LB / 신규 submission 없음",
        "- ADOPT / REJECT / HOLD 판정 없음",
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
        "## Weight definition",
        "",
        "```text",
        "lag = y - s",
        "L = max lag in the fold",
        "w(lag) = m ** (1 - (lag - 1) / (L - 1))",
        "w(1) = m",
        "w(L) = 1",
        "L == 1 -> w = m",
        "```",
        "",
        "Calendar-year fixed weights are not used. The same training year can have",
        "different weights across validation folds.",
        "",
        "## Fold definitions",
        "",
        "| fold | valid | train seasons |",
        "|---|---|---|",
    ]
    for valid in VALID_SEASONS:
        seasons = list(FULL_HISTORY_WINDOWS[valid])
        lines.append(f"| {FOLD_NAMES[valid]} | {valid} | `{seasons}` |")
    lines.extend(
        [
            "",
            "## Fit-level results",
            "",
            "| valid | m | train seasons | train rows | valid rows | season weights | raw BSS | Brier | fit_s |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for valid in VALID_SEASONS:
        for m in MULTIPLIERS:
            row = by_cell[(valid, float(m))]
            lines.append(
                f"| {valid} | {m:.1f} | `{row['train_seasons']}` | {row['train_rows']} | "
                f"{row['valid_rows']} | `{row['season_to_weight']}` | {_fmt(row['raw_bss'])} | "
                f"{_fmt(row['brier'], 8)} | {_fmt(row['fit_seconds'], 1)} |"
            )
    lines.extend(
        [
            "",
            "## Compact raw BSS",
            "",
            "| m | 2022 | 2023 | 2024 | mean | std | min |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for m in MULTIPLIERS:
        c = contrast_map[float(m)]
        lines.append(
            f"| {m:.1f} | {_fmt(c['raw_bss_2022'])} | {_fmt(c['raw_bss_2023'])} | "
            f"{_fmt(c['raw_bss_2024'])} | {_fmt(c['mean_raw_bss'])} | "
            f"{_fmt(c['std_raw_bss'])} | {_fmt(c['min_raw_bss'])} |"
        )
    lines.extend(
        [
            "",
            "## Contrasts vs m=1.0 CONTROL",
            "",
            "`delta_raw_bss = treatment_raw_bss - control_raw_bss`",
            "",
            "| m | Δ2022 | Δ2023 | Δ2024 | mean Δ | std Δ | min Δ |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for m in MULTIPLIERS:
        c = contrast_map[float(m)]
        lines.append(
            f"| {m:.1f} | {_fmt(c['delta_2022'])} | {_fmt(c['delta_2023'])} | "
            f"{_fmt(c['delta_2024'])} | {_fmt(c['mean_delta'])} | "
            f"{_fmt(c['std_delta'])} | {_fmt(c['min_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Fold stability",
            "",
            "Values below are descriptive statistics only. No adoption decision is made here.",
            "",
        ]
    )
    for m in MULTIPLIERS:
        c = contrast_map[float(m)]
        lines.append(
            f"- m={m:.1f}: mean raw BSS `{_fmt(c['mean_raw_bss'])}`, "
            f"std `{_fmt(c['std_raw_bss'])}`, min `{_fmt(c['min_raw_bss'])}`; "
            f"mean Δ `{_fmt(c['mean_delta'])}`, std Δ `{_fmt(c['std_delta'])}`, "
            f"min Δ `{_fmt(c['min_delta'])}`"
        )
    lines.extend(["", "## m=1.0 reproduction gate", ""])
    if gate is None:
        lines.append("Gate skipped (smoke).")
    else:
        lines.append(f"- status: `{gate['status']}`")
        lines.append(f"- raw BSS atol: `{gate['atol_raw_bss']}`")
        lines.append(f"- brier atol: `{gate['atol_brier']}`")
        lines.append("")
        lines.append("| valid | recency raw BSS | control raw BSS | Δ | train rows | valid rows |")
        lines.append("|---|---|---|---|---|---|")
        for valid in VALID_SEASONS:
            fold = gate["folds"][str(valid)]
            lines.append(
                f"| {valid} | {_fmt(fold['recency_raw_bss'])} | {_fmt(fold['control_raw_bss'])} | "
                f"{_fmt(fold['raw_bss_diff'])} | {fold['control_train_rows']} | "
                f"{fold['control_valid_rows']} |"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- single seed 42",
            "- CatBoost L42 only",
            "- full-history window locked from RECENCY-AUDIT-001 B_expanding",
            "- Public LB not used",
            "- no submission",
            "- `load_competition_data` opens test.csv, but this experiment never uses it for weights, preprocessing, or metrics",
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
            "이번 실험은 `outputs/experiments.csv`에 append하지 않았다.",
            "RECENCY-AUDIT-001 / VAL-AUDIT-001과 같은 진단 convention이다.",
        ]
    )
    if smoke:
        lines.extend(
            [
                "",
                "> SMOKE RUN. 이 숫자는 canonical 결과가 아니다.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_ID}: season-lag sample weighting on CatBoost L42."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", default="outputs/analysis/weight_lag_001")
    parser.add_argument("--report-path", default="reports/weight_lag_001.md")
    parser.add_argument(
        "--reproduction-reference",
        default=DEFAULT_RECENCY_RESULTS,
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--smoke-rows-per-season", type=int, default=800)
    parser.add_argument("--smoke-iterations", type=int, default=20)
    return parser.parse_args()


def _execute_cells(
    *,
    specs: list[CellSpec],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    train_df: pd.DataFrame,
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    params: dict[str, Any],
    seed: int,
    scale: float,
    feature_fp: str,
    smoke: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    n_executed = 0
    n_reused = 0
    current_cell = "before_fits"
    try:
        for spec in specs:
            current_cell = spec.cell_id
            key = cell_cache_key(
                spec=spec,
                seed=seed,
                params=params,
                n_features=len(used),
                smoke=bool(smoke),
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
                feature_fp=feature_fp,
            )
            rows.append(row)
            n_executed += 1
            record = {
                "cell_key": key,
                "experiment_id": EXPERIMENT_ID,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "metrics": row,
            }
            append_fit_cache(cache_path, record)
            cache[key] = record
            print(
                f"  [{spec.cell_id}] raw_bss={row['raw_bss']:.6f} "
                f"brier={row['brier']:.8f} fit_s={row['fit_seconds']:.1f}",
                flush=True,
            )
    except Exception:
        completed = [r["cell_id"] for r in rows]
        print(
            f"FAILED at {current_cell}. completed={completed} "
            f"cache={cache_path} (resume with the same command)",
            flush=True,
        )
        raise
    return rows, n_executed, n_reused


def main() -> None:
    args = parse_args()
    wall0 = time.perf_counter()
    assert_full_history_matches_recency()
    multipliers = (CONTROL_M,) if args.control_only else MULTIPLIERS
    cells = iter_cell_specs(multipliers)
    control_specs = [c for c in cells if c.is_control]
    treatment_specs = [c for c in cells if not c.is_control]

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

    print(f"=== {EXPERIMENT_ID} ===", flush=True)
    print(
        f"cells={len(cells)} weight_policy={WEIGHT_POLICY} seed={seed} "
        f"smoke={args.smoke} control_only={args.control_only}",
        flush=True,
    )
    if args.dry_run:
        for cell in cells:
            mapping = season_weight_map(cell.train_seasons, cell.valid_season, cell.multiplier)
            print(
                f"  {cell.cell_id}: train={list(cell.train_seasons)} weights={mapping}",
                flush=True,
            )
        return

    loaded = load_competition_data(config)
    train_df = loaded["train"].reset_index(drop=True)
    _ = loaded["test"]  # loaded by the shared helper; never used in this experiment
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
    feature_fp = feature_list_hash(used)
    print(
        f"L42 features: {len(used)} (cats={len(categorical_cols)}; drop={list(ID_DROP)}; "
        f"fingerprint={feature_fp[:12]})",
        flush=True,
    )

    out_dir = ensure_dir(resolve_path(args.out_dir + ("_smoke" if args.smoke else "")))
    report_path = None if args.smoke else resolve_path(args.report_path)
    assert_protected_paths(out_dir, report_path)
    cache_path = out_dir / "fits.jsonl"
    cache = {} if args.no_resume else load_fit_cache(cache_path)
    if cache:
        print(f"resume: {len(cache)} cached fits in {cache_path}", flush=True)

    control_rows, n_exec_c, n_reuse_c = _execute_cells(
        specs=control_specs,
        cache=cache,
        cache_path=cache_path,
        train_df=train_df,
        used=used,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        scale=scale,
        feature_fp=feature_fp,
        smoke=bool(args.smoke),
    )
    n_executed = n_exec_c
    n_reused = n_reuse_c

    gate: dict[str, Any] | None = None
    recency_fp: dict[str, Any] | None = None
    if not args.smoke:
        ref_path = resolve_path(args.reproduction_reference)
        recency = load_recency_b_expanding(ref_path)
        recency_bytes = ref_path.read_bytes()
        recency_fp = {
            "path": str(ref_path),
            "sha256": hashlib.sha256(recency_bytes).hexdigest(),
            "nbytes": len(recency_bytes),
        }
        gate = reproduction_gate(control_rows, recency)
        print(f"reproduction gate: {gate['status']}", flush=True)

    treatment_rows: list[dict[str, Any]] = []
    if treatment_specs:
        t_rows, n_exec_t, n_reuse_t = _execute_cells(
            specs=treatment_specs,
            cache=cache,
            cache_path=cache_path,
            train_df=train_df,
            used=used,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_col=target_col,
            season_col=season_col,
            params=params,
            seed=seed,
            scale=scale,
            feature_fp=feature_fp,
            smoke=bool(args.smoke),
        )
        treatment_rows = t_rows
        n_executed += n_exec_t
        n_reused += n_reuse_t

    rows = control_rows + treatment_rows
    generated_at = datetime.now().isoformat(timespec="seconds")
    command = "python -m scripts.run_weight_lag_001"
    if args.smoke:
        command += " --smoke"
    if args.control_only:
        command += " --control-only"
    setup = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": generated_at,
        "config": str(args.config),
        "seed": seed,
        "catboost_params": params,
        "n_features": len(used),
        "features": used,
        "feature_fingerprint": feature_fp,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dropped_ids": list(ID_DROP),
        "weight_policy": WEIGHT_POLICY,
        "lag_definition": LAG_DEFINITION,
        "multipliers": list(multipliers),
        "smoke": bool(args.smoke),
        "control_only": bool(args.control_only),
        "full_history_windows": {
            str(valid): list(seasons) for valid, seasons in FULL_HISTORY_WINDOWS.items()
        },
        "reproduction_reference": recency_fp,
    }
    results_df = pd.DataFrame(rows)
    results_path = out_dir / "results.csv"
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": generated_at,
        "setup": setup,
        "n_fits_executed": n_executed,
        "n_fits_reused": n_reused,
        "failed_cell": None,
        "reproduction_gate": gate,
        "elapsed_seconds_wall": float(time.perf_counter() - wall0),
        "decision": "results_only_no_adopt_reject",
        "experiments_csv": "not_appended_audit_convention",
        "command": command,
    }

    contrasts = None
    contrasts_df = None
    contrasts_path = out_dir / "contrasts.csv"
    if not args.control_only:
        assert_rowcount_and_feature_lock(rows)
        contrasts = compute_contrasts(rows)
        contrasts_df = contrasts_table(contrasts)
        assert_artifact_integrity(rows, contrasts, results_df, contrasts_df)
        metadata["contrasts"] = contrasts
        contrasts_df.to_csv(contrasts_path, index=False)

    metadata_path = out_dir / "metadata.json"
    results_df.to_csv(results_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_text = None
    if contrasts is not None:
        report_text = build_markdown(
            setup=setup,
            rows=rows,
            contrasts=contrasts,
            gate=gate,
            commands=[command],
            smoke=bool(args.smoke),
        )
        (out_dir / "report.md").write_text(report_text, encoding="utf-8")
        if report_path is not None:
            ensure_dir(report_path.parent)
            report_path.write_text(report_text, encoding="utf-8")

    print(f"wrote {results_path}", flush=True)
    if contrasts_df is not None:
        print(f"wrote {contrasts_path}", flush=True)
    print(f"wrote {metadata_path}", flush=True)
    if report_text is not None:
        print(f"wrote {out_dir / 'report.md'}", flush=True)
        if report_path is not None:
            print(f"wrote {report_path}", flush=True)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "n_executed": n_executed,
                "n_reused": n_reused,
                "n_rows": len(rows),
                "reproduction_gate": None if gate is None else gate["status"],
                "elapsed_seconds_wall": metadata["elapsed_seconds_wall"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
