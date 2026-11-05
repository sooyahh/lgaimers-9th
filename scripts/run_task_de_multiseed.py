"""Task D (C-0 vs C-4 multi-seed) then Task E (seed ensemble). Results only."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from scripts.tree_exp_common import (
    build_feature_lists,
    positive_class_proba,
    prepare_native_frames,
)
from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.models import build_model
from src.utils import ensure_dir, load_config, resolve_path, set_seed

ID_DROP = ["pitcher_id", "batter_id"]
HOLD_TRAIN = [2019, 2020, 2021, 2022, 2023]
HOLD_VALID = [2024]
DEFAULT_WEIGHTS = {int(k): float(v) for k, v in SEASON_WEIGHTS.items()}
SEEDS = [42, 1, 7, 123, 2024]
ADD_THREE = ["li", "home_win_expectancy", "away_win_expectancy"]
EXP006_CATBOOST = {
    "depth": 7,
    "learning_rate": 0.028199996263578355,
    "l2_leaf_reg": 1.2151617026673374,
    "iterations": 361,
    "bagging_temperature": 0.9656320330745594,
    "loss_function": "Logloss",
}


def make_weights(seasons: np.ndarray, weight_map: dict[int, float]) -> np.ndarray:
    return np.asarray([float(weight_map[int(s)]) for s in seasons], dtype=float)


def base_feature_lists(
    train_df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str],
    categorical_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    used0, nums0, cats0 = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols,
    )
    used = [c for c in used0 if c not in ID_DROP]
    cats = [c for c in cats0 if c not in ID_DROP]
    nums = [c for c in nums0 if c in used]
    assert "game_type" in used
    assert "pitcher_id" not in used and "batter_id" not in used
    return used, nums, cats


def run_holdout(
    *,
    label: str,
    train_df: pd.DataFrame,
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    params: dict[str, Any],
    seed: int,
    scale: float,
    weight_map: dict[int, float],
    return_pred: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    set_seed(seed)
    fold = build_season_holdout(
        train_df, season_col, HOLD_TRAIN, HOLD_VALID, name=label
    )
    cats = [c for c in categorical_cols if c in used]
    nums = [c for c in numeric_cols if c in used]
    X_tr_raw = train_df.iloc[fold.train_idx][used]
    y_tr = train_df.iloc[fold.train_idx][target_col]
    X_va_raw = train_df.iloc[fold.valid_idx][used]
    y_va = train_df.iloc[fold.valid_idx][target_col]
    X_tr, X_va = prepare_native_frames(
        X_tr_raw, X_va_raw, numeric_cols=nums, categorical_cols=cats, backend="catboost"
    )
    sw = make_weights(train_df.iloc[fold.train_idx][season_col].to_numpy(), weight_map)
    fit_params = {
        **params,
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    model = build_model("catboost", fit_params, seed)
    model.fit(X_tr, y_tr, cat_features=cats, sample_weight=sw)
    pred, _ = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    elapsed = time.perf_counter() - t0
    print(
        f"  [{label}] seed={seed} n_feat={len(used)} "
        f"raw_bss={metrics['raw_brier_skill_score']:.6f} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    out: dict[str, Any] = {
        "label": label,
        "seed": seed,
        "n_features": len(used),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        "brier_score": float(metrics["brier_score"]),
        "elapsed_seconds": float(elapsed),
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
    }
    if return_pred:
        out["pred"] = np.asarray(pred, dtype=float)
        out["y_true"] = np.asarray(y_va, dtype=float)
        out["row_id"] = train_df.iloc[fold.valid_idx]["row_id"].to_numpy()
        out["valid_idx"] = np.asarray(fold.valid_idx)
    return out


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def task_d(
    train_df: pd.DataFrame,
    base_exclude: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    params: dict[str, Any],
    scale: float,
    eda_dir: Path,
) -> dict[str, Any]:
    print("=== Task D: C-0 vs C-4 multi-seed ===", flush=True)
    used_c0, nums_c0, cats_c0 = base_feature_lists(
        train_df, target_col, base_exclude, categorical_cols
    )
    still_exclude = [c for c in base_exclude if c not in ADD_THREE]
    used_c4, nums_c4, cats_c4 = base_feature_lists(
        train_df, target_col, still_exclude, categorical_cols
    )
    assert len(used_c0) == 42 and len(used_c4) == 45

    c0_rows: list[dict[str, Any]] = []
    c4_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        c0 = run_holdout(
            label=f"D_C0_seed_{seed}",
            train_df=train_df,
            used=used_c0,
            numeric_cols=nums_c0,
            categorical_cols=cats_c0,
            target_col=target_col,
            season_col=season_col,
            params=params,
            seed=seed,
            scale=scale,
            weight_map=DEFAULT_WEIGHTS,
            return_pred=True,
        )
        c4 = run_holdout(
            label=f"D_C4_seed_{seed}",
            train_df=train_df,
            used=used_c4,
            numeric_cols=nums_c4,
            categorical_cols=cats_c4,
            target_col=target_col,
            season_col=season_col,
            params=params,
            seed=seed,
            scale=scale,
            weight_map=DEFAULT_WEIGHTS,
            return_pred=False,
        )
        c0_rows.append(c0)
        c4_rows.append(c4)

    paired = []
    for a, b in zip(c0_rows, c4_rows):
        assert a["seed"] == b["seed"]
        d = float(b["raw_brier_skill_score"] - a["raw_brier_skill_score"])
        paired.append(
            {
                "seed": a["seed"],
                "c0_raw_bss": a["raw_brier_skill_score"],
                "c4_raw_bss": b["raw_brier_skill_score"],
                "paired_delta_c4_minus_c0": d,
            }
        )
    deltas = np.asarray([p["paired_delta_c4_minus_c0"] for p in paired], dtype=float)
    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

    payload = {
        "task": "D_task_c_multiseed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "setup": {
            "model": "catboost",
            "params": params,
            "seeds": SEEDS,
            "c0_n_features": 42,
            "c4_n_features": 45,
            "c4_added": ADD_THREE,
            "train_seasons": HOLD_TRAIN,
            "valid_seasons": HOLD_VALID,
            "sample_weight_by_season": DEFAULT_WEIGHTS,
        },
        "paired": paired,
        "summary": {
            "paired_delta_mean": mean_d,
            "paired_delta_std": std_d,
            "c0_mean_raw_bss": float(np.mean([p["c0_raw_bss"] for p in paired])),
            "c4_mean_raw_bss": float(np.mean([p["c4_raw_bss"] for p in paired])),
        },
        "c0_runs": [{k: v for k, v in r.items() if k not in ("pred", "y_true", "row_id", "valid_idx")} for r in c0_rows],
        "c4_runs": c4_rows,
    }

    md = [
        "# Task C multi-seed recheck (Task D)",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "- C-0: EXP-005 L 42 features; C-4: +li +home_win_expectancy +away_win_expectancy (45)",
        f"- params: `{json.dumps(params, ensure_ascii=False)}`",
        f"- seeds: `{SEEDS}`",
        "- paired Δ = C-4 raw BSS − C-0 raw BSS (same seed)",
        "- **results only — no adopt/reject**",
        "",
        "| seed | C-0 raw BSS | C-4 raw BSS | paired Δ (C4−C0) |",
        "|---:|---:|---:|---:|",
    ]
    for p in paired:
        md.append(
            f"| {p['seed']} | {p['c0_raw_bss']:.6f} | {p['c4_raw_bss']:.6f} | "
            f"{p['paired_delta_c4_minus_c0']:+.6f} |"
        )
    md.extend(
        [
            "",
            f"- paired Δ mean: `{mean_d:+.6f}`",
            f"- paired Δ std (sample, ddof=1): `{std_d:.6f}`",
            f"- C-0 mean raw BSS: `{payload['summary']['c0_mean_raw_bss']:.6f}`",
            f"- C-4 mean raw BSS: `{payload['summary']['c4_mean_raw_bss']:.6f}`",
            "",
        ]
    )
    write_text(eda_dir / "task_c_multiseed.md", "\n".join(md))
    write_json(eda_dir / "task_c_multiseed.json", payload)
    return {"payload": payload, "c0_rows": c0_rows}


def ensemble_table(
    *,
    name: str,
    params: dict[str, Any],
    seed_preds: dict[int, np.ndarray],
    y_true: np.ndarray,
    scale: float,
    singles: list[dict[str, Any]],
) -> dict[str, Any]:
    """1 / 3 / 5 seed prediction averages (seed order = SEEDS)."""
    ordered = [seed_preds[s] for s in SEEDS]
    combos = [
        ("n1_seed_42", ordered[:1]),
        ("n3_seeds_42_1_7", ordered[:3]),
        ("n5_all_seeds", ordered[:5]),
    ]
    rows = []
    for label, preds_list in combos:
        avg = np.mean(np.vstack(preds_list), axis=0)
        avg = np.clip(avg, 0.0, 1.0)
        m = evaluate_predictions(y_true, avg, scale=scale)
        rows.append(
            {
                "ensemble": label,
                "n_models": len(preds_list),
                "seeds": SEEDS[: len(preds_list)],
                "raw_brier_skill_score": float(m["raw_brier_skill_score"]),
                "official_brier_skill_score": float(m["official_brier_skill_score"]),
                "brier_score": float(m["brier_score"]),
            }
        )
    single_scores = [float(s["raw_brier_skill_score"]) for s in singles]
    return {
        "name": name,
        "params": params,
        "singles": [
            {
                "seed": s["seed"],
                "raw_brier_skill_score": s["raw_brier_skill_score"],
                "official_brier_skill_score": s["official_brier_skill_score"],
                "brier_score": s["brier_score"],
                "elapsed_seconds": s["elapsed_seconds"],
            }
            for s in singles
        ],
        "single_mean_raw_bss": float(np.mean(single_scores)),
        "single_std_raw_bss": float(np.std(single_scores, ddof=1)),
        "ensembles": rows,
    }


def task_e(
    train_df: pd.DataFrame,
    used: list[str],
    nums: list[str],
    cats: list[str],
    target_col: str,
    season_col: str,
    params_exp005: dict[str, Any],
    scale: float,
    eda_dir: Path,
    pred_dir: Path,
    c0_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    print("=== Task E: seed ensemble ===", flush=True)
    ensure_dir(pred_dir)

    # --- EXP-005 params: reuse C-0 preds from Task D ---
    y_true = c0_rows[0]["y_true"]
    row_id = c0_rows[0]["row_id"]
    for r in c0_rows[1:]:
        assert np.array_equal(r["valid_idx"], c0_rows[0]["valid_idx"])

    pred_map_005 = {int(r["seed"]): r["pred"] for r in c0_rows}
    df_005 = pd.DataFrame({"row_id": row_id, "y_true": y_true})
    for s in SEEDS:
        df_005[f"pred_seed_{s}"] = pred_map_005[s]
    df_005.to_csv(pred_dir / "exp005_seed_preds_2024.csv", index=False)

    block_005 = ensemble_table(
        name="EXP-005_params",
        params=params_exp005,
        seed_preds=pred_map_005,
        y_true=y_true,
        scale=scale,
        singles=c0_rows,
    )

    # --- EXP-006 Optuna best params ---
    print("=== Task E: EXP-006 tuned params seed runs ===", flush=True)
    rows_006: list[dict[str, Any]] = []
    for seed in SEEDS:
        r = run_holdout(
            label=f"E_exp006_seed_{seed}",
            train_df=train_df,
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            target_col=target_col,
            season_col=season_col,
            params=EXP006_CATBOOST,
            seed=seed,
            scale=scale,
            weight_map=DEFAULT_WEIGHTS,
            return_pred=True,
        )
        rows_006.append(r)

    pred_map_006 = {int(r["seed"]): r["pred"] for r in rows_006}
    df_006 = pd.DataFrame({"row_id": row_id, "y_true": y_true})
    for s in SEEDS:
        df_006[f"pred_seed_{s}"] = pred_map_006[s]
    df_006.to_csv(pred_dir / "exp006_seed_preds_2024.csv", index=False)

    block_006 = ensemble_table(
        name="EXP-006_optuna_best_params",
        params=EXP006_CATBOOST,
        seed_preds=pred_map_006,
        y_true=y_true,
        scale=scale,
        singles=rows_006,
    )

    payload = {
        "task": "E_seed_ensemble",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "setup": {
            "features_n": len(used),
            "drop_ids": ID_DROP,
            "game_type": "kept",
            "train_seasons": HOLD_TRAIN,
            "valid_seasons": HOLD_VALID,
            "sample_weight_by_season": DEFAULT_WEIGHTS,
            "seeds": SEEDS,
            "ensemble_definition": {
                "n1": "mean of preds from seed [42]",
                "n3": "mean of preds from seeds [42, 1, 7]",
                "n5": "mean of preds from seeds [42, 1, 7, 123, 2024]",
            },
        },
        "exp005": block_005,
        "exp006": block_006,
        "pred_paths": {
            "exp005": str(pred_dir / "exp005_seed_preds_2024.csv"),
            "exp006": str(pred_dir / "exp006_seed_preds_2024.csv"),
        },
    }

    md = [
        "# Seed ensemble (Task E)",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "- holdout: train 2019–2023 → valid 2024; L 42 features; current season weights",
        f"- seeds: `{SEEDS}`",
        "- ensemble: arithmetic mean of positive-class probabilities, then clip to [0,1]",
        "- n1 = seed 42 only; n3 = seeds 42,1,7; n5 = all five",
        "- **results only — no adopt/reject**",
        "",
        "## EXP-005 params",
        "",
        f"- params: `{json.dumps(params_exp005, ensure_ascii=False)}`",
        "",
        "| seed | raw BSS | official BSS | Brier | elapsed_s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in block_005["singles"]:
        md.append(
            f"| {s['seed']} | {s['raw_brier_skill_score']:.6f} | "
            f"{s['official_brier_skill_score']:.6f} | {s['brier_score']:.6f} | "
            f"{s['elapsed_seconds']:.1f} |"
        )
    md.extend(
        [
            "",
            f"- single-model mean / std: `{block_005['single_mean_raw_bss']:.6f}` / "
            f"`{block_005['single_std_raw_bss']:.6f}`",
            "",
            "| ensemble | n | seeds | raw BSS | official BSS | Brier |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for e in block_005["ensembles"]:
        md.append(
            f"| {e['ensemble']} | {e['n_models']} | `{e['seeds']}` | "
            f"{e['raw_brier_skill_score']:.6f} | {e['official_brier_skill_score']:.6f} | "
            f"{e['brier_score']:.6f} |"
        )

    md.extend(
        [
            "",
            "## EXP-006 Optuna best CatBoost params",
            "",
            f"- params: `{json.dumps(EXP006_CATBOOST, ensure_ascii=False)}`",
            "",
            "| seed | raw BSS | official BSS | Brier | elapsed_s |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for s in block_006["singles"]:
        md.append(
            f"| {s['seed']} | {s['raw_brier_skill_score']:.6f} | "
            f"{s['official_brier_skill_score']:.6f} | {s['brier_score']:.6f} | "
            f"{s['elapsed_seconds']:.1f} |"
        )
    md.extend(
        [
            "",
            f"- single-model mean / std: `{block_006['single_mean_raw_bss']:.6f}` / "
            f"`{block_006['single_std_raw_bss']:.6f}`",
            "",
            "| ensemble | n | seeds | raw BSS | official BSS | Brier |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for e in block_006["ensembles"]:
        md.append(
            f"| {e['ensemble']} | {e['n_models']} | `{e['seeds']}` | "
            f"{e['raw_brier_skill_score']:.6f} | {e['official_brier_skill_score']:.6f} | "
            f"{e['brier_score']:.6f} |"
        )

    md.extend(
        [
            "",
            "## Side-by-side ensemble raw BSS",
            "",
            "| ensemble | EXP-005 params | EXP-006 tuned params |",
            "|---|---:|---:|",
        ]
    )
    for e5, e6 in zip(block_005["ensembles"], block_006["ensembles"]):
        md.append(
            f"| {e5['ensemble']} | {e5['raw_brier_skill_score']:.6f} | "
            f"{e6['raw_brier_skill_score']:.6f} |"
        )
    md.extend(
        [
            "",
            "## Saved predictions",
            "",
            f"- `{pred_dir / 'exp005_seed_preds_2024.csv'}`",
            f"- `{pred_dir / 'exp006_seed_preds_2024.csv'}`",
            "",
        ]
    )
    write_text(eda_dir / "seed_ensemble.md", "\n".join(md))
    # JSON without large arrays
    write_json(eda_dir / "seed_ensemble.json", payload)
    return payload


def main() -> None:
    config = load_config("configs/baseline.yaml")
    params = dict(config.get("models", {}).get("params", {}).get("catboost", {}))
    scale = float(resolve_metric_settings(config.get("task", {}))["scale"])
    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    target_col = data_cfg["target_col"]
    season_col = data_cfg["season_col"]
    exclude_cols = list(feat_cfg.get("exclude_cols") or [])
    categorical_cols = list(feat_cfg.get("categorical_cols") or [])

    train_df = load_competition_data(config)["train"]
    used, nums, cats = base_feature_lists(
        train_df, target_col, exclude_cols, categorical_cols
    )
    assert len(used) == 42

    eda_dir = ensure_dir(resolve_path("outputs/eda"))
    pred_dir = ensure_dir(resolve_path("outputs/predictions/task_de"))

    d = task_d(
        train_df,
        exclude_cols,
        categorical_cols,
        target_col,
        season_col,
        params,
        scale,
        eda_dir,
    )
    task_e(
        train_df,
        used,
        nums,
        cats,
        target_col,
        season_col,
        params,
        scale,
        eda_dir,
        pred_dir,
        d["c0_rows"],
    )
    print("Task D/E complete.", flush=True)


if __name__ == "__main__":
    main()
