"""Regime-shift experiments A/B/C on EXP-003 CatBoost primary-2024 setup."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from scripts.tree_exp_common import (
    build_feature_lists,
    positive_class_proba,
    prepare_native_frames,
)
from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.models import build_model
from src.utils import ensure_dir, load_config, resolve_path, set_seed

SEASON_WEIGHTS = {2019: 1.0, 2020: 1.2, 2021: 1.4, 2022: 1.6, 2023: 2.0}
SHAP_SAMPLE = 30_000


def fit_catboost(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    categorical_cols: list[str],
    params: dict[str, Any],
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> Any:
    model = build_model("catboost", params, seed)
    fit_kwargs: dict[str, Any] = {"cat_features": categorical_cols}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_tr, y_tr, **fit_kwargs)
    return model


def run_primary(
    *,
    label: str,
    train_df: pd.DataFrame,
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    train_seasons: list[int],
    valid_seasons: list[int],
    params: dict[str, Any],
    seed: int,
    scale: float,
    sample_weight_map: dict[int, float] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    fold = build_season_holdout(
        train_df,
        season_col,
        train_seasons,
        valid_seasons,
        name=f"{label}_primary_2024",
    )
    X_tr_raw = train_df.iloc[fold.train_idx][used]
    y_tr = train_df.iloc[fold.train_idx][target_col]
    X_va_raw = train_df.iloc[fold.valid_idx][used]
    y_va = train_df.iloc[fold.valid_idx][target_col]

    X_tr, X_va = prepare_native_frames(
        X_tr_raw,
        X_va_raw,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        backend="catboost",
    )

    sample_weight = None
    if sample_weight_map is not None:
        seasons = train_df.iloc[fold.train_idx][season_col].to_numpy()
        sample_weight = np.asarray(
            [float(sample_weight_map[int(s)]) for s in seasons], dtype=float
        )

    model = fit_catboost(
        X_tr, y_tr, categorical_cols, params, seed, sample_weight=sample_weight
    )
    pred, _ = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    elapsed = float(time.perf_counter() - t0)

    weight_summary = None
    if sample_weight is not None:
        weight_summary = {
            str(k): float(v) for k, v in sorted(sample_weight_map.items())
        }
        uniq, counts = np.unique(sample_weight, return_counts=True)
        weight_summary["applied_value_counts"] = {
            str(float(u)): int(c) for u, c in zip(uniq, counts)
        }

    return {
        "label": label,
        "train_seasons": list(train_seasons),
        "valid_seasons": list(valid_seasons),
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
        "train_target_mean": float(y_tr.mean()),
        "valid_target_rate": float(metrics["target_rate"]),
        "brier_score": float(metrics["brier_score"]),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "sample_weight": weight_summary,
        "elapsed_seconds": elapsed,
        "model": model,
        "X_va": X_va,
        "categorical_cols": categorical_cols,
        "used": used,
    }


def shap_season_rank(
    model: Any,
    X_va: pd.DataFrame,
    categorical_cols: list[str],
    used: list[str],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(X_va)
    if n > SHAP_SAMPLE:
        idx = np.sort(rng.choice(n, size=SHAP_SAMPLE, replace=False))
        X_s = X_va.iloc[idx]
    else:
        idx = np.arange(n)
        X_s = X_va

    pool = Pool(X_s, cat_features=categorical_cols)
    # CatBoost native SHAP: shape (n, n_features+1); last col = bias
    shap_mat = model.get_feature_importance(type="ShapValues", data=pool)
    shap_mat = np.asarray(shap_mat, dtype=float)
    if shap_mat.ndim != 2 or shap_mat.shape[1] != len(used) + 1:
        raise ValueError(f"Unexpected ShapValues shape: {shap_mat.shape}, n_features={len(used)}")
    mean_abs = np.mean(np.abs(shap_mat[:, :-1]), axis=0)
    order = np.argsort(-mean_abs)
    ranking = [
        {
            "rank": int(i + 1),
            "feature": used[int(j)],
            "mean_abs_shap": float(mean_abs[int(j)]),
        }
        for i, j in enumerate(order)
    ]
    season_rows = [r for r in ranking if r["feature"] == "season"]
    season_rank = int(season_rows[0]["rank"]) if season_rows else None
    season_mean_abs = float(season_rows[0]["mean_abs_shap"]) if season_rows else None
    return {
        "method": "CatBoost get_feature_importance(type='ShapValues')",
        "n_shap_rows": int(len(X_s)),
        "n_valid_total": int(n),
        "season_in_features": "season" in used,
        "season_rank": season_rank,
        "season_mean_abs_shap": season_mean_abs,
        "n_features": len(used),
        "top_15": ranking[:15],
        "full_ranking": ranking,
    }


def main() -> None:
    config = load_config("configs/baseline.yaml")
    seed = int(config.get("runtime", {}).get("seed", 42))
    set_seed(seed)
    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    params = dict(config.get("models", {}).get("params", {}).get("catboost", {}))
    metric_settings = resolve_metric_settings(config.get("task", {}))
    scale = float(metric_settings["scale"])

    target_col = data_cfg["target_col"]
    season_col = data_cfg["season_col"]
    exclude_cols = list(feat_cfg.get("exclude_cols") or [])
    categorical_cols_cfg = list(feat_cfg.get("categorical_cols") or [])

    loaded = load_competition_data(config)
    train_df = loaded["train"]
    used, numeric_cols, categorical_cols = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols_cfg,
    )

    baseline_path = resolve_path("outputs/predictions/EXP-003/summary.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_primary = baseline["primary_2024"]

    results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_exp003_primary_2024": baseline_primary,
        "params": params,
        "seed": seed,
        "features_used": used,
        "season_in_44_features": "season" in used,
    }

    print("=== Experiment A: sample_weight ===", flush=True)
    res_a = run_primary(
        label="A_sample_weight",
        train_df=train_df,
        used=used,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        season_col=season_col,
        train_seasons=[2019, 2020, 2021, 2022, 2023],
        valid_seasons=[2024],
        params=params,
        seed=seed,
        scale=scale,
        sample_weight_map=SEASON_WEIGHTS,
    )
    results["A"] = {k: v for k, v in res_a.items() if k not in {"model", "X_va", "categorical_cols", "used"}}
    print(
        f"A raw_bss={res_a['raw_brier_skill_score']:.3f} "
        f"official={res_a['official_brier_skill_score']:.3f}",
        flush=True,
    )

    print("=== Experiment B: train 2022-2023 only ===", flush=True)
    res_b = run_primary(
        label="B_recent_seasons",
        train_df=train_df,
        used=used,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        season_col=season_col,
        train_seasons=[2022, 2023],
        valid_seasons=[2024],
        params=params,
        seed=seed,
        scale=scale,
        sample_weight_map=None,
    )
    results["B"] = {k: v for k, v in res_b.items() if k not in {"model", "X_va", "categorical_cols", "used"}}
    print(
        f"B raw_bss={res_b['raw_brier_skill_score']:.3f} "
        f"official={res_b['official_brier_skill_score']:.3f}",
        flush=True,
    )

    print("=== Experiment C: season feature + SHAP ===", flush=True)
    # season already in features → no retrain for metrics; reuse EXP-003 primary model
    model_path = resolve_path("outputs/models/EXP-003/catboost_primary.joblib")
    model_c = joblib.load(model_path)
    # Rebuild X_va with same prep as EXP-003 primary for SHAP
    fold = build_season_holdout(
        train_df,
        season_col,
        [2019, 2020, 2021, 2022, 2023],
        [2024],
        name="C_shap_prep",
    )
    X_tr_raw = train_df.iloc[fold.train_idx][used]
    X_va_raw = train_df.iloc[fold.valid_idx][used]
    _, X_va_c = prepare_native_frames(
        X_tr_raw,
        X_va_raw,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        backend="catboost",
    )
    shap_info = shap_season_rank(model_c, X_va_c, categorical_cols, used, seed)
    results["C"] = {
        "label": "C_season_feature_shap",
        "season_already_in_features": True,
        "action": "no_retrain_SHAP_only_on_EXP003_primary_model",
        "metrics_same_as": "EXP-003 primary_2024",
        "raw_brier_skill_score": float(baseline_primary["raw_brier_skill_score"]),
        "official_brier_skill_score": float(baseline_primary["official_brier_skill_score"]),
        "brier_score": float(baseline_primary["brier_score"]),
        "shap": shap_info,
    }
    print(
        f"C season_rank={shap_info['season_rank']} / {shap_info['n_features']} "
        f"mean_abs_shap={shap_info['season_mean_abs_shap']:.6g}",
        flush=True,
    )

    out_dir = ensure_dir(resolve_path("outputs/eda"))
    json_path = out_dir / "regime_shift_experiments.json"
    # drop full ranking from json? keep it
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    md: list[str] = []
    md.append("# Regime Shift Experiments (EXP-003 CatBoost / primary 2024)")
    md.append("")
    md.append(f"- generated_at: `{results['generated_at']}`")
    md.append("- common eval: valid=`2024` (primary fold)")
    md.append(
        f"- model params: `{json.dumps(params, ensure_ascii=False)}`; seed=`{seed}`"
    )
    md.append("- **no adopt/reject judgment — results only**")
    md.append("")
    md.append("## Summary: raw / official BSS on 2024")
    md.append("")
    md.append(
        "| experiment | train seasons | n_train | raw BSS | official BSS | Brier |"
    )
    md.append("|---|---|---:|---:|---:|---:|")
    md.append(
        f"| EXP-003 baseline | 2019–2023 | 1,221,585 | "
        f"{baseline_primary['raw_brier_skill_score']:.6f} | "
        f"{baseline_primary['official_brier_skill_score']:.6f} | "
        f"{baseline_primary['brier_score']:.6f} |"
    )
    md.append(
        f"| A sample_weight | 2019–2023 (weighted) | {res_a['n_train']:,} | "
        f"{res_a['raw_brier_skill_score']:.6f} | "
        f"{res_a['official_brier_skill_score']:.6f} | "
        f"{res_a['brier_score']:.6f} |"
    )
    md.append(
        f"| B recent-only | 2022–2023 | {res_b['n_train']:,} | "
        f"{res_b['raw_brier_skill_score']:.6f} | "
        f"{res_b['official_brier_skill_score']:.6f} | "
        f"{res_b['brier_score']:.6f} |"
    )
    md.append(
        f"| C season feature | 2019–2023 (same as EXP-003) | 1,221,585 | "
        f"{baseline_primary['raw_brier_skill_score']:.6f} | "
        f"{baseline_primary['official_brier_skill_score']:.6f} | "
        f"{baseline_primary['brier_score']:.6f} |"
    )
    md.append("")
    md.append(
        f"| Δ A − EXP-003 (raw BSS) | {res_a['raw_brier_skill_score'] - baseline_primary['raw_brier_skill_score']:+.6f} |"
    )
    md.append(
        f"| Δ B − EXP-003 (raw BSS) | {res_b['raw_brier_skill_score'] - baseline_primary['raw_brier_skill_score']:+.6f} |"
    )
    md.append("")

    md.append("## Experiment A — sample_weight")
    md.append("")
    md.append("Weights: `2019=1.0, 2020=1.2, 2021=1.4, 2022=1.6, 2023=2.0`.")
    md.append("")
    md.append(
        f"- train/valid: `{res_a['train_seasons']}` → `{res_a['valid_seasons']}`"
    )
    md.append(f"- n_train / n_valid: `{res_a['n_train']:,}` / `{res_a['n_valid']:,}`")
    md.append(f"- train target mean: `{res_a['train_target_mean']:.6f}`")
    md.append(f"- valid target rate: `{res_a['valid_target_rate']:.6f}`")
    md.append(f"- raw BSS: `{res_a['raw_brier_skill_score']:.6f}`")
    md.append(f"- official BSS: `{res_a['official_brier_skill_score']:.6f}`")
    md.append(f"- Brier: `{res_a['brier_score']:.6f}`")
    md.append(f"- elapsed_s: `{res_a['elapsed_seconds']:.1f}`")
    md.append("")

    md.append("## Experiment B — drop 2019–2021")
    md.append("")
    md.append(
        f"- train/valid: `{res_b['train_seasons']}` → `{res_b['valid_seasons']}`"
    )
    md.append(f"- n_train / n_valid: `{res_b['n_train']:,}` / `{res_b['n_valid']:,}`")
    md.append(f"- train target mean: `{res_b['train_target_mean']:.6f}`")
    md.append(f"- valid target rate: `{res_b['valid_target_rate']:.6f}`")
    md.append(f"- raw BSS: `{res_b['raw_brier_skill_score']:.6f}`")
    md.append(f"- official BSS: `{res_b['official_brier_skill_score']:.6f}`")
    md.append(f"- Brier: `{res_b['brier_score']:.6f}`")
    md.append(f"- elapsed_s: `{res_b['elapsed_seconds']:.1f}`")
    md.append("")
    md.append("### Side-by-side with EXP-003 (full 2019–2023 train)")
    md.append("")
    md.append("| | EXP-003 | B (2022–2023 train) |")
    md.append("|---|---:|---:|")
    md.append(
        f"| n_train | 1,221,585 | {res_b['n_train']:,} |"
    )
    md.append(
        f"| raw BSS | {baseline_primary['raw_brier_skill_score']:.6f} | "
        f"{res_b['raw_brier_skill_score']:.6f} |"
    )
    md.append(
        f"| official BSS | {baseline_primary['official_brier_skill_score']:.6f} | "
        f"{res_b['official_brier_skill_score']:.6f} |"
    )
    md.append(
        f"| Brier | {baseline_primary['brier_score']:.6f} | {res_b['brier_score']:.6f} |"
    )
    md.append("")

    md.append("## Experiment C — season feature / SHAP")
    md.append("")
    md.append(
        f"- `season` in the 44 input features? **`{'yes' if 'season' in used else 'no'}`**"
    )
    md.append(
        "- because it is already included: **no retrain**; metrics identical to EXP-003 primary_2024"
    )
    md.append(
        f"- SHAP method: `{shap_info['method']}` on valid-2024 subsample "
        f"n=`{shap_info['n_shap_rows']:,}` (of `{shap_info['n_valid_total']:,}`)"
    )
    md.append(
        f"- **`season` mean|SHAP| rank: `{shap_info['season_rank']}` / "
        f"`{shap_info['n_features']}`** "
        f"(mean_abs_shap=`{shap_info['season_mean_abs_shap']:.6g}`)"
    )
    md.append("")
    md.append("### Top-15 features by mean |SHAP|")
    md.append("")
    md.append("| rank | feature | mean_abs_shap |")
    md.append("|---:|---|---:|")
    for r in shap_info["top_15"]:
        md.append(f"| {r['rank']} | `{r['feature']}` | {r['mean_abs_shap']:.6g} |")
    md.append("")
    md.append(f"- machine-readable: `{json_path}`")
    md.append("")

    md_path = out_dir / "regime_shift_experiments.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {md_path}", flush=True)
    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
