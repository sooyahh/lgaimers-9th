"""Gap diagnosis: 2023 holdout under EXP-004 / season-ablation settings."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

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

TRAIN_SEASONS = [2019, 2020, 2021, 2022]
VALID_SEASONS = [2023]


def fit_catboost(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    categorical_cols: list[str],
    params: dict[str, Any],
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> Any:
    model = build_model("catboost", params, seed)
    kw: dict[str, Any] = {"cat_features": categorical_cols}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    model.fit(X_tr, y_tr, **kw)
    return model


def run_one(
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
    sample_weight_map: dict[int, float] | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    fold = build_season_holdout(
        train_df,
        season_col,
        TRAIN_SEASONS,
        VALID_SEASONS,
        name=label,
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
    return {
        "label": label,
        "train_seasons": list(TRAIN_SEASONS),
        "valid_seasons": list(VALID_SEASONS),
        "n_features": len(used),
        "has_season": "season" in used,
        "sample_weight_by_season": (
            {str(k): float(v) for k, v in sorted(sample_weight_map.items())}
            if sample_weight_map
            else None
        ),
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
        "train_target_mean": float(y_tr.mean()),
        "valid_target_rate": float(metrics["target_rate"]),
        "brier_score": float(metrics["brier_score"]),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    config = load_config("configs/baseline.yaml")
    seed = int(config.get("runtime", {}).get("seed", 42))
    set_seed(seed)
    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    params = dict(config.get("models", {}).get("params", {}).get("catboost", {}))
    scale = float(resolve_metric_settings(config.get("task", {}))["scale"])
    target_col = data_cfg["target_col"]
    season_col = data_cfg["season_col"]
    exclude_cols = list(feat_cfg.get("exclude_cols") or [])
    categorical_cols_cfg = list(feat_cfg.get("categorical_cols") or [])

    train_df = load_competition_data(config)["train"]
    used0, nums0, cats0 = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols_cfg,
    )
    used_no_season = [c for c in used0 if c != "season"]
    nums_no_season = [c for c in nums0 if c != "season"]

    # Reference from prior artifacts
    exp003 = pd.read_csv(resolve_path("outputs/predictions/EXP-003/fold_metrics.csv"))
    exp004 = pd.read_csv(resolve_path("outputs/predictions/EXP-004/fold_metrics.csv"))
    ref_003 = float(
        exp003.loc[exp003["fold_id"] == "secondary_2023", "raw_brier_skill_score"].iloc[0]
    )
    ref_004_logged = float(
        exp004.loc[exp004["fold_id"] == "secondary_2023", "raw_brier_skill_score"].iloc[0]
    )

    weights_train = {y: float(SEASON_WEIGHTS[y]) for y in TRAIN_SEASONS}

    print("=== 1. EXP-004 style (season + weights) 2019-2022 → 2023 ===", flush=True)
    r_exp004 = run_one(
        label="EXP004_style_secondary_2023",
        train_df=train_df,
        used=used0,
        numeric_cols=nums0,
        categorical_cols=cats0,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        scale=scale,
        sample_weight_map=weights_train,
    )
    print(f"raw_bss={r_exp004['raw_brier_skill_score']:.6f}", flush=True)

    print("=== 3a. season kept, no weights 2019-2022 → 2023 ===", flush=True)
    r_keep = run_one(
        label="season_kept_unweighted_2023",
        train_df=train_df,
        used=used0,
        numeric_cols=nums0,
        categorical_cols=cats0,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        scale=scale,
        sample_weight_map=None,
    )
    print(f"raw_bss={r_keep['raw_brier_skill_score']:.6f}", flush=True)

    print("=== 3b. season dropped (D), no weights 2019-2022 → 2023 ===", flush=True)
    r_drop = run_one(
        label="season_dropped_D_unweighted_2023",
        train_df=train_df,
        used=used_no_season,
        numeric_cols=nums_no_season,
        categorical_cols=cats0,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        scale=scale,
        sample_weight_map=None,
    )
    print(f"raw_bss={r_drop['raw_brier_skill_score']:.6f}", flush=True)

    print("=== 3c. season dropped + EXP-004 weights 2019-2022 → 2023 ===", flush=True)
    r_drop_w = run_one(
        label="season_dropped_weighted_2023",
        train_df=train_df,
        used=used_no_season,
        numeric_cols=nums_no_season,
        categorical_cols=cats0,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        scale=scale,
        sample_weight_map=weights_train,
    )
    print(f"raw_bss={r_drop_w['raw_brier_skill_score']:.6f}", flush=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": (
            "Predicting a future season (e.g. 2025/test) may share the same failure "
            "mode as secondary_2023 (train past seasons → valid next season)."
        ),
        "common": {
            "model": "catboost",
            "params": params,
            "seed": seed,
            "train_seasons": TRAIN_SEASONS,
            "valid_seasons": VALID_SEASONS,
        },
        "references": {
            "EXP-003_secondary_2023_raw_bss": ref_003,
            "EXP-004_fold_metrics_secondary_2023_raw_bss": ref_004_logged,
        },
        "runs": {
            "exp004_style_season_plus_weights": r_exp004,
            "season_kept_unweighted": r_keep,
            "season_dropped_D_unweighted": r_drop,
            "season_dropped_plus_weights": r_drop_w,
        },
    }

    out_dir = ensure_dir(resolve_path("outputs/eda"))
    json_path = out_dir / "gap_diagnosis.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def row(name: str, r: dict[str, Any] | None, raw: float | None = None) -> str:
        if r is None:
            return f"| {name} | — | — | {raw:.6f} | — | — |"
        return (
            f"| {name} | {r['n_features']} | "
            f"{'yes' if r.get('sample_weight_by_season') else 'no'} | "
            f"{r['raw_brier_skill_score']:.6f} | "
            f"{r['official_brier_skill_score']:.6f} | "
            f"{r['brier_score']:.6f} |"
        )

    md = [
        "# Gap Diagnosis: 2023 holdout vs future-season risk",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "- setup: CatBoost EXP-003/004 params; train=`2019–2022`; valid=`2023`",
        "- **no adopt/reject judgment — results only**",
        "",
        "## Hypothesis under inspection",
        "",
        payload["hypothesis"],
        "",
        "## 1. EXP-004 settings on secondary_2023",
        "",
        "Weights applied on train seasons present: "
        f"`{json.dumps(weights_train)}` (from `SEASON_WEIGHTS`, 2023 key unused).",
        "",
        "| run | n_features | sample_weight | raw BSS | official BSS | Brier |",
        "|---|---:|---|---:|---:|---:|",
        f"| EXP-003 baseline (logged fold_metrics) | 44 | no | {ref_003:.6f} | — | — |",
        f"| EXP-004 logged fold_metrics secondary_2023 | 44 | yes | {ref_004_logged:.6f} | — | — |",
        row("EXP-004 style re-run (season + weights)", r_exp004),
        "",
        f"- re-run vs EXP-003 Δ raw BSS: "
        f"`{r_exp004['raw_brier_skill_score'] - ref_003:+.6f}`",
        f"- re-run vs EXP-004 logged Δ raw BSS: "
        f"`{r_exp004['raw_brier_skill_score'] - ref_004_logged:+.6f}`",
        "",
        "### Run details (EXP-004 style re-run)",
        "",
        f"- n_train / n_valid: `{r_exp004['n_train']:,}` / `{r_exp004['n_valid']:,}`",
        f"- train target mean: `{r_exp004['train_target_mean']:.6f}`",
        f"- valid target rate: `{r_exp004['valid_target_rate']:.6f}`",
        f"- pred mean±std: `{r_exp004['prediction_mean']:.6f}` ± `{r_exp004['prediction_std']:.6f}`",
        f"- elapsed_s: `{r_exp004['elapsed_seconds']:.1f}`",
        "",
        "## 3. Season kept vs dropped on the same 2019–2022 → 2023 split",
        "",
        "| run | season | sample_weight | n_features | raw BSS | official BSS | Brier |",
        "|---|---|---|---:|---:|---:|---:|",
        (
            f"| season kept, unweighted | yes | no | {r_keep['n_features']} | "
            f"{r_keep['raw_brier_skill_score']:.6f} | "
            f"{r_keep['official_brier_skill_score']:.6f} | "
            f"{r_keep['brier_score']:.6f} |"
        ),
        (
            f"| season dropped (D), unweighted | no | no | {r_drop['n_features']} | "
            f"{r_drop['raw_brier_skill_score']:.6f} | "
            f"{r_drop['official_brier_skill_score']:.6f} | "
            f"{r_drop['brier_score']:.6f} |"
        ),
        (
            f"| season kept + weights (EXP-004 style) | yes | yes | {r_exp004['n_features']} | "
            f"{r_exp004['raw_brier_skill_score']:.6f} | "
            f"{r_exp004['official_brier_skill_score']:.6f} | "
            f"{r_exp004['brier_score']:.6f} |"
        ),
        (
            f"| season dropped + weights | no | yes | {r_drop_w['n_features']} | "
            f"{r_drop_w['raw_brier_skill_score']:.6f} | "
            f"{r_drop_w['official_brier_skill_score']:.6f} | "
            f"{r_drop_w['brier_score']:.6f} |"
        ),
        "",
        "### Paired deltas (raw BSS)",
        "",
        f"- drop − kept (unweighted): "
        f"`{r_drop['raw_brier_skill_score'] - r_keep['raw_brier_skill_score']:+.6f}`",
        f"- drop − kept (weighted): "
        f"`{r_drop_w['raw_brier_skill_score'] - r_exp004['raw_brier_skill_score']:+.6f}`",
        "",
        f"- machine-readable: `{json_path}`",
        "",
    ]
    md_path = out_dir / "gap_diagnosis.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
