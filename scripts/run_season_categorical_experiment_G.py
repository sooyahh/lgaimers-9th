"""Experiment G: treat season as categorical (CatBoost primary 2024)."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.tree_exp_common import (
    build_feature_lists,
    positive_class_proba,
    prepare_native_frames,
)
from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.models import build_model
from src.utils import ensure_dir, load_config, resolve_path, set_seed

TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
VALID_SEASONS = [2024]


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

    # G: include season as categorical
    if "season" not in categorical_cols_cfg:
        categorical_cols_cfg = categorical_cols_cfg + ["season"]

    loaded = load_competition_data(config)
    train_df = loaded["train"]

    used, numeric_cols, categorical_cols = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols_cfg,
    )
    assert "season" in categorical_cols
    assert "season" not in numeric_cols
    assert "season" in used

    t0 = time.perf_counter()
    fold = build_season_holdout(
        train_df, season_col, TRAIN_SEASONS, VALID_SEASONS, name="G_season_cat"
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
    # sanity: season values as strings for CatBoost cats
    train_season_levels = sorted(X_tr["season"].astype(str).unique().tolist())
    valid_season_levels = sorted(X_va["season"].astype(str).unique().tolist())
    unseen_in_valid = sorted(set(valid_season_levels) - set(train_season_levels))

    model = build_model("catboost", params, seed)
    model.fit(X_tr, y_tr, cat_features=categorical_cols)
    pred, _ = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    elapsed = float(time.perf_counter() - t0)

    res_g = {
        "label": "G_season_categorical",
        "n_features": len(used),
        "categorical_cols": categorical_cols,
        "numeric_has_season": "season" in numeric_cols,
        "categorical_has_season": "season" in categorical_cols,
        "train_season_levels": train_season_levels,
        "valid_season_levels": valid_season_levels,
        "unseen_season_levels_in_valid": unseen_in_valid,
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
        "experiment_H": "skipped (clipping has no effect for valid=2024=train_max+1)",
    }
    print(
        f"G raw_bss={res_g['raw_brier_skill_score']:.6f} "
        f"unseen_levels={unseen_in_valid}",
        flush=True,
    )

    baseline = json.loads(
        resolve_path("outputs/predictions/EXP-003/summary.json").read_text(encoding="utf-8")
    )["primary_2024"]
    prior_path = resolve_path("outputs/eda/regime_shift_experiments.json")
    a_raw, a_off, a_brier = 578.184416, 578.184416, float("nan")
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if "A" in prior:
            a_raw = float(prior["A"]["raw_brier_skill_score"])
            a_off = float(prior["A"]["official_brier_skill_score"])
            a_brier = float(prior["A"]["brier_score"])
        prior["G"] = res_g
        prior["H"] = {"status": "skipped", "reason": res_g["experiment_H"]}
        prior_path.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")

    b_raw = float(baseline["raw_brier_skill_score"])
    b_off = float(baseline["official_brier_skill_score"])
    b_brier = float(baseline["brier_score"])
    g_raw = res_g["raw_brier_skill_score"]
    g_off = res_g["official_brier_skill_score"]
    g_brier = res_g["brier_score"]

    out_dir = ensure_dir(resolve_path("outputs/eda"))
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "seed": seed,
        "G": res_g,
        "H": {"status": "skipped", "reason": res_g["experiment_H"]},
        "baseline_exp003": {
            "raw_brier_skill_score": b_raw,
            "official_brier_skill_score": b_off,
            "brier_score": b_brier,
        },
        "experiment_A": {
            "raw_brier_skill_score": a_raw,
            "official_brier_skill_score": a_off,
            "brier_score": a_brier,
        },
    }
    json_path = out_dir / "season_categorical_experiment_G.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Season Categorical Experiment G (H skipped)",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "- common: CatBoost; train=`2019–2023`; valid=`2024`",
        "- **H skipped**: season clipping has no effect when valid year is train_max+1 (2024).",
        "- **no adopt/reject judgment — results only**",
        "",
        "## Comparison (raw BSS primary)",
        "",
        "| experiment | season handling | raw BSS | official BSS | Brier | Δ raw vs EXP-003 |",
        "|---|---|---:|---:|---:|---:|",
        f"| EXP-003 baseline | numeric absolute year | {b_raw:.6f} | {b_off:.6f} | {b_brier:.6f} | +0.000000 |",
        f"| A sample_weight | numeric season + weights | {a_raw:.6f} | {a_off:.6f} | {a_brier:.6f} | {a_raw - b_raw:+.6f} |",
        f"| G season categorical | `season` in categorical_cols | {g_raw:.6f} | {g_off:.6f} | {g_brier:.6f} | {g_raw - b_raw:+.6f} |",
        f"| H season clip | skipped | — | — | — | — |",
        "",
        "## Experiment G details",
        "",
        f"- n_features: `{res_g['n_features']}`",
        f"- categorical includes season: `{res_g['categorical_has_season']}`",
        f"- numeric includes season: `{res_g['numeric_has_season']}`",
        f"- train season levels (as cat strings): `{train_season_levels}`",
        f"- valid season levels: `{valid_season_levels}`",
        f"- unseen in valid vs train: `{unseen_in_valid}`",
        f"- n_train / n_valid: `{res_g['n_train']:,}` / `{res_g['n_valid']:,}`",
        f"- raw BSS: `{g_raw:.6f}`",
        f"- official BSS: `{g_off:.6f}`",
        f"- Brier: `{g_brier:.6f}`",
        f"- elapsed_s: `{elapsed:.1f}`",
        "",
        f"- machine-readable: `{json_path}`",
        "",
    ]
    md_path = out_dir / "season_categorical_experiment_G.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # append to season_extrapolation_experiments.md if present
    extrap = out_dir / "season_extrapolation_experiments.md"
    if extrap.exists():
        extrap.write_text(
            extrap.read_text(encoding="utf-8").rstrip()
            + "\n\n## Follow-up: Experiment G (season categorical; H skipped)\n\n"
            + f"| EXP-003 | A | G |\n|---:|---:|---:|\n"
            + f"| {b_raw:.6f} | {a_raw:.6f} | {g_raw:.6f} |\n\n"
            + f"Details: `{md_path.name}`\n",
            encoding="utf-8",
        )

    regime_md = out_dir / "regime_shift_experiments.md"
    if regime_md.exists():
        regime_md.write_text(
            regime_md.read_text(encoding="utf-8").rstrip()
            + "\n\n## Follow-up: Experiment G (season as categorical; H skipped)\n\n"
            + "| experiment | raw BSS | official BSS | Δ raw vs EXP-003 |\n"
            + "|---|---:|---:|---:|\n"
            + f"| EXP-003 baseline | {b_raw:.6f} | {b_off:.6f} | +0.000000 |\n"
            + f"| A sample_weight | {a_raw:.6f} | {a_off:.6f} | {a_raw - b_raw:+.6f} |\n"
            + f"| G season categorical | {g_raw:.6f} | {g_off:.6f} | {g_raw - b_raw:+.6f} |\n"
            + f"| H season clip | skipped | — | — |\n\n"
            + f"Details: `{md_path.name}`\n",
            encoding="utf-8",
        )

    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
