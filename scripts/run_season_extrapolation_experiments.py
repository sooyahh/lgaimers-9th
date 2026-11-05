"""Season extrapolation alternatives D/E/F vs EXP-003 and A."""

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

SEASON_WEIGHTS = {2019: 1.0, 2020: 1.2, 2021: 1.4, 2022: 1.6, 2023: 2.0}
SEASON_START = 2019
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
VALID_SEASONS = [2024]


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


def run_fold(
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
    sample_weight_map: dict[int, float] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    fold = build_season_holdout(
        train_df,
        season_col,
        TRAIN_SEASONS,
        VALID_SEASONS,
        name=f"{label}_primary_2024",
    )
    X_tr_raw = train_df.iloc[fold.train_idx][used].copy()
    y_tr = train_df.iloc[fold.train_idx][target_col]
    X_va_raw = train_df.iloc[fold.valid_idx][used].copy()
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
        "features_used": used,
        "n_features": len(used),
        "has_season": "season" in used,
        "has_season_since_start": "season_since_start" in used,
        "sample_weight": (
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
        "season_since_start_train_range": (
            [
                int(X_tr_raw["season_since_start"].min()),
                int(X_tr_raw["season_since_start"].max()),
            ]
            if "season_since_start" in X_tr_raw.columns
            else None
        ),
        "season_since_start_valid_range": (
            [
                int(X_va_raw["season_since_start"].min()),
                int(X_va_raw["season_since_start"].max()),
            ]
            if "season_since_start" in X_va_raw.columns
            else None
        ),
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
    train_df = loaded["train"].copy()

    used0, numeric0, cats0 = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols_cfg,
    )

    # Prior results
    baseline = json.loads(
        resolve_path("outputs/predictions/EXP-003/summary.json").read_text(encoding="utf-8")
    )["primary_2024"]
    prior = {}
    prior_path = resolve_path("outputs/eda/regime_shift_experiments.json")
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    exp_a = prior.get("A", {})
    a_raw = float(exp_a.get("raw_brier_skill_score", 578.184416))
    a_off = float(exp_a.get("official_brier_skill_score", a_raw))
    a_brier = float(exp_a.get("brier_score", float("nan")))

    results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "seed": seed,
        "baseline_exp003": {
            "raw_brier_skill_score": float(baseline["raw_brier_skill_score"]),
            "official_brier_skill_score": float(baseline["official_brier_skill_score"]),
            "brier_score": float(baseline["brier_score"]),
        },
        "experiment_A": {
            "raw_brier_skill_score": a_raw,
            "official_brier_skill_score": a_off,
            "brier_score": a_brier,
            "note": "from regime_shift_experiments.json if present",
        },
    }

    # --- D: drop season ---
    print("=== Experiment D: drop season ===", flush=True)
    used_d = [c for c in used0 if c != "season"]
    nums_d = [c for c in numeric0 if c != "season"]
    res_d = run_fold(
        label="D_drop_season",
        train_df=train_df,
        used=used_d,
        numeric_cols=nums_d,
        categorical_cols=cats0,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        scale=scale,
        sample_weight_map=None,
    )
    results["D"] = res_d
    print(f"D raw_bss={res_d['raw_brier_skill_score']:.6f}", flush=True)

    # --- E: season_since_start ---
    print("=== Experiment E: season_since_start ===", flush=True)
    train_e = train_df.copy()
    train_e["season_since_start"] = train_e[season_col].astype(int) - SEASON_START
    # feature list: replace season with season_since_start
    used_e = [("season_since_start" if c == "season" else c) for c in used0]
    nums_e = [("season_since_start" if c == "season" else c) for c in numeric0]
    # ensure column exists only once
    assert used_e.count("season_since_start") == 1
    assert "season" not in used_e
    res_e = run_fold(
        label="E_season_since_start",
        train_df=train_e,
        used=used_e,
        numeric_cols=nums_e,
        categorical_cols=cats0,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        scale=scale,
        sample_weight_map=None,
    )
    results["E"] = res_e
    print(
        f"E raw_bss={res_e['raw_brier_skill_score']:.6f} "
        f"train_range={res_e['season_since_start_train_range']} "
        f"valid_range={res_e['season_since_start_valid_range']}",
        flush=True,
    )

    # --- F: drop season + sample_weight ---
    print("=== Experiment F: no season + sample_weight ===", flush=True)
    res_f = run_fold(
        label="F_no_season_sample_weight",
        train_df=train_df,
        used=used_d,
        numeric_cols=nums_d,
        categorical_cols=cats0,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        scale=scale,
        sample_weight_map=SEASON_WEIGHTS,
    )
    results["F"] = res_f
    print(f"F raw_bss={res_f['raw_brier_skill_score']:.6f}", flush=True)

    out_dir = ensure_dir(resolve_path("outputs/eda"))
    json_path = out_dir / "season_extrapolation_experiments.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Merge summary into regime_shift_experiments.md (append section) + dedicated file
    b_raw = float(baseline["raw_brier_skill_score"])
    b_off = float(baseline["official_brier_skill_score"])
    b_brier = float(baseline["brier_score"])

    rows = [
        ("EXP-003 baseline", "2019–2023", "season absolute year", b_raw, b_off, b_brier),
        ("A sample_weight", "2019–2023 weighted", "season kept + weights", a_raw, a_off, a_brier),
        (
            "D drop season",
            "2019–2023",
            "season removed",
            res_d["raw_brier_skill_score"],
            res_d["official_brier_skill_score"],
            res_d["brier_score"],
        ),
        (
            "E season_since_start",
            "2019–2023",
            "season→0..4 (valid=5)",
            res_e["raw_brier_skill_score"],
            res_e["official_brier_skill_score"],
            res_e["brier_score"],
        ),
        (
            "F no season + weights",
            "2019–2023 weighted",
            "season removed + A weights",
            res_f["raw_brier_skill_score"],
            res_f["official_brier_skill_score"],
            res_f["brier_score"],
        ),
    ]

    md: list[str] = []
    md.append("# Season Extrapolation Alternatives (D / E / F)")
    md.append("")
    md.append(f"- generated_at: `{results['generated_at']}`")
    md.append("- common: CatBoost EXP-003 params; train=`2019–2023`; valid=`2024`")
    md.append(f"- params: `{json.dumps(params, ensure_ascii=False)}`; seed=`{seed}`")
    md.append("- **no adopt/reject judgment — results only**")
    md.append("")
    md.append("## Comparison (raw BSS primary)")
    md.append("")
    md.append(
        "| experiment | train | season handling | n_features | raw BSS | "
        "official BSS | Brier | Δ raw vs EXP-003 |"
    )
    md.append("|---|---|---|---:|---:|---:|---:|---:|")
    nfeat = {
        "EXP-003 baseline": 44,
        "A sample_weight": 44,
        "D drop season": res_d["n_features"],
        "E season_since_start": res_e["n_features"],
        "F no season + weights": res_f["n_features"],
    }
    for name, train, handling, raw, off, brier in rows:
        md.append(
            f"| {name} | {train} | {handling} | {nfeat[name]} | "
            f"{raw:.6f} | {off:.6f} | {brier:.6f} | {raw - b_raw:+.6f} |"
        )
    md.append("")

    md.append("## Experiment D — season removed")
    md.append("")
    md.append(f"- features: `{res_d['n_features']}` (excluded `season`)")
    md.append(f"- n_train / n_valid: `{res_d['n_train']:,}` / `{res_d['n_valid']:,}`")
    md.append(f"- raw BSS: `{res_d['raw_brier_skill_score']:.6f}`")
    md.append(f"- official BSS: `{res_d['official_brier_skill_score']:.6f}`")
    md.append(f"- Brier: `{res_d['brier_score']:.6f}`")
    md.append(f"- elapsed_s: `{res_d['elapsed_seconds']:.1f}`")
    md.append("")

    md.append("## Experiment E — `season_since_start = season - 2019`")
    md.append("")
    md.append("- replaces absolute `season` with elapsed years from 2019")
    md.append(
        f"- train range: `{res_e['season_since_start_train_range']}` "
        f"(expect `[0, 4]`); valid range: `{res_e['season_since_start_valid_range']}` "
        "(expect `[5, 5]`)"
    )
    md.append(f"- features: `{res_e['n_features']}`")
    md.append(f"- n_train / n_valid: `{res_e['n_train']:,}` / `{res_e['n_valid']:,}`")
    md.append(f"- raw BSS: `{res_e['raw_brier_skill_score']:.6f}`")
    md.append(f"- official BSS: `{res_e['official_brier_skill_score']:.6f}`")
    md.append(f"- Brier: `{res_e['brier_score']:.6f}`")
    md.append(f"- elapsed_s: `{res_e['elapsed_seconds']:.1f}`")
    md.append("")

    md.append("## Experiment F — no season + sample_weight (A weights)")
    md.append("")
    md.append("- `season` removed from features; weights `2019=1.0 … 2023=2.0`")
    md.append(f"- features: `{res_f['n_features']}`")
    md.append(f"- n_train / n_valid: `{res_f['n_train']:,}` / `{res_f['n_valid']:,}`")
    md.append(f"- raw BSS: `{res_f['raw_brier_skill_score']:.6f}`")
    md.append(f"- official BSS: `{res_f['official_brier_skill_score']:.6f}`")
    md.append(f"- Brier: `{res_f['brier_score']:.6f}`")
    md.append(f"- elapsed_s: `{res_f['elapsed_seconds']:.1f}`")
    md.append("")
    md.append(f"- machine-readable: `{json_path}`")
    md.append("")

    md_path = out_dir / "season_extrapolation_experiments.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Append short pointer section to regime_shift_experiments.md
    regime_md = out_dir / "regime_shift_experiments.md"
    if regime_md.exists():
        append = [
            "",
            "## Follow-up: season extrapolation D / E / F",
            "",
            f"See `{md_path.name}` for full detail.",
            "",
            "| experiment | raw BSS | official BSS | Δ raw vs EXP-003 |",
            "|---|---:|---:|---:|",
        ]
        for name, _, _, raw, off, _ in rows:
            append.append(f"| {name} | {raw:.6f} | {off:.6f} | {raw - b_raw:+.6f} |")
        append.append("")
        regime_md.write_text(
            regime_md.read_text(encoding="utf-8").rstrip() + "\n" + "\n".join(append) + "\n",
            encoding="utf-8",
        )

    # Also merge D/E/F into prior json if exists
    if prior_path.exists():
        prior["D"] = res_d
        prior["E"] = res_e
        prior["F"] = res_f
        prior["season_extrapolation_generated_at"] = results["generated_at"]
        prior_path.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {md_path}", flush=True)
    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
