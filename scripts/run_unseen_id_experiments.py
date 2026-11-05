"""Unseen-ID / asof-relative / game_type experiments on 2023 & 2024 folds."""

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

ID_DROP = ["pitcher_id", "batter_id"]
FOLDS = [
    {
        "fold_id": "secondary_2023",
        "train_seasons": [2019, 2020, 2021, 2022],
        "valid_seasons": [2023],
    },
    {
        "fold_id": "primary_2024",
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "valid_seasons": [2024],
    },
]


def asof_rate_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("asof_") and c.endswith("_rate")]


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


def apply_asof_relative(
    train_df: pd.DataFrame,
    frame: pd.DataFrame,
    rate_cols: list[str],
    season_col: str,
    train_idx: np.ndarray,
) -> pd.DataFrame:
    """Subtract prior-season train means (train rows with season < row.season).

    Means are computed only from train_idx rows. If no prior season exists in
    train for a given season S, subtract 0.0.
    """
    out = frame.copy()
    tr = train_df.iloc[train_idx]
    seasons_sorted = sorted(out[season_col].dropna().unique().tolist())
    prior_mean: dict[int, dict[str, float]] = {}
    for s in seasons_sorted:
        s = int(s)
        prior = tr.loc[tr[season_col] < s, rate_cols]
        means: dict[str, float] = {}
        for c in rate_cols:
            if prior.empty or prior[c].notna().sum() == 0:
                means[c] = 0.0
            else:
                means[c] = float(prior[c].mean())
        prior_mean[s] = means

    for c in rate_cols:
        base = out[season_col].map(lambda s, col=c: prior_mean[int(s)][col])
        out[c] = out[c].astype(float) - base.astype(float)
    return out


def run_config(
    *,
    label: str,
    train_df: pd.DataFrame,
    fold_spec: dict[str, Any],
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    params: dict[str, Any],
    seed: int,
    scale: float,
    sample_weight_map: dict[int, float] | None,
    asof_relative: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    fold = build_season_holdout(
        train_df,
        season_col,
        fold_spec["train_seasons"],
        fold_spec["valid_seasons"],
        name=f"{label}_{fold_spec['fold_id']}",
    )

    rate_cols = asof_rate_cols(used)
    work = train_df
    if asof_relative:
        # Transform whole frame columns used+season for idx selection, but only
        # mutate copies for train/valid slices via apply on full needed rows.
        # Efficient path: transform train_idx and valid_idx subsets separately
        # using means from train_idx only.
        tr_part = apply_asof_relative(
            train_df, train_df.iloc[fold.train_idx].copy(), rate_cols, season_col, fold.train_idx
        )
        va_part = apply_asof_relative(
            train_df, train_df.iloc[fold.valid_idx].copy(), rate_cols, season_col, fold.train_idx
        )
        X_tr_raw = tr_part[used]
        y_tr = tr_part[target_col]
        X_va_raw = va_part[used]
        y_va = va_part[target_col]
        # document prior means for valid season
        valid_s = int(fold_spec["valid_seasons"][0])
        tr = train_df.iloc[fold.train_idx]
        prior = tr.loc[tr[season_col] < valid_s, rate_cols]
        prior_means_valid = {
            c: (float(prior[c].mean()) if prior[c].notna().any() else 0.0) for c in rate_cols
        }
    else:
        X_tr_raw = train_df.iloc[fold.train_idx][used]
        y_tr = train_df.iloc[fold.train_idx][target_col]
        X_va_raw = train_df.iloc[fold.valid_idx][used]
        y_va = train_df.iloc[fold.valid_idx][target_col]
        prior_means_valid = None

    cats = [c for c in categorical_cols if c in used]
    nums = [c for c in numeric_cols if c in used]

    X_tr, X_va = prepare_native_frames(
        X_tr_raw,
        X_va_raw,
        numeric_cols=nums,
        categorical_cols=cats,
        backend="catboost",
    )

    sample_weight = None
    if sample_weight_map is not None:
        seasons = train_df.iloc[fold.train_idx][season_col].to_numpy()
        sample_weight = np.asarray(
            [float(sample_weight_map[int(s)]) for s in seasons], dtype=float
        )

    model = fit_catboost(X_tr, y_tr, cats, params, seed, sample_weight=sample_weight)
    pred, _ = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    elapsed = float(time.perf_counter() - t0)
    return {
        "label": label,
        "fold_id": fold_spec["fold_id"],
        "train_seasons": list(fold_spec["train_seasons"]),
        "valid_seasons": list(fold_spec["valid_seasons"]),
        "n_features": len(used),
        "features_used": used,
        "dropped_ids": [c for c in ID_DROP if c not in used],
        "asof_relative": asof_relative,
        "asof_rate_cols": rate_cols if asof_relative else [],
        "prior_means_for_valid_season": prior_means_valid,
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
    }


def game_type_crosstab(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    g = (
        train_df.groupby(["season", "game_type"], dropna=False)["control_success"]
        .agg(n="size", rate="mean")
        .reset_index()
    )
    wide = g.pivot(index="game_type", columns="season", values="rate")
    # check sign of (rate - overall season rate) flip, or correlation with year within type
    analysis: dict[str, Any] = {"per_type": {}}
    seasons = sorted(train_df["season"].unique().tolist())
    overall = train_df.groupby("season")["control_success"].mean().to_dict()
    reversal_flags = []
    for gt, part in train_df.groupby("game_type"):
        by_s = part.groupby("season")["control_success"].mean()
        # relative to season overall
        rel = {int(s): float(by_s.get(s, np.nan) - overall[s]) for s in seasons if s in by_s.index}
        # early (2019-2022) mean rel vs 2023/2024
        early = [rel[s] for s in (2019, 2020, 2021, 2022) if s in rel and pd.notna(rel[s])]
        late = [rel[s] for s in (2023, 2024) if s in rel and pd.notna(rel[s])]
        flip = False
        if early and late:
            e = float(np.mean(early))
            l = float(np.mean(late))
            flip = (e > 0 and l < 0) or (e < 0 and l > 0)
        analysis["per_type"][str(gt)] = {
            "rel_to_season_mean": rel,
            "early_mean_rel": float(np.mean(early)) if early else None,
            "late_mean_rel": float(np.mean(late)) if late else None,
            "sign_flip_early_vs_late": flip,
            "n": int(len(part)),
        }
        if flip:
            reversal_flags.append(str(gt))
    analysis["types_with_sign_flip"] = reversal_flags
    analysis["reversal_confirmed"] = len(reversal_flags) > 0
    return g, analysis


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

    # EXP-004 weights (only seasons that appear in a given train fold)
    weights_all = {int(k): float(v) for k, v in SEASON_WEIGHTS.items()}

    # References
    exp004 = pd.read_csv(resolve_path("outputs/predictions/EXP-004/fold_metrics.csv"))
    gap = {}
    gap_path = resolve_path("outputs/eda/gap_diagnosis.json")
    if gap_path.exists():
        gap = json.loads(gap_path.read_text(encoding="utf-8"))
    ref_004_2023 = float(
        exp004.loc[exp004.fold_id == "secondary_2023", "raw_brier_skill_score"].iloc[0]
    )
    ref_004_2024 = float(
        exp004.loc[exp004.fold_id == "primary_2024", "raw_brier_skill_score"].iloc[0]
    )
    ref_drop_2023 = float(
        gap.get("runs", {})
        .get("season_dropped_D_unweighted", {})
        .get("raw_brier_skill_score", -1044.664)
    )

    # Unseen ID rates on 2023/2024 for context
    id_stats = {}
    for spec in FOLDS:
        tr = train_df.loc[train_df[season_col].isin(spec["train_seasons"])]
        va = train_df.loc[train_df[season_col].isin(spec["valid_seasons"])]
        id_stats[spec["fold_id"]] = {}
        for col in ID_DROP:
            seen = set(tr[col].dropna().unique())
            va_ids = va[col]
            unseen = ~va_ids.isin(seen)
            id_stats[spec["fold_id"]][col] = {
                "n_valid": int(len(va)),
                "unseen_rows": int(unseen.sum()),
                "unseen_row_rate": float(unseen.mean()),
                "unseen_nunique": int(va_ids[unseen].nunique()),
            }

    print("=== O diagnostic: game_type x season ===", flush=True)
    gt_long, gt_analysis = game_type_crosstab(train_df)
    print("reversal_confirmed", gt_analysis["reversal_confirmed"], gt_analysis["types_with_sign_flip"], flush=True)

    used_L = [c for c in used0 if c not in ID_DROP]
    cats_L = [c for c in cats0 if c not in ID_DROP]
    used_O = [c for c in used0 if c != "game_type"]
    cats_O = [c for c in cats0 if c != "game_type"]
    used_N = [c for c in used_L]  # L+M
    cats_N = cats_L

    results: list[dict[str, Any]] = []

    experiments = [
        ("L_drop_ids", used_L, nums0, cats_L, False),
        ("M_asof_relative", used0, nums0, cats0, True),
        ("N_L_plus_M", used_N, nums0, cats_N, True),
    ]
    if gt_analysis["reversal_confirmed"]:
        experiments.append(("O_drop_game_type", used_O, nums0, cats_O, False))
    else:
        print("O skipped: no sign-flip reversal confirmed", flush=True)

    for label, used, nums, cats, asof_rel in experiments:
        for spec in FOLDS:
            wmap = {y: weights_all[y] for y in spec["train_seasons"] if y in weights_all}
            print(f"=== {label} / {spec['fold_id']} ===", flush=True)
            r = run_config(
                label=label,
                train_df=train_df,
                fold_spec=spec,
                used=used,
                numeric_cols=nums,
                categorical_cols=cats,
                target_col=target_col,
                season_col=season_col,
                params=params,
                seed=seed,
                scale=scale,
                sample_weight_map=wmap,
                asof_relative=asof_rel,
            )
            results.append(r)
            print(
                f" raw_bss={r['raw_brier_skill_score']:.6f} elapsed={r['elapsed_seconds']:.1f}s",
                flush=True,
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "seed": seed,
        "sample_weight_policy": "EXP-004 SEASON_WEIGHTS on train seasons present",
        "references": {
            "EXP004_secondary_2023_raw_bss": ref_004_2023,
            "EXP004_primary_2024_raw_bss": ref_004_2024,
            "season_dropped_D_unweighted_2023_raw_bss": ref_drop_2023,
        },
        "unseen_id_stats": id_stats,
        "game_type_analysis": gt_analysis,
        "game_type_long": gt_long.to_dict(orient="records"),
        "runs": results,
    }

    out_dir = ensure_dir(resolve_path("outputs/eda"))
    json_path = out_dir / "unseen_id_experiments.json"
    # shrink prior_means in json if huge - keep
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown
    by = {(r["label"], r["fold_id"]): r for r in results}
    labels = [e[0] for e in experiments]

    md: list[str] = []
    md.append("# Unseen-ID / asof-relative / game_type experiments")
    md.append("")
    md.append(f"- generated_at: `{payload['generated_at']}`")
    md.append("- base: CatBoost + EXP-004 season sample weights; season feature kept (except O)")
    md.append("- **no adopt/reject judgment — results only**")
    md.append("")
    md.append("## References")
    md.append("")
    md.append("| reference | fold | raw BSS |")
    md.append("|---|---|---:|")
    md.append(f"| EXP-004 | secondary_2023 | {ref_004_2023:.6f} |")
    md.append(f"| season dropped (D), unweighted | secondary_2023 | {ref_drop_2023:.6f} |")
    md.append(f"| EXP-004 | primary_2024 | {ref_004_2024:.6f} |")
    md.append("")
    md.append("## Unseen ID rates (context for L)")
    md.append("")
    md.append("| fold | column | unseen_row_rate | unseen_rows | unseen_nunique |")
    md.append("|---|---|---:|---:|---:|")
    for fid, cols in id_stats.items():
        for col, st in cols.items():
            md.append(
                f"| {fid} | `{col}` | {st['unseen_row_rate']:.4f} | "
                f"{st['unseen_rows']:,} | {st['unseen_nunique']} |"
            )
    md.append("")
    md.append("## O diagnostic: game_type × season target rate")
    md.append("")
    md.append(
        f"- types with early(2019–22) vs late(2023–24) **relative-to-season-mean sign flip**: "
        f"`{gt_analysis['types_with_sign_flip']}`"
    )
    md.append(f"- reversal_confirmed: `{gt_analysis['reversal_confirmed']}`")
    md.append("")
    # rate wide table
    wide = gt_long.pivot_table(index="game_type", columns="season", values="rate")
    seasons = list(wide.columns)
    md.append("| game_type | " + " | ".join(str(int(s)) for s in seasons) + " |")
    md.append("|---|" + "|".join(["---:"] * len(seasons)) + "|")
    for gt in wide.index:
        vals = []
        for s in seasons:
            v = wide.loc[gt, s]
            vals.append("" if pd.isna(v) else f"{float(v):.4f}")
        md.append(f"| `{gt}` | " + " | ".join(vals) + " |")
    md.append("")
    if not gt_analysis["reversal_confirmed"]:
        md.append("- Experiment O training **skipped** (no confirmed sign-flip under the relative-to-season-mean rule).")
        md.append("")

    md.append("## Results: raw BSS by experiment × fold")
    md.append("")
    md.append(
        "| experiment | secondary_2023 raw BSS | Δ vs EXP-004(2023) | "
        "primary_2024 raw BSS | Δ vs EXP-004(2024) |"
    )
    md.append("|---|---:|---:|---:|---:|")
    md.append(
        f"| EXP-004 reference | {ref_004_2023:.6f} | 0.000000 | "
        f"{ref_004_2024:.6f} | 0.000000 |"
    )
    md.append(
        f"| season dropped D (ref, unweighted, 2023 only) | {ref_drop_2023:.6f} | "
        f"{ref_drop_2023 - ref_004_2023:+.6f} | — | — |"
    )
    for lab in labels:
        r23 = by.get((lab, "secondary_2023"))
        r24 = by.get((lab, "primary_2024"))
        if not r23 or not r24:
            continue
        md.append(
            f"| {lab} | {r23['raw_brier_skill_score']:.6f} | "
            f"{r23['raw_brier_skill_score'] - ref_004_2023:+.6f} | "
            f"{r24['raw_brier_skill_score']:.6f} | "
            f"{r24['raw_brier_skill_score'] - ref_004_2024:+.6f} |"
        )
    md.append("")
    md.append("## Detail rows")
    md.append("")
    md.append(
        "| experiment | fold | n_feat | asof_rel | raw BSS | official BSS | Brier | elapsed_s |"
    )
    md.append("|---|---|---:|---|---:|---:|---:|---:|")
    for r in results:
        md.append(
            f"| {r['label']} | {r['fold_id']} | {r['n_features']} | "
            f"{r['asof_relative']} | {r['raw_brier_skill_score']:.6f} | "
            f"{r['official_brier_skill_score']:.6f} | {r['brier_score']:.6f} | "
            f"{r['elapsed_seconds']:.1f} |"
        )
    md.append("")
    md.append("### M/N asof relative definition")
    md.append("")
    md.append(
        "- For each row with `season=S`, subtract mean(`rate_col`) over **train-fold rows with season < S**."
    )
    md.append("- Computed from train partition only; valid/test never enter the mean.")
    md.append("- If no prior season exists in train (e.g. S=2019), subtract `0.0`.")
    md.append(f"- Rate columns: `{asof_rate_cols(used0)}`")
    md.append("")
    md.append(f"- machine-readable: `{json_path}`")
    md.append("")

    md_path = out_dir / "unseen_id_experiments.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
