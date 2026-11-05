"""Overnight A/B/C after EXP-006: seed variance, weight grid, excluded-feature recheck.

Does not modify EXP-000..006 artifacts. No adopt/reject; no submission.
"""

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
WEIGHT_SCHEDULES: dict[str, dict[int, float]] = {
    "B-1_current": {2019: 1.0, 2020: 1.2, 2021: 1.4, 2022: 1.6, 2023: 2.0},
    "B-2_mild": {2019: 1.0, 2020: 1.1, 2021: 1.2, 2022: 1.3, 2023: 1.5},
    "B-3_aggressive": {2019: 1.0, 2020: 1.3, 2021: 1.7, 2022: 2.2, 2023: 3.0},
    "B-4_extreme": {2019: 0.5, 2020: 0.7, 2021: 1.0, 2022: 1.5, 2023: 3.0},
}
CANDIDATE_EXCLUDED = ["li", "home_win_expectancy", "away_win_expectancy"]
# data_description.md §주자 및 상황 중요도: all three labeled as pre-pitch (투구 직전).
FEATURE_SAFETY = {
    "li": {
        "pre_pitch": True,
        "source": "data_description.md",
        "quote": "투구 직전 상황 중요도 지표입니다.",
    },
    "home_win_expectancy": {
        "pre_pitch": True,
        "source": "data_description.md",
        "quote": "투구 직전 경기 상황에서 홈 팀의 기대 승률입니다.",
    },
    "away_win_expectancy": {
        "pre_pitch": True,
        "source": "data_description.md",
        "quote": "투구 직전 경기 상황에서 원정 팀의 기대 승률입니다.",
    },
}


def make_weights(seasons: np.ndarray, weight_map: dict[int, float]) -> np.ndarray:
    return np.asarray([float(weight_map[int(s)]) for s in seasons], dtype=float)


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
        f"raw_bss={metrics['raw_brier_skill_score']:.6f} "
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return {
        "label": label,
        "seed": seed,
        "n_features": len(used),
        "features_used": used,
        "sample_weight_by_season": {str(k): float(weight_map[k]) for k in HOLD_TRAIN},
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        "brier_score": float(metrics["brier_score"]),
        "elapsed_seconds": float(elapsed),
    }


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


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def task_a(
    train_df: pd.DataFrame,
    used: list[str],
    nums: list[str],
    cats: list[str],
    target_col: str,
    season_col: str,
    params: dict[str, Any],
    scale: float,
    eda_dir: Path,
) -> dict[str, Any]:
    print("=== Task A: seed variance ===", flush=True)
    rows = []
    for seed in SEEDS:
        r = run_holdout(
            label=f"A_seed_{seed}",
            train_df=train_df,
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            target_col=target_col,
            season_col=season_col,
            params=params,
            seed=seed,
            scale=scale,
            weight_map=DEFAULT_WEIGHTS,
        )
        rows.append(r)

    scores = np.asarray([r["raw_brier_skill_score"] for r in rows], dtype=float)
    mean_bss = float(np.mean(scores))
    std_bss = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    payload = {
        "task": "A_seed_variance",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "setup": {
            "model": "catboost",
            "params": params,
            "features_n": len(used),
            "drop_ids": ID_DROP,
            "game_type": "kept",
            "train_seasons": HOLD_TRAIN,
            "valid_seasons": HOLD_VALID,
            "sample_weight_by_season": DEFAULT_WEIGHTS,
            "seeds": SEEDS,
            "note": "EXP-005 holdout setting; seed only varies",
        },
        "runs": rows,
        "summary": {
            "mean_raw_bss": mean_bss,
            "std_raw_bss": std_bss,
            "min_raw_bss": float(np.min(scores)),
            "max_raw_bss": float(np.max(scores)),
            "range_raw_bss": float(np.max(scores) - np.min(scores)),
        },
    }
    md = [
        "# Seed variance (Task A)",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "- setup: EXP-005 holdout (CatBoost L + game_type kept + current season weights; train 2019–2023 → valid 2024)",
        f"- params: `{json.dumps(params, ensure_ascii=False)}`",
        f"- n_features: `{len(used)}`",
        "- **results only — no adopt/reject**",
        "",
        "| seed | raw BSS | official BSS | Brier | elapsed_s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['seed']} | {r['raw_brier_skill_score']:.6f} | "
            f"{r['official_brier_skill_score']:.6f} | {r['brier_score']:.6f} | "
            f"{r['elapsed_seconds']:.1f} |"
        )
    md.extend(
        [
            "",
            f"- mean raw BSS: `{mean_bss:.6f}`",
            f"- std raw BSS (sample, ddof=1): `{std_bss:.6f}`",
            f"- min / max: `{float(np.min(scores)):.6f}` / `{float(np.max(scores)):.6f}`",
            f"- range (max−min): `{float(np.max(scores) - np.min(scores)):.6f}`",
            "",
        ]
    )
    write_text(eda_dir / "seed_variance.md", "\n".join(md))
    write_json(eda_dir / "seed_variance.json", payload)
    return payload


def task_b(
    train_df: pd.DataFrame,
    used: list[str],
    nums: list[str],
    cats: list[str],
    target_col: str,
    season_col: str,
    params: dict[str, Any],
    scale: float,
    seed: int,
    eda_dir: Path,
    baseline_raw: float | None,
) -> dict[str, Any]:
    print("=== Task B: season weight schedule grid ===", flush=True)
    rows = []
    for name, wmap in WEIGHT_SCHEDULES.items():
        r = run_holdout(
            label=name,
            train_df=train_df,
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            target_col=target_col,
            season_col=season_col,
            params=params,
            seed=seed,
            scale=scale,
            weight_map=wmap,
        )
        r["schedule_name"] = name
        r["delta_vs_B1"] = None
        rows.append(r)

    b1 = next(r for r in rows if r["schedule_name"] == "B-1_current")
    b1_raw = float(b1["raw_brier_skill_score"])
    for r in rows:
        r["delta_vs_B1"] = float(r["raw_brier_skill_score"] - b1_raw)

    payload = {
        "task": "B_weight_schedule_grid",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "setup": {
            "model": "catboost",
            "params": params,
            "features_n": len(used),
            "seed": seed,
            "train_seasons": HOLD_TRAIN,
            "valid_seasons": HOLD_VALID,
            "schedules": {k: v for k, v in WEIGHT_SCHEDULES.items()},
        },
        "runs": rows,
        "baseline_B1_raw_bss": b1_raw,
        "reference_seed42_from_A": baseline_raw,
    }
    md = [
        "# Season weight schedule grid (Task B)",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- setup: CatBoost L + game_type kept; seed=`{seed}`; train 2019–2023 → valid 2024",
        f"- params: `{json.dumps(params, ensure_ascii=False)}`",
        "- **results only — no adopt/reject**",
        "",
        "| schedule | weights | raw BSS | Δ vs B-1 | official BSS | Brier | elapsed_s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        w = WEIGHT_SCHEDULES[r["schedule_name"]]
        md.append(
            f"| {r['schedule_name']} | `{w}` | {r['raw_brier_skill_score']:.6f} | "
            f"{r['delta_vs_B1']:+.6f} | {r['official_brier_skill_score']:.6f} | "
            f"{r['brier_score']:.6f} | {r['elapsed_seconds']:.1f} |"
        )
    md.extend(["", f"- B-1 (current) raw BSS: `{b1_raw:.6f}`", ""])
    write_text(eda_dir / "weight_schedule_grid.md", "\n".join(md))
    write_json(eda_dir / "weight_schedule_grid.json", payload)
    return payload


def task_c(
    train_df: pd.DataFrame,
    base_exclude: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    params: dict[str, Any],
    scale: float,
    seed: int,
    eda_dir: Path,
) -> dict[str, Any]:
    print("=== Task C: excluded feature recheck ===", flush=True)
    safety_lines = [
        "# Excluded feature recheck (Task C)",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- question: are `li` / `home_win_expectancy` / `away_win_expectancy` pre-pitch or post-target?",
        "",
        "## Safety check (`data_description.md`)",
        "",
        "| feature | pre-pitch? | quote |",
        "|---|---|---|",
    ]
    all_safe = True
    for feat in CANDIDATE_EXCLUDED:
        info = FEATURE_SAFETY[feat]
        all_safe = all_safe and bool(info["pre_pitch"])
        safety_lines.append(
            f"| `{feat}` | {'yes' if info['pre_pitch'] else 'NO — skip'} | {info['quote']} |"
        )
    safety_lines.extend(
        [
            "",
            "- conclusion: all three are labeled **투구 직전** (pre-pitch situation metrics).",
            "- therefore: **no skip** — proceed with 4 add-back experiments.",
            "- baseline features remain EXP-005 L set (42); variants add the named columns.",
            "- seed / weights: seed=42, B-1 current schedule.",
            "- **results only — no adopt/reject**",
            "",
        ]
    )

    # baseline exclude from config; reinstate candidates selectively
    variants: list[tuple[str, list[str]]] = [
        ("C-1_add_li", ["li"]),
        ("C-2_add_home_win_expectancy", ["home_win_expectancy"]),
        ("C-3_add_away_win_expectancy", ["away_win_expectancy"]),
        ("C-4_add_all_three", list(CANDIDATE_EXCLUDED)),
    ]

    # reference: current 42-feature baseline (same as A seed 42 / B-1)
    used_ref, nums_ref, cats_ref = base_feature_lists(
        train_df, target_col, base_exclude, categorical_cols
    )
    ref = run_holdout(
        label="C-0_baseline_42",
        train_df=train_df,
        used=used_ref,
        numeric_cols=nums_ref,
        categorical_cols=cats_ref,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        scale=scale,
        weight_map=DEFAULT_WEIGHTS,
    )
    ref["variant"] = "C-0_baseline_42"
    ref["added_features"] = []

    rows = [ref]
    if all_safe:
        for name, add_feats in variants:
            # remove candidates from exclude except those not being added
            still_exclude = [c for c in base_exclude if c not in add_feats]
            used, nums, cats = base_feature_lists(
                train_df, target_col, still_exclude, categorical_cols
            )
            for f in add_feats:
                assert f in used, f"expected {f} in used after reinstate"
            r = run_holdout(
                label=name,
                train_df=train_df,
                used=used,
                numeric_cols=nums,
                categorical_cols=cats,
                target_col=target_col,
                season_col=season_col,
                params=params,
                seed=seed,
                scale=scale,
                weight_map=DEFAULT_WEIGHTS,
            )
            r["variant"] = name
            r["added_features"] = add_feats
            r["delta_vs_baseline"] = float(
                r["raw_brier_skill_score"] - ref["raw_brier_skill_score"]
            )
            rows.append(r)
    else:
        safety_lines.append("- experiments skipped due to post-pitch / leakage concern.")
        safety_lines.append("")

    ref_raw = float(ref["raw_brier_skill_score"])
    for r in rows:
        if "delta_vs_baseline" not in r:
            r["delta_vs_baseline"] = float(r["raw_brier_skill_score"] - ref_raw)

    payload = {
        "task": "C_excluded_feature_recheck",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "feature_safety": FEATURE_SAFETY,
        "all_pre_pitch": all_safe,
        "setup": {
            "model": "catboost",
            "params": params,
            "seed": seed,
            "sample_weight_by_season": DEFAULT_WEIGHTS,
            "train_seasons": HOLD_TRAIN,
            "valid_seasons": HOLD_VALID,
            "id_drop": ID_DROP,
        },
        "runs": rows,
        "baseline_raw_bss": ref_raw,
    }

    md = list(safety_lines)
    md.extend(
        [
            "## Results",
            "",
            "| variant | added | n_feat | raw BSS | Δ vs C-0 | official BSS | Brier | elapsed_s |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        added = ",".join(r.get("added_features") or []) or "—"
        md.append(
            f"| {r['variant']} | `{added}` | {r['n_features']} | "
            f"{r['raw_brier_skill_score']:.6f} | {r['delta_vs_baseline']:+.6f} | "
            f"{r['official_brier_skill_score']:.6f} | {r['brier_score']:.6f} | "
            f"{r['elapsed_seconds']:.1f} |"
        )
    md.append("")
    write_text(eda_dir / "excluded_feature_recheck.md", "\n".join(md))
    write_json(eda_dir / "excluded_feature_recheck.json", payload)
    return payload


def write_summary(a: dict, b: dict, c: dict, eda_dir: Path) -> None:
    seed_std = float(a["summary"]["std_raw_bss"])
    seed_mean = float(a["summary"]["mean_raw_bss"])
    b1 = float(b["baseline_B1_raw_bss"])
    c0 = float(c["baseline_raw_bss"])

    def noise_flag(delta: float) -> str:
        if seed_std <= 0:
            return "n/a"
        if abs(delta) > seed_std:
            return f"|Δ| > seed_std ({seed_std:.6f})"
        return f"|Δ| ≤ seed_std ({seed_std:.6f})"

    md = [
        "# Overnight summary (A / B / C after EXP-006)",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- common: CatBoost EXP-005 holdout (L, game_type kept); train 2019–2023 → valid 2024; raw BSS",
        "- **results only — no adopt/reject; no submission**",
        "",
        "## Seed noise reference (Task A)",
        "",
        f"- seeds: `{SEEDS}`",
        f"- mean raw BSS: `{seed_mean:.6f}`",
        f"- **seed std (sample): `{seed_std:.6f}`**",
        f"- range (max−min): `{a['summary']['range_raw_bss']:.6f}`",
        "",
        "| seed | raw BSS |",
        "|---:|---:|",
    ]
    for r in a["runs"]:
        md.append(f"| {r['seed']} | {r['raw_brier_skill_score']:.6f} |")

    md.extend(
        [
            "",
            "## Task B — weight schedules vs B-1",
            "",
            f"- B-1 raw BSS: `{b1:.6f}`",
            f"- compare Δ to seed_std `{seed_std:.6f}`",
            "",
            "| schedule | raw BSS | Δ vs B-1 | vs seed noise |",
            "|---|---:|---:|---|",
        ]
    )
    for r in b["runs"]:
        d = float(r["delta_vs_B1"])
        md.append(
            f"| {r['schedule_name']} | {r['raw_brier_skill_score']:.6f} | "
            f"{d:+.6f} | {noise_flag(d)} |"
        )

    md.extend(
        [
            "",
            "## Task C — reinstate excluded features vs C-0",
            "",
            f"- C-0 (42-feat baseline) raw BSS: `{c0:.6f}`",
            f"- safety: all of {CANDIDATE_EXCLUDED} are pre-pitch per data_description.md",
            f"- compare Δ to seed_std `{seed_std:.6f}`",
            "",
            "| variant | n_feat | raw BSS | Δ vs C-0 | vs seed noise |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for r in c["runs"]:
        d = float(r["delta_vs_baseline"])
        md.append(
            f"| {r['variant']} | {r['n_features']} | {r['raw_brier_skill_score']:.6f} | "
            f"{d:+.6f} | {noise_flag(d)} |"
        )

    md.extend(
        [
            "",
            "## Artifact paths",
            "",
            "- `outputs/eda/seed_variance.md`",
            "- `outputs/eda/weight_schedule_grid.md`",
            "- `outputs/eda/excluded_feature_recheck.md`",
            "- `outputs/eda/overnight_summary.md`",
            "",
        ]
    )
    write_text(eda_dir / "overnight_summary.md", "\n".join(md))
    write_json(
        eda_dir / "overnight_summary.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seed_std": seed_std,
            "seed_mean": seed_mean,
            "A": a["summary"],
            "B": [
                {
                    "name": r["schedule_name"],
                    "raw": r["raw_brier_skill_score"],
                    "delta_vs_B1": r["delta_vs_B1"],
                }
                for r in b["runs"]
            ],
            "C": [
                {
                    "name": r["variant"],
                    "n_features": r["n_features"],
                    "raw": r["raw_brier_skill_score"],
                    "delta_vs_baseline": r["delta_vs_baseline"],
                }
                for r in c["runs"]
            ],
        },
    )


def main() -> None:
    # Guard: do not start while EXP-006 tuning script might still be running.
    # EXP-006 is expected complete; we only check for submit zip presence.
    zip006 = resolve_path("outputs/submissions/EXP-006_submit.zip")
    if not zip006.exists():
        raise SystemExit(
            "EXP-006_submit.zip missing — wait for EXP-006 completion before overnight A/B/C."
        )

    config = load_config("configs/baseline.yaml")
    params = dict(config.get("models", {}).get("params", {}).get("catboost", {}))
    scale = float(resolve_metric_settings(config.get("task", {}))["scale"])
    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    target_col = data_cfg["target_col"]
    season_col = data_cfg["season_col"]
    exclude_cols = list(feat_cfg.get("exclude_cols") or [])
    categorical_cols = list(feat_cfg.get("categorical_cols") or [])
    seed_default = int(config.get("runtime", {}).get("seed", 42))

    train_df = load_competition_data(config)["train"]
    used, nums, cats = base_feature_lists(
        train_df, target_col, exclude_cols, categorical_cols
    )
    assert len(used) == 42, f"expected 42 features, got {len(used)}"

    eda_dir = ensure_dir(resolve_path("outputs/eda"))
    print(
        f"Overnight A/B/C start n_train_rows={len(train_df):,} n_feat={len(used)}",
        flush=True,
    )

    a = task_a(
        train_df, used, nums, cats, target_col, season_col, params, scale, eda_dir
    )
    baseline_a42 = next(
        r["raw_brier_skill_score"] for r in a["runs"] if r["seed"] == 42
    )

    b = task_b(
        train_df,
        used,
        nums,
        cats,
        target_col,
        season_col,
        params,
        scale,
        seed_default,
        eda_dir,
        baseline_a42,
    )

    c = task_c(
        train_df,
        exclude_cols,
        categorical_cols,
        target_col,
        season_col,
        params,
        scale,
        seed_default,
        eda_dir,
    )

    write_summary(a, b, c, eda_dir)
    print("Overnight A/B/C complete.", flush=True)


if __name__ == "__main__":
    main()
