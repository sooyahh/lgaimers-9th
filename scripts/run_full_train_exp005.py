"""Experiment S (game_type ablation under L) then R full-train EXP-005 submit."""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
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
from src.utils import (
    append_experiment_log,
    ensure_dir,
    load_config,
    resolve_path,
    set_seed,
    to_json_text,
)

ID_DROP = ["pitcher_id", "batter_id"]
FULL_WEIGHTS = {**{int(k): float(v) for k, v in SEASON_WEIGHTS.items()}, 2024: 2.4}
# S uses EXP-004 weights on 2019-2023 only
S_WEIGHTS = {int(k): float(v) for k, v in SEASON_WEIGHTS.items()}


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


def run_holdout(
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
    weight_map: dict[int, float],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    fold = build_season_holdout(
        train_df, season_col, train_seasons, valid_seasons, name=label
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
    seasons = train_df.iloc[fold.train_idx][season_col].to_numpy()
    sw = np.asarray([float(weight_map[int(s)]) for s in seasons], dtype=float)
    model = fit_catboost(X_tr, y_tr, cats, params, seed, sample_weight=sw)
    pred, _ = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    return {
        "label": label,
        "train_seasons": train_seasons,
        "valid_seasons": valid_seasons,
        "n_features": len(used),
        "features_used": used,
        "has_game_type": "game_type" in used,
        "dropped_ids": ID_DROP,
        "sample_weight_by_season": {str(k): weight_map[k] for k in train_seasons},
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
        "train_target_mean": float(y_tr.mean()),
        "valid_target_rate": float(metrics["target_rate"]),
        "brier_score": float(metrics["brier_score"]),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }


def train_full(
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
    weight_map: dict[int, float],
    out_model_dir: Path,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cats = [c for c in categorical_cols if c in used]
    nums = [c for c in numeric_cols if c in used]
    # dummy valid for prepare_native_frames API (medians from train only)
    X_all = train_df[used]
    y_all = train_df[target_col]
    X_tr, _ = prepare_native_frames(
        X_all, X_all.iloc[:1].copy(), numeric_cols=nums, categorical_cols=cats, backend="catboost"
    )
    medians = {
        c: float(X_all[c].median()) if X_all[c].notna().any() else 0.0 for c in nums
    }
    # re-prep with explicit fill matching prepare_native_frames
    X_fit = X_all.copy()
    for c in nums:
        X_fit[c] = X_fit[c].fillna(medians[c])
    for c in cats:
        X_fit[c] = X_fit[c].astype("string").fillna("__NA__").astype(str)

    seasons = train_df[season_col].to_numpy()
    sw = np.asarray([float(weight_map[int(s)]) for s in seasons], dtype=float)
    model = fit_catboost(X_fit, y_all, cats, params, seed, sample_weight=sw)

    ensure_dir(out_model_dir)
    model_path = out_model_dir / "catboost_full.joblib"
    joblib.dump(model, model_path)
    meta = {
        "experiment_id": "EXP-005",
        "variant": label,
        "features_used": used,
        "numeric_cols": nums,
        "categorical_cols": cats,
        "exclude_cols": ["row_id", "home_win_expectancy", "away_win_expectancy", "li"],
        "dropped_ids": ID_DROP,
        "has_game_type": "game_type" in used,
        "numeric_medians_full_train_2019_2024": medians,
        # script.py compatibility alias
        "numeric_medians_primary_train_2019_2023": medians,
        "sample_weight_by_season": {str(k): float(v) for k, v in sorted(weight_map.items())},
        "train_seasons": sorted(int(s) for s in train_df[season_col].unique()),
        "n_train": int(len(train_df)),
        "model_filename": "catboost_full.joblib",
        "id_col": "row_id",
        "target_col": "control_success",
        "feature_names_": list(getattr(model, "feature_names_", used)),
    }
    meta_path = out_model_dir / "feature_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = float(time.perf_counter() - t0)
    return {
        "label": label,
        "n_features": len(used),
        "has_game_type": "game_type" in used,
        "n_train": int(len(train_df)),
        "train_target_mean": float(y_all.mean()),
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "sample_weight_by_season": meta["sample_weight_by_season"],
        "elapsed_seconds": elapsed,
        "feature_names_match": list(getattr(model, "feature_names_", [])) == used,
    }


SCRIPT_TEMPLATE = '''# script.py — EXP-005 CatBoost full-train inference for DACON code submission
"""Evaluation-server entrypoint.

Expected layout (server adds data/ and output/):
  model/catboost_full.joblib
  model/feature_meta.json
  script.py
  requirements.txt
  data/test.csv
  data/sample_submission.csv
  output/submission.csv  (created here)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
OUT_DIR = ROOT / "output"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
MODEL_PATH = MODEL_DIR / "catboost_full.joblib"
META_PATH = MODEL_DIR / "feature_meta.json"
OUT_PATH = OUT_DIR / "submission.csv"


def load_meta() -> dict:
    with META_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def prepare_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    used = list(meta["features_used"])
    numeric_cols = list(meta["numeric_cols"])
    categorical_cols = list(meta["categorical_cols"])
    medians = meta.get("numeric_medians_full_train_2019_2024") or meta[
        "numeric_medians_primary_train_2019_2023"
    ]
    missing = [c for c in used if c not in df.columns]
    if missing:
        raise ValueError(f"test is missing required feature columns: {missing}")
    X = df[used].copy()
    for col in numeric_cols:
        fill = float(medians.get(col, 0.0))
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill)
    for col in categorical_cols:
        X[col] = X[col].astype("string").fillna("__NA__").astype(str)
    return X


def merge_predictions(sub: pd.DataFrame, ids, preds) -> pd.DataFrame:
    pred_map = dict(zip(ids, preds))
    values = []
    n_missing = 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(float(cur))
        else:
            values.append(float(p))
    if n_missing:
        print(f" warning: {n_missing} row_id(s) missing predictions; kept placeholder")
    out = sub.copy()
    out[TARGET_COL] = values
    return out


def main() -> None:
    t_wall0 = time.perf_counter()
    print("Load meta / model...")
    t0 = time.perf_counter()
    meta = load_meta()
    model = joblib.load(MODEL_PATH)
    t_load = time.perf_counter() - t0
    print(f" OK. features={len(meta['features_used'])} load_s={t_load:.3f}")

    print("Load data...")
    t0 = time.perf_counter()
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    t_data = time.perf_counter() - t0
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission columns must start with ({ID_COL}, {TARGET_COL}), "
            f"got {list(sub.columns)}"
        )
    print(f" test={len(test)}  sample_submission={len(sub)} data_load_s={t_data:.3f}")

    print("Build features...")
    t0 = time.perf_counter()
    X = prepare_features(test, meta)
    ids = test[ID_COL].tolist()
    t_feat = time.perf_counter() - t0
    print(f" features={X.shape[1]} feature_prep_s={t_feat:.3f}")

    print("Inference...")
    t0 = time.perf_counter()
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        pos_index = classes.index(1)
    elif 1.0 in classes:
        pos_index = classes.index(1.0)
    else:
        raise ValueError(f"Positive class 1 not in model.classes_: {classes}")
    preds = np.asarray(model.predict_proba(X), dtype=float)[:, pos_index]
    preds = np.clip(preds, 0.0, 1.0)
    t_pred = time.perf_counter() - t0
    print(f" preds={len(preds)} mean={float(preds.mean()):.6f} predict_proba_s={t_pred:.3f}")

    print("Write submission...")
    t0 = time.perf_counter()
    out = merge_predictions(sub, ids, preds)
    os.makedirs(OUT_DIR, exist_ok=True)
    out[[ID_COL, TARGET_COL]].to_csv(OUT_PATH, index=False, encoding="utf-8")
    t_write = time.perf_counter() - t0
    t_wall = time.perf_counter() - t_wall0
    print(f" Saved: {OUT_PATH} (rows={len(out)}) write_s={t_write:.3f} wall_s={t_wall:.3f}")
    print(
        " TIMING_SUMMARY "
        f"model_load_s={t_load:.3f} data_load_s={t_data:.3f} "
        f"feature_prep_s={t_feat:.3f} predict_proba_s={t_pred:.3f} "
        f"write_s={t_write:.3f} wall_s={t_wall:.3f}"
    )


if __name__ == "__main__":
    main()
'''


def build_submit_zip(
    *,
    variant_dir: Path,
    requirements_src: Path,
    zip_path: Path,
    bundle_dir: Path,
) -> list[str]:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    ensure_dir(bundle_dir / "model")
    (bundle_dir / "script.py").write_text(SCRIPT_TEMPLATE, encoding="utf-8")
    shutil.copy2(requirements_src, bundle_dir / "requirements.txt")
    shutil.copy2(variant_dir / "catboost_full.joblib", bundle_dir / "model" / "catboost_full.joblib")
    shutil.copy2(variant_dir / "feature_meta.json", bundle_dir / "model" / "feature_meta.json")

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(bundle_dir / "script.py", arcname="script.py")
        zf.write(bundle_dir / "requirements.txt", arcname="requirements.txt")
        zf.write(
            bundle_dir / "model" / "catboost_full.joblib",
            arcname="model/catboost_full.joblib",
        )
        zf.write(
            bundle_dir / "model" / "feature_meta.json",
            arcname="model/feature_meta.json",
        )
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    return names


def smoke_submit(bundle_dir: Path, test_src: Path, sample_src: Path) -> dict[str, Any]:
    import subprocess
    import sys

    data = ensure_dir(bundle_dir / "data")
    out = ensure_dir(bundle_dir / "output")
    for p in out.glob("*"):
        p.unlink()
    shutil.copy2(test_src, data / "test.csv")
    shutil.copy2(sample_src, data / "sample_submission.csv")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "script.py"],
        cwd=str(bundle_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall = time.perf_counter() - t0
    sub = pd.read_csv(out / "submission.csv") if (out / "submission.csv").exists() else None
    return {
        "returncode": proc.returncode,
        "wall_s": wall,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-15:]),
        "n_rows": None if sub is None else int(len(sub)),
        "columns": None if sub is None else list(sub.columns),
        "pred_mean": None if sub is None else float(sub["control_success"].mean()),
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
    used_L = [c for c in used0 if c not in ID_DROP]
    cats_L = [c for c in cats0 if c not in ID_DROP]
    used_LO = [c for c in used_L if c != "game_type"]
    cats_LO = [c for c in cats_L if c != "game_type"]

    # --- S-1 from prior artifact ---
    s1_raw = 739.969723
    unseen_path = resolve_path("outputs/eda/unseen_id_experiments.json")
    if unseen_path.exists():
        unseen = json.loads(unseen_path.read_text(encoding="utf-8"))
        for r in unseen.get("runs", []):
            if r.get("label") == "L_drop_ids" and r.get("fold_id") == "primary_2024":
                s1_raw = float(r["raw_brier_skill_score"])
                break

    print("=== S-2: L + drop game_type, 2019-2023 → 2024 ===", flush=True)
    s2 = run_holdout(
        label="S2_L_plus_O",
        train_df=train_df,
        used=used_LO,
        numeric_cols=nums0,
        categorical_cols=cats_LO,
        target_col=target_col,
        season_col=season_col,
        train_seasons=[2019, 2020, 2021, 2022, 2023],
        valid_seasons=[2024],
        params=params,
        seed=seed,
        scale=scale,
        weight_map=S_WEIGHTS,
    )
    print(f"S-2 raw_bss={s2['raw_brier_skill_score']:.6f}", flush=True)

    # Decision rule for submit (mechanical): higher primary_2024 raw BSS under L
    choose_drop_game_type = s2["raw_brier_skill_score"] > s1_raw
    selected = "R_L_drop_game_type" if choose_drop_game_type else "R_L_keep_game_type"
    print(
        f"S compare: S1={s1_raw:.6f} S2={s2['raw_brier_skill_score']:.6f} "
        f"selected_for_submit={selected}",
        flush=True,
    )

    # --- R full train both variants ---
    models_root = ensure_dir(resolve_path("outputs/models/EXP-005"))
    print("=== R-keep: full 2019-2024, L, game_type kept ===", flush=True)
    r_keep = train_full(
        label="R_L_keep_game_type",
        train_df=train_df,
        used=used_L,
        numeric_cols=nums0,
        categorical_cols=cats_L,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        weight_map=FULL_WEIGHTS,
        out_model_dir=ensure_dir(models_root / "R_L_keep_game_type"),
    )
    print(f"R-keep done n_train={r_keep['n_train']} elapsed={r_keep['elapsed_seconds']:.1f}s", flush=True)

    print("=== R-drop: full 2019-2024, L, game_type dropped ===", flush=True)
    r_drop = train_full(
        label="R_L_drop_game_type",
        train_df=train_df,
        used=used_LO,
        numeric_cols=nums0,
        categorical_cols=cats_LO,
        target_col=target_col,
        season_col=season_col,
        params=params,
        seed=seed,
        weight_map=FULL_WEIGHTS,
        out_model_dir=ensure_dir(models_root / "R_L_drop_game_type"),
    )
    print(f"R-drop done n_train={r_drop['n_train']} elapsed={r_drop['elapsed_seconds']:.1f}s", flush=True)

    # Copy selected to EXP-005 primary paths
    selected_dir = models_root / selected
    primary_dir = ensure_dir(models_root)
    shutil.copy2(selected_dir / "catboost_full.joblib", primary_dir / "catboost_full.joblib")
    shutil.copy2(selected_dir / "feature_meta.json", primary_dir / "feature_meta.json")

    req_src = resolve_path("outputs/submissions/EXP-004_bundle/requirements.txt")
    zip_path = resolve_path("outputs/submissions/EXP-005_submit.zip")
    bundle_dir = resolve_path("outputs/submissions/EXP-005_bundle")
    zip_names = build_submit_zip(
        variant_dir=selected_dir,
        requirements_src=req_src,
        zip_path=zip_path,
        bundle_dir=bundle_dir,
    )
    smoke = smoke_submit(
        bundle_dir,
        resolve_path("data/raw/test.csv"),
        resolve_path("data/raw/sample_submission.csv"),
    )
    print(f"submit smoke rc={smoke['returncode']} rows={smoke['n_rows']}", flush=True)

    # predictions stub summary
    pred_dir = ensure_dir(resolve_path("outputs/predictions/EXP-005"))
    summary = {
        "experiment_id": "EXP-005",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "catboost",
        "seed": seed,
        "params": params,
        "full_train": True,
        "train_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
        "sample_weight_by_season": {str(k): float(v) for k, v in sorted(FULL_WEIGHTS.items())},
        "S": {
            "S1_L_keep_game_type_primary_2024_raw_bss": s1_raw,
            "S2_L_drop_game_type": s2,
            "selected_for_submit": selected,
            "selection_rule": "higher raw BSS on primary_2024 under L (+/- game_type)",
        },
        "R": {"keep": r_keep, "drop": r_drop},
        "submit_zip": str(zip_path),
        "zip_entries": zip_names,
        "smoke": smoke,
    }
    (pred_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    append_experiment_log(
        config["output"]["experiments_path"],
        {
            "experiment_id": "EXP-005",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "catboost_full_train_2019_2024_L_weights_selected_by_S",
            "data_version": config.get("project", {}).get("data_version", ""),
            "features": to_json_text(
                {
                    "n": (r_drop if choose_drop_game_type else r_keep)["n_features"],
                    "drop_ids": ID_DROP,
                    "drop_game_type": choose_drop_game_type,
                    "selected_variant": selected,
                }
            ),
            "model": "catboost",
            "params": to_json_text({**params, "sample_weight_by_season": FULL_WEIGHTS}),
            "cv_strategy": to_json_text(
                {
                    "final": "full_train_2019_2024_no_holdout",
                    "selection": "S primary_2024 under L +/- game_type",
                }
            ),
            "metric": "brier_skill_score",
            "metric_direction": "maximize",
            "cv_score": s2["raw_brier_skill_score"] if choose_drop_game_type else s1_raw,
            "cv_std": "",
            "holdout_score": "",
            "public_lb_score": "",
            "cv_lb_gap": "",
            "seed": seed,
            "notes": to_json_text(
                {
                    "S1_raw_bss": s1_raw,
                    "S2_raw_bss": s2["raw_brier_skill_score"],
                    "selected": selected,
                    "submit_zip": str(zip_path),
                    "decision": "results_only_no_adopt_reject",
                }
            ),
            "next_action": "Submit EXP-005_submit.zip if desired; no adopt/reject in docs.",
            "status": "success",
            "error": "",
        },
    )

    # markdown
    md = [
        "# Full-train experiments (S → R → EXP-005)",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        "- **no adopt/reject judgment — results only**",
        "",
        "## Experiment S (config check on primary_2024)",
        "",
        "Common: CatBoost; train `2019–2023`; valid `2024`; EXP-004 season weights; **L** (drop `pitcher_id`, `batter_id`).",
        "",
        "| run | game_type | raw BSS | official BSS | Brier |",
        "|---|---|---:|---:|---:|",
        f"| S-1 L only (from unseen_id_experiments L_drop_ids) | kept | {s1_raw:.6f} | {s1_raw:.6f} | — |",
        (
            f"| S-2 L + drop game_type | dropped | {s2['raw_brier_skill_score']:.6f} | "
            f"{s2['official_brier_skill_score']:.6f} | {s2['brier_score']:.6f} |"
        ),
        "",
        f"- Δ (S-2 − S-1) raw BSS: `{s2['raw_brier_skill_score'] - s1_raw:+.6f}`",
        f"- mechanical select for submit (higher raw BSS): **`{selected}`**",
        "",
        "### S-2 details",
        "",
        f"- n_train / n_valid: `{s2['n_train']:,}` / `{s2['n_valid']:,}`",
        f"- n_features: `{s2['n_features']}`",
        f"- elapsed_s: `{s2['elapsed_seconds']:.1f}`",
        "",
        "## Experiment R (full train 2019–2024)",
        "",
        f"- weights: `{json.dumps({str(k): v for k, v in sorted(FULL_WEIGHTS.items())})}`",
        "- L applied on both variants; no holdout metrics",
        "",
        "| variant | game_type | n_features | n_train | model path | elapsed_s |",
        "|---|---|---:|---:|---|---:|",
        (
            f"| R_L_keep_game_type | kept | {r_keep['n_features']} | {r_keep['n_train']:,} | "
            f"`{r_keep['model_path']}` | {r_keep['elapsed_seconds']:.1f} |"
        ),
        (
            f"| R_L_drop_game_type | dropped | {r_drop['n_features']} | {r_drop['n_train']:,} | "
            f"`{r_drop['model_path']}` | {r_drop['elapsed_seconds']:.1f} |"
        ),
        "",
        "## EXP-005 submit.zip",
        "",
        f"- selected variant: `{selected}`",
        f"- zip: `{zip_path}`",
        f"- zip entries: `{zip_names}`",
        f"- smoke test (5-row official test): rc=`{smoke['returncode']}`, "
        f"rows=`{smoke['n_rows']}`, cols=`{smoke['columns']}`, "
        f"pred_mean=`{smoke['pred_mean']}`, wall_s=`{smoke['wall_s']:.3f}`",
        "",
        f"- summary: `{pred_dir / 'summary.json'}`",
        "",
    ]
    md_path = ensure_dir(resolve_path("outputs/eda")) / "full_train_experiments.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    json_path = resolve_path("outputs/eda/full_train_experiments.json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
