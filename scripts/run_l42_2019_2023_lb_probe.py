"""LB-PROBE-L42-2019-2023-001: L42 features, train 2019-2023 only.

Single change vs EXP-004: drop pitcher_id and batter_id (44 → L42).
Same 2019-2023 window and EXP-004 season weights. Not an adopt/reject experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exp004_catboost_weighted import fit_catboost
from scripts.run_full_train_exp005 import ID_DROP
from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from scripts.tree_exp_common import (
    build_feature_lists,
    positive_class_proba,
    prepare_native_frames,
)
from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.submission import validate_submission
from src.utils import (
    append_experiment_log,
    ensure_dir,
    load_config,
    resolve_path,
    set_seed,
    to_json_text,
)

EXPERIMENT_ID = "LB-PROBE-L42-2019-2023-001"
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
VALID_SEASONS = [2024]
WEIGHTS = {int(k): float(v) for k, v in SEASON_WEIGHTS.items()}
CANONICAL_L42_2024_RAW_BSS = 739.9697230569768  # unseen_id L_drop_ids / EXP-005 S1
CANONICAL_EXP004_2024_RAW_BSS = 578.1844155969828
CANONICAL_EXP004_2024_BRIER = 0.24836258203719846
SANITY_ABS_TOL = 1.0
MODEL_FILENAME = "catboost_primary.joblib"

PROTECTED = [
    ROOT / "outputs/predictions/EXP-004/summary.json",
    ROOT / "outputs/predictions/EXP-004/fold_metrics.csv",
    ROOT / "outputs/predictions/EXP-004/oof_holdout_predictions.csv",
    ROOT / "outputs/models/EXP-004/catboost_primary.joblib",
    ROOT / "outputs/submissions/EXP-004_submit.zip",
    ROOT / "outputs/predictions/EXP-005/summary.json",
    ROOT / "outputs/models/EXP-005/catboost_full.joblib",
    ROOT / "outputs/models/EXP-005/feature_meta.json",
    ROOT / "outputs/models/EXP-005/R_L_keep_game_type/catboost_full.joblib",
    ROOT / "outputs/models/EXP-005/R_L_keep_game_type/feature_meta.json",
    ROOT / "outputs/submissions/EXP-005_submit.zip",
]

SCRIPT_TEMPLATE = '''# script.py — LB-PROBE-L42-2019-2023-001 CatBoost inference for DACON code submission
"""Evaluation-server entrypoint.

Expected layout (server adds data/ and output/):
  model/catboost_primary.joblib
  model/feature_meta.json
  script.py
  requirements.txt
  data/test.csv
  data/sample_submission.csv
  output/submission.csv  (created here)

Features are row-local: numeric NA fill uses train-fold medians baked into
feature_meta.json. No test-row aggregates, rolling stats, frequency counts,
or target encoding.
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
MODEL_PATH = MODEL_DIR / "catboost_primary.joblib"
META_PATH = MODEL_DIR / "feature_meta.json"
OUT_PATH = OUT_DIR / "submission.csv"


def load_meta() -> dict:
    with META_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def prepare_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    used = list(meta["features_used"])
    numeric_cols = list(meta["numeric_cols"])
    categorical_cols = list(meta["categorical_cols"])
    medians = meta["numeric_medians_primary_train_2019_2023"]
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
    t_wall = time.perf_counter() - t_wall0
    t_write = time.perf_counter() - t0
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


def feature_list_hash(names: list[str]) -> str:
    return hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path),
        "exists": True,
        "nbytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def snapshot_protected() -> dict[str, dict[str, Any]]:
    return {str(path): file_fingerprint(path) for path in PROTECTED}


def assert_protected_unchanged(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> None:
    for key, prev in before.items():
        got = after.get(key) or {}
        if prev.get("sha256") != got.get("sha256") or prev.get("nbytes") != got.get("nbytes"):
            raise RuntimeError(f"Protected artifact changed: {key}")


def l42_feature_lists(
    columns: list[str],
    *,
    target_col: str,
    exclude_cols: list[str],
    categorical_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    used0, nums0, cats0 = build_feature_lists(
        columns,
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols,
    )
    used = [c for c in used0 if c not in ID_DROP]
    nums = [c for c in nums0 if c not in ID_DROP]
    cats = [c for c in cats0 if c not in ID_DROP]
    return used, nums, cats


def assert_l42_contract(used: list[str], cats: list[str]) -> None:
    if "pitcher_id" in used or "batter_id" in used:
        raise AssertionError("L42 contract failed: raw IDs still present")
    if "game_type" not in used:
        raise AssertionError("L42 contract failed: game_type missing")
    if len(used) != 42:
        raise AssertionError(f"L42 contract failed: expected 42 features, got {len(used)}")
    if "pitcher_id" in cats or "batter_id" in cats:
        raise AssertionError("L42 contract failed: raw IDs still categorical")
    if "game_type" not in cats:
        raise AssertionError("L42 contract failed: game_type not categorical")


def assert_no_2024_train_leakage(train_seasons: list[int], seasons: np.ndarray) -> None:
    if max(train_seasons) != 2023:
        raise AssertionError(f"train season max must be 2023, got {max(train_seasons)}")
    if 2024 in train_seasons:
        raise AssertionError("2024 leaked into train_seasons")
    unique = {int(s) for s in seasons}
    if 2024 in unique:
        raise AssertionError("2024 leaked into training rows")
    extra = unique - set(train_seasons)
    if extra:
        raise AssertionError(f"unexpected train seasons present: {sorted(extra)}")


def numeric_medians(frame: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in numeric_cols:
        out[col] = float(frame[col].median()) if frame[col].notna().any() else 0.0
    return out


def run_primary_holdout(
    *,
    train_df: pd.DataFrame,
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    id_col: str,
    params: dict[str, Any],
    seed: int,
    scale: float,
    weight_map: dict[int, float],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    fold = build_season_holdout(
        train_df, season_col, TRAIN_SEASONS, VALID_SEASONS, name="primary_2024"
    )
    cats = [c for c in categorical_cols if c in used]
    nums = [c for c in numeric_cols if c in used]
    X_tr_raw = train_df.iloc[fold.train_idx][used]
    y_tr = train_df.iloc[fold.train_idx][target_col]
    X_va_raw = train_df.iloc[fold.valid_idx][used]
    y_va = train_df.iloc[fold.valid_idx][target_col]
    seasons = train_df.iloc[fold.train_idx][season_col].to_numpy()
    assert_l42_contract(used, cats)
    assert_no_2024_train_leakage(TRAIN_SEASONS, seasons)
    medians = numeric_medians(X_tr_raw, nums)
    X_tr, X_va = prepare_native_frames(
        X_tr_raw, X_va_raw, numeric_cols=nums, categorical_cols=cats, backend="catboost"
    )
    missing_w = sorted({int(s) for s in seasons if int(s) not in weight_map})
    if missing_w:
        raise ValueError(f"sample_weight_by_season missing seasons {missing_w}")
    sw = np.asarray([float(weight_map[int(s)]) for s in seasons], dtype=float)
    model = fit_catboost(X_tr, y_tr, cats, params, seed, sample_weight=sw)
    pred, pos_index = positive_class_proba(model, X_va)
    if not np.isfinite(pred).all():
        raise AssertionError("holdout predictions are not finite")
    if ((pred < 0) | (pred > 1)).any():
        raise AssertionError("holdout predictions outside [0, 1]")
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    ids = train_df.iloc[fold.valid_idx][id_col].to_numpy()
    pred_frame = pd.DataFrame(
        {
            id_col: ids,
            season_col: train_df.iloc[fold.valid_idx][season_col].to_numpy(),
            target_col: y_va.to_numpy(),
            "prediction": pred,
            "fold_id": "primary_2024",
        }
    )
    elapsed = float(time.perf_counter() - t0)
    return {
        "model": model,
        "medians": medians,
        "used": used,
        "numeric_cols": nums,
        "categorical_cols": cats,
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
        "train_seasons": list(TRAIN_SEASONS),
        "valid_seasons": list(VALID_SEASONS),
        "train_season_max": int(max(int(s) for s in seasons)),
        "train_target_mean": float(y_tr.mean()),
        "valid_target_rate": float(metrics["target_rate"]),
        "brier_score": float(metrics["brier_score"]),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        "prediction_mean": float(np.mean(pred)),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
        "positive_class_index": int(pos_index),
        "elapsed_seconds": elapsed,
        "pred_frame": pred_frame,
        "sample_weight_by_season": {str(k): float(weight_map[k]) for k in TRAIN_SEASONS},
        "feature_list_hash": feature_list_hash(used),
        "seed": seed,
        "params": params,
    }


def save_model_bundle(holdout: dict[str, Any], model_dir: Path) -> dict[str, str]:
    ensure_dir(model_dir)
    model_path = model_dir / MODEL_FILENAME
    joblib.dump(holdout["model"], model_path)
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "features_used": holdout["used"],
        "numeric_cols": holdout["numeric_cols"],
        "categorical_cols": holdout["categorical_cols"],
        "exclude_cols": ["row_id", "home_win_expectancy", "away_win_expectancy", "li"],
        "dropped_ids": list(ID_DROP),
        "has_game_type": True,
        "numeric_medians_primary_train_2019_2023": holdout["medians"],
        "sample_weight_by_season": holdout["sample_weight_by_season"],
        "train_seasons": holdout["train_seasons"],
        "n_train": holdout["n_train"],
        "model_filename": MODEL_FILENAME,
        "id_col": "row_id",
        "target_col": "control_success",
        "seed": holdout["seed"],
        "params": holdout["params"],
        "feature_list_hash": holdout["feature_list_hash"],
        "feature_names_": list(getattr(holdout["model"], "feature_names_", holdout["used"])),
    }
    meta_path = model_dir / "feature_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "meta_path": str(meta_path)}


def build_submit_zip(*, model_dir: Path, requirements_src: Path, zip_path: Path, bundle_dir: Path) -> list[str]:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    ensure_dir(bundle_dir / "model")
    (bundle_dir / "script.py").write_text(SCRIPT_TEMPLATE, encoding="utf-8")
    shutil.copy2(requirements_src, bundle_dir / "requirements.txt")
    shutil.copy2(model_dir / MODEL_FILENAME, bundle_dir / "model" / MODEL_FILENAME)
    shutil.copy2(model_dir / "feature_meta.json", bundle_dir / "model" / "feature_meta.json")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(bundle_dir / "script.py", arcname="script.py")
        zf.write(bundle_dir / "requirements.txt", arcname="requirements.txt")
        zf.write(bundle_dir / "model" / MODEL_FILENAME, arcname=f"model/{MODEL_FILENAME}")
        zf.write(bundle_dir / "model" / "feature_meta.json", arcname="model/feature_meta.json")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    expected = {"script.py", "requirements.txt", f"model/{MODEL_FILENAME}", "model/feature_meta.json"}
    if set(names) != expected:
        raise AssertionError(f"unexpected zip entries: {names}")
    return names


def _write_head_csv(src: Path, dst: Path, n: int) -> int:
    df = pd.read_csv(src, encoding="utf-8-sig", nrows=n)
    ensure_dir(dst.parent)
    df.to_csv(dst, index=False, encoding="utf-8")
    return int(len(df))


def run_bundle_script(bundle_dir: Path, test_src: Path, sample_src: Path, n_rows: int | None) -> dict[str, Any]:
    data = ensure_dir(bundle_dir / "data")
    out = ensure_dir(bundle_dir / "output")
    for path in out.glob("*"):
        path.unlink()
    if n_rows is None:
        shutil.copy2(test_src, data / "test.csv")
        shutil.copy2(sample_src, data / "sample_submission.csv")
    else:
        n_test = _write_head_csv(test_src, data / "test.csv", n_rows)
        sample = pd.read_csv(sample_src, encoding="utf-8-sig", nrows=n_rows)
        if len(sample) != n_test:
            raise AssertionError("5-row smoke: test/sample row counts differ")
        sample.to_csv(data / "sample_submission.csv", index=False, encoding="utf-8")
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
    out_csv = out / "submission.csv"
    sub = pd.read_csv(out_csv) if out_csv.exists() else None
    sample = pd.read_csv(data / "sample_submission.csv", encoding="utf-8-sig")
    validation_ok = False
    validation_error = None
    if sub is not None:
        try:
            validate_submission(sample, sub, ["row_id"], probability=True)
            validation_ok = True
        except Exception as exc:  # noqa: BLE001 — record, then re-raise below
            validation_error = str(exc)
    payload = {
        "returncode": proc.returncode,
        "wall_s": wall,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
        "n_rows": None if sub is None else int(len(sub)),
        "columns": None if sub is None else list(sub.columns),
        "pred_min": None if sub is None else float(sub["control_success"].min()),
        "pred_max": None if sub is None else float(sub["control_success"].max()),
        "pred_mean": None if sub is None else float(sub["control_success"].mean()),
        "nan": None if sub is None else int(sub["control_success"].isna().sum()),
        "validation_ok": validation_ok,
        "validation_error": validation_error,
    }
    if proc.returncode != 0:
        raise RuntimeError(f"bundle script failed: {payload}")
    if not validation_ok:
        raise RuntimeError(f"submission validation failed: {payload}")
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    hold = summary["holdout"]
    sanity = summary["sanity"]
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        "- **RESULTS ONLY — no ADOPT / REJECT / HOLD decision**",
        f"- generated_at: `{summary['generated_at']}`",
        "- purpose: LB probe / condition split (44→L42 on 2019–2023; vs EXP-005 training-block add)",
        "- unique change: EXP-004 2019–2023 frame with feature set 44 → L42 (`pitcher_id`, `batter_id` dropped; `game_type` kept)",
        "",
        "## Conditions",
        "",
        f"- train seasons: `{hold['train_seasons']}`",
        f"- valid season: `{hold['valid_seasons']}`",
        f"- feature count: `{len(hold['used'])}`",
        f"- feature_list_hash: `{hold['feature_list_hash']}`",
        f"- excluded IDs: `{ID_DROP}`",
        f"- game_type: kept",
        f"- seed: `{hold['seed']}`",
        f"- CatBoost params: `{json.dumps(hold['params'], ensure_ascii=False)}`",
        f"- sample weights: `{json.dumps(hold['sample_weight_by_season'], ensure_ascii=False)}`",
        "- preprocessing: `prepare_native_frames` CatBoost branch (train-fold medians, `__NA__` cats)",
        "",
        "## Local 2024 sanity",
        "",
        f"- n_train / n_valid: `{hold['n_train']:,}` / `{hold['n_valid']:,}`",
        f"- Brier: `{hold['brier_score']:.12f}`",
        f"- raw BSS: `{hold['raw_brier_skill_score']:.12f}`",
        f"- elapsed_s: `{hold['elapsed_seconds']:.1f}`",
        f"- canonical L42 2024 raw BSS: `{CANONICAL_L42_2024_RAW_BSS:.12f}`",
        f"- delta vs canonical L42: `{sanity['delta_vs_canonical_l42']:+.12f}`",
        f"- delta vs EXP-004 2024: `{sanity['delta_vs_exp004']:+.12f}`",
        f"- replication verdict: `{sanity['verdict']}`",
        "",
        "## Submission",
        "",
        f"- zip: `{summary.get('submit_zip')}`",
        f"- zip sha256: `{summary.get('zip_sha256')}`",
        f"- zip entries: `{summary.get('zip_entries')}`",
        f"- local data/raw/test.csv rows: `{summary.get('local_official_test_n')}`",
        f"- 5-row smoke: `{json.dumps(summary.get('smoke_5row'), ensure_ascii=False)}`",
        f"- local test infer (this workspace copy): `{json.dumps(summary.get('full_infer'), ensure_ascii=False)}`",
        f"- 50k train-format scale infer: `{json.dumps(summary.get('scale_50k'), ensure_ascii=False)}`",
        "",
        "- no DACON upload was performed.",
        "- local `data/raw/test.csv` in this workspace is a 5-row smoke copy, not the full DACON 2025 test file.",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_summary_with_scale(summary_path: Path, scale: dict[str, Any], local_test_n: int) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["local_official_test_n"] = int(local_test_n)
    summary["scale_50k"] = scale
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def run_scale_check_only(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    bundle_dir = resolve_path(f"outputs/submissions/{EXPERIMENT_ID}_bundle")
    summary_path = resolve_path(f"outputs/predictions/{EXPERIMENT_ID}/summary.json")
    if not (bundle_dir / "script.py").exists() or not summary_path.exists():
        raise FileNotFoundError("scale-check-only requires an existing probe bundle and summary")
    scale_test = resolve_path("outputs/submissions/EXP-004_bundle/data/test.csv")
    scale_sample = resolve_path("outputs/submissions/EXP-004_bundle/data/sample_submission.csv")
    if not scale_test.exists() or not scale_sample.exists():
        raise FileNotFoundError("EXP-004 50k scale-check CSVs are missing; not sampling a new file")
    protected_before = snapshot_protected()
    scale = run_bundle_script(bundle_dir, scale_test, scale_sample, n_rows=None)
    if scale.get("n_rows") != 50000:
        raise AssertionError(f"expected 50000 scale rows, got {scale.get('n_rows')}")
    local_test = pd.read_csv(resolve_path(config["data"]["test_path"]), encoding="utf-8-sig")
    compact = {k: v for k, v in scale.items() if k not in {"stdout_tail", "stderr_tail"}}
    compact["stdout_timing"] = "\n".join(
        line
        for line in (scale.get("stdout_tail") or "").splitlines()
        if "TIMING_SUMMARY" in line or "test=" in line
    )
    compact["source"] = "EXP-004_bundle 50k train-format rows; not DACON 2025 test"
    summary = update_summary_with_scale(summary_path, compact, len(local_test))
    write_report(resolve_path("reports/l42_2019_2023_lb_probe.md"), summary)
    assert_protected_unchanged(protected_before, snapshot_protected())
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L42 2019-2023 LB probe.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--skip-submit", action="store_true")
    parser.add_argument("--skip-log", action="store_true")
    parser.add_argument(
        "--scale-check-only",
        action="store_true",
        help="Reuse saved model/bundle; run 50k train-format scale inference; do not retrain or append log.",
    )
    args = parser.parse_args()
    if args.scale_check_only:
        run_scale_check_only(args.config)
        return

    protected_before = snapshot_protected()
    config = load_config(args.config)
    seed = int(config.get("runtime", {}).get("seed", 42))
    set_seed(seed)
    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    params = dict(config.get("models", {}).get("params", {}).get("catboost", {}))
    scale = float(resolve_metric_settings(config.get("task", {}))["scale"])
    target_col = data_cfg["target_col"]
    season_col = data_cfg["season_col"]
    id_col = data_cfg["id_col"]
    exclude_cols = list(feat_cfg.get("exclude_cols") or [])
    categorical_cols_cfg = list(feat_cfg.get("categorical_cols") or [])

    if 2024 in WEIGHTS:
        raise AssertionError("EXP-004 2019-2023 weights must not include 2024")
    if WEIGHTS != {2019: 1.0, 2020: 1.2, 2021: 1.4, 2022: 1.6, 2023: 2.0}:
        raise AssertionError(f"unexpected 2019-2023 weights: {WEIGHTS}")

    train_df = load_competition_data(config)["train"]
    used, nums, cats = l42_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=exclude_cols,
        categorical_cols=categorical_cols_cfg,
    )
    assert_l42_contract(used, cats)

    print("=== primary 2024 holdout: L42, train 2019-2023 ===", flush=True)
    holdout = run_primary_holdout(
        train_df=train_df,
        used=used,
        numeric_cols=nums,
        categorical_cols=cats,
        target_col=target_col,
        season_col=season_col,
        id_col=id_col,
        params=params,
        seed=seed,
        scale=scale,
        weight_map=WEIGHTS,
    )
    delta_l42 = holdout["raw_brier_skill_score"] - CANONICAL_L42_2024_RAW_BSS
    delta_004 = holdout["raw_brier_skill_score"] - CANONICAL_EXP004_2024_RAW_BSS
    replicated = abs(delta_l42) <= SANITY_ABS_TOL
    verdict = "PASS" if replicated else "FAIL"
    print(
        f"raw_bss={holdout['raw_brier_skill_score']:.6f} "
        f"delta_vs_L42={delta_l42:+.6f} delta_vs_EXP004={delta_004:+.6f} "
        f"verdict={verdict}",
        flush=True,
    )
    if not replicated:
        raise SystemExit(
            "STOP: 2024 raw BSS did not reproduce canonical L42 "
            f"({CANONICAL_L42_2024_RAW_BSS}); |delta|={abs(delta_l42):.6f} > {SANITY_ABS_TOL}. "
            "Submission artifact not canonicalized."
        )

    pred_dir = ensure_dir(resolve_path(f"outputs/predictions/{EXPERIMENT_ID}"))
    model_dir = ensure_dir(resolve_path(f"outputs/models/{EXPERIMENT_ID}"))
    report_dir = ensure_dir(resolve_path("reports"))
    holdout["pred_frame"].to_csv(pred_dir / "holdout_2024_predictions.csv", index=False)
    paths = save_model_bundle(holdout, model_dir)

    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "results_only_no_adopt_reject",
        "holdout": {k: v for k, v in holdout.items() if k not in {"model", "pred_frame"}},
        "sanity": {
            "canonical_l42_raw_bss": CANONICAL_L42_2024_RAW_BSS,
            "canonical_exp004_raw_bss": CANONICAL_EXP004_2024_RAW_BSS,
            "delta_vs_canonical_l42": delta_l42,
            "delta_vs_exp004": delta_004,
            "abs_tol": SANITY_ABS_TOL,
            "verdict": verdict,
        },
        "model_paths": paths,
        "leakage_check": (
            "train_seasons_disjoint_from_2024; features_row_local; "
            "no_test_aggregates_rolling_frequency_or_target_encoding"
        ),
    }

    if not args.skip_submit:
        req_src = resolve_path("outputs/submissions/EXP-004_bundle/requirements.txt")
        zip_path = resolve_path(f"outputs/submissions/{EXPERIMENT_ID}_submit.zip")
        bundle_dir = resolve_path(f"outputs/submissions/{EXPERIMENT_ID}_bundle")
        zip_names = build_submit_zip(
            model_dir=model_dir,
            requirements_src=req_src,
            zip_path=zip_path,
            bundle_dir=bundle_dir,
        )
        test_src = resolve_path(config["data"]["test_path"])
        sample_src = resolve_path(config["data"]["sample_submission_path"])
        smoke = run_bundle_script(bundle_dir, test_src, sample_src, n_rows=5)
        full_infer = run_bundle_script(bundle_dir, test_src, sample_src, n_rows=None)
        zip_fp = file_fingerprint(zip_path)
        summary.update(
            {
                "submit_zip": str(zip_path),
                "zip_entries": zip_names,
                "zip_sha256": zip_fp.get("sha256"),
                "zip_nbytes": zip_fp.get("nbytes"),
                "smoke_5row": smoke,
                "full_infer": {
                    k: v
                    for k, v in full_infer.items()
                    if k not in {"stdout_tail", "stderr_tail"}
                },
                "full_infer_stdout_tail": full_infer.get("stdout_tail"),
            }
        )
        print(
            f"submit zip={zip_path} smoke_rows={smoke['n_rows']} "
            f"full_rows={full_infer['n_rows']} full_wall_s={full_infer['wall_s']:.3f}",
            flush=True,
        )

    report_path = report_dir / "l42_2019_2023_lb_probe.md"
    write_report(report_path, summary)
    (pred_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    if not args.skip_log:
        append_experiment_log(
            config["output"]["experiments_path"],
            {
                "experiment_id": EXPERIMENT_ID,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "purpose": "l42_2019_2023_lb_probe_feature_set_split",
                "data_version": config.get("project", {}).get("data_version", ""),
                "features": to_json_text(
                    {
                        "n": len(used),
                        "drop_ids": list(ID_DROP),
                        "game_type": "kept",
                        "feature_list_hash": holdout["feature_list_hash"],
                    }
                ),
                "model": "catboost",
                "params": to_json_text({**params, "sample_weight_by_season": WEIGHTS}),
                "cv_strategy": to_json_text(
                    {
                        "primary": {
                            "train_seasons": TRAIN_SEASONS,
                            "valid_seasons": VALID_SEASONS,
                        }
                    }
                ),
                "metric": "brier_skill_score",
                "metric_direction": "maximize",
                "cv_score": holdout["raw_brier_skill_score"],
                "cv_std": "",
                "holdout_score": holdout["raw_brier_skill_score"],
                "public_lb_score": "",
                "cv_lb_gap": "",
                "seed": seed,
                "notes": to_json_text(
                    {
                        "decision": "results_only_no_adopt_reject",
                        "canonical_l42_raw_bss": CANONICAL_L42_2024_RAW_BSS,
                        "delta_vs_canonical_l42": delta_l42,
                        "delta_vs_exp004": delta_004,
                        "sanity_verdict": verdict,
                        "submit_zip": summary.get("submit_zip"),
                        "unique_change": "EXP-004 2019-2023 training frame, 44 -> L42",
                    }
                ),
                "next_action": "Public LB probe only; no adopt/reject.",
                "status": "success",
                "error": "",
            },
        )

    assert_protected_unchanged(protected_before, snapshot_protected())
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
