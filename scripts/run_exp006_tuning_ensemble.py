"""EXP-006: Optuna tune CatBoost/LightGBM (L + weights), ensemble, full-train submit."""

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
import optuna
import pandas as pd
from lightgbm import LGBMClassifier

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

optuna.logging.set_verbosity(optuna.logging.WARNING)

ID_DROP = ["pitcher_id", "batter_id"]
HOLD_TRAIN = [2019, 2020, 2021, 2022, 2023]
HOLD_VALID = [2024]
FULL_WEIGHTS = {**{int(k): float(v) for k, v in SEASON_WEIGHTS.items()}, 2024: 2.4}
HOLD_WEIGHTS = {int(k): float(v) for k, v in SEASON_WEIGHTS.items()}
N_TRIALS = 35
SEED = 42
EARLY_STOPPING_ROUNDS = 50


def make_weights(seasons: np.ndarray, weight_map: dict[int, float]) -> np.ndarray:
    return np.asarray([float(weight_map[int(s)]) for s in seasons], dtype=float)


def eval_raw_bss(y: np.ndarray | pd.Series, pred: np.ndarray, scale: float) -> float:
    return float(evaluate_predictions(y, pred, scale=scale)["raw_brier_skill_score"])


def prepare_holdout_frames(
    train_df: pd.DataFrame,
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    backend: str,
) -> dict[str, Any]:
    fold = build_season_holdout(
        train_df, season_col, HOLD_TRAIN, HOLD_VALID, name="exp006_primary_2024"
    )
    cats = [c for c in categorical_cols if c in used]
    nums = [c for c in numeric_cols if c in used]
    X_tr_raw = train_df.iloc[fold.train_idx][used]
    y_tr = train_df.iloc[fold.train_idx][target_col]
    X_va_raw = train_df.iloc[fold.valid_idx][used]
    y_va = train_df.iloc[fold.valid_idx][target_col]
    X_tr, X_va = prepare_native_frames(
        X_tr_raw, X_va_raw, numeric_cols=nums, categorical_cols=cats, backend=backend
    )
    sw = make_weights(train_df.iloc[fold.train_idx][season_col].to_numpy(), HOLD_WEIGHTS)
    return {
        "fold": fold,
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_va": X_va,
        "y_va": y_va,
        "sw": sw,
        "cats": cats,
        "nums": nums,
        "used": used,
    }


def _optuna_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    val = trial.value
    best = study.best_value if study.best_trial is not None else None
    print(
        f"  [{study.study_name}] trial={trial.number} raw_bss={val} best={best}",
        flush=True,
    )


def tune_catboost(data: dict[str, Any], scale: float, n_trials: int, seed: int) -> dict[str, Any]:
    X_tr, y_tr, X_va, y_va, sw, cats = (
        data["X_tr"],
        data["y_tr"],
        data["X_va"],
        data["y_va"],
        data["sw"],
        data["cats"],
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "iterations": trial.suggest_int("iterations", 200, 500),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "loss_function": "Logloss",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "od_type": "Iter",
            "od_wait": EARLY_STOPPING_ROUNDS,
            "use_best_model": True,
        }
        model = build_model("catboost", params, seed)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cats,
            sample_weight=sw,
            eval_set=(X_va, y_va),
        )
        pred, _ = positive_class_proba(model, X_va)
        trial.set_user_attr("best_iteration", int(model.get_best_iteration() or params["iterations"]))
        return eval_raw_bss(y_va, pred, scale)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="exp006_catboost",
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_optuna_progress])
    elapsed = time.perf_counter() - t0

    best_params = dict(study.best_params)
    best_iter = int(study.best_trial.user_attrs.get("best_iteration", best_params["iterations"]))
    best_params["iterations"] = max(best_iter, 50)
    best_params.update(
        {
            "loss_function": "Logloss",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        }
    )
    # Holdout OOF pred: same early-stopping protocol as search (single-season holdout).
    fit_params = {
        **best_params,
        "od_type": "Iter",
        "od_wait": EARLY_STOPPING_ROUNDS,
        "use_best_model": True,
        "iterations": int(study.best_params["iterations"]),
    }
    model = build_model("catboost", fit_params, seed)
    model.fit(X_tr, y_tr, cat_features=cats, sample_weight=sw, eval_set=(X_va, y_va))
    pred, _ = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    best_params["iterations"] = int(model.get_best_iteration() or best_params["iterations"])
    return {
        "backend": "catboost",
        "n_trials": n_trials,
        "best_params": {
            k: best_params[k]
            for k in ("depth", "learning_rate", "l2_leaf_reg", "iterations", "bagging_temperature")
        },
        "best_raw_bss": float(study.best_value),
        "refit_raw_bss": float(metrics["raw_brier_skill_score"]),
        "refit_official_bss": float(metrics["official_brier_skill_score"]),
        "refit_brier": float(metrics["brier_score"]),
        "pred": pred,
        "model": model,
        "elapsed_seconds": float(elapsed),
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "best_iteration": t.user_attrs.get("best_iteration"),
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }


def tune_lightgbm(data: dict[str, Any], scale: float, n_trials: int, seed: int) -> dict[str, Any]:
    from lightgbm import early_stopping, log_evaluation

    X_tr, y_tr, X_va, y_va, sw, cats = (
        data["X_tr"],
        data["y_tr"],
        data["X_va"],
        data["y_va"],
        data["sw"],
        data["cats"],
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sw,
            categorical_feature=cats,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
        )
        pred, _ = positive_class_proba(model, X_va)
        best_it = int(getattr(model, "best_iteration_", None) or params["n_estimators"])
        trial.set_user_attr("best_iteration", best_it)
        return eval_raw_bss(y_va, pred, scale)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="exp006_lightgbm",
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_optuna_progress])
    elapsed = time.perf_counter() - t0

    best_params = dict(study.best_params)
    best_params.update({"random_state": seed, "n_jobs": -1, "verbosity": -1})
    model = LGBMClassifier(**{**best_params, "n_estimators": int(study.best_params["n_estimators"])})
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sw,
        categorical_feature=cats,
        eval_set=[(X_va, y_va)],
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
    )
    pred, _ = positive_class_proba(model, X_va)
    metrics = evaluate_predictions(y_va, pred, scale=scale)
    best_params["n_estimators"] = int(
        getattr(model, "best_iteration_", None) or best_params["n_estimators"]
    )
    return {
        "backend": "lightgbm",
        "n_trials": n_trials,
        "best_params": best_params,
        "best_raw_bss": float(study.best_value),
        "refit_raw_bss": float(metrics["raw_brier_skill_score"]),
        "refit_official_bss": float(metrics["official_brier_skill_score"]),
        "refit_brier": float(metrics["brier_score"]),
        "pred": pred,
        "model": model,
        "elapsed_seconds": float(elapsed),
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "best_iteration": t.user_attrs.get("best_iteration"),
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }


def ensemble_grid(y_va, pred_cb, pred_lgb, scale: float) -> list[dict[str, Any]]:
    rows = []
    corr = float(np.corrcoef(pred_cb, pred_lgb)[0, 1])
    for w_cb in (0.3, 0.5, 0.7):
        pred = w_cb * pred_cb + (1.0 - w_cb) * pred_lgb
        pred = np.clip(pred, 0.0, 1.0)
        m = evaluate_predictions(y_va, pred, scale=scale)
        rows.append(
            {
                "name": f"ensemble_w_cb={w_cb}",
                "w_catboost": w_cb,
                "w_lightgbm": 1.0 - w_cb,
                "raw_brier_skill_score": float(m["raw_brier_skill_score"]),
                "official_brier_skill_score": float(m["official_brier_skill_score"]),
                "brier_score": float(m["brier_score"]),
            }
        )
    # simple mean == 0.5
    return corr, rows


def train_full_backend(
    *,
    backend: str,
    train_df: pd.DataFrame,
    used: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    season_col: str,
    params: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    cats = [c for c in categorical_cols if c in used]
    nums = [c for c in numeric_cols if c in used]
    X_all = train_df[used]
    y_all = train_df[target_col]
    medians = {c: float(X_all[c].median()) if X_all[c].notna().any() else 0.0 for c in nums}
    X_fit = X_all.copy()
    for c in nums:
        X_fit[c] = X_fit[c].fillna(medians[c])
    if backend == "catboost":
        for c in cats:
            X_fit[c] = X_fit[c].astype("string").fillna("__NA__").astype(str)
        sw = make_weights(train_df[season_col].to_numpy(), FULL_WEIGHTS)
        model = build_model("catboost", {**params, "verbose": False, "allow_writing_files": False}, seed)
        model.fit(X_fit, y_all, cat_features=cats, sample_weight=sw)
        fname = "catboost_full.joblib"
    elif backend == "lightgbm":
        # lightgbm native cats via prepare on dummy
        X_tr, _ = prepare_native_frames(
            X_all, X_all.iloc[:1].copy(), numeric_cols=nums, categorical_cols=cats, backend="lightgbm"
        )
        sw = make_weights(train_df[season_col].to_numpy(), FULL_WEIGHTS)
        model = LGBMClassifier(**{**params, "random_state": seed, "n_jobs": -1, "verbosity": -1})
        model.fit(X_tr, y_all, sample_weight=sw, categorical_feature=cats)
        X_fit = X_tr
        fname = "lightgbm_full.joblib"
    else:
        raise ValueError(backend)

    ensure_dir(out_dir)
    path = out_dir / fname
    joblib.dump(model, path)
    meta = {
        "experiment_id": "EXP-006",
        "backend": backend,
        "features_used": used,
        "numeric_cols": nums,
        "categorical_cols": cats,
        "numeric_medians_full_train_2019_2024": medians,
        "numeric_medians_primary_train_2019_2023": medians,
        "sample_weight_by_season": {str(k): float(v) for k, v in sorted(FULL_WEIGHTS.items())},
        "params": params,
        "model_filename": fname,
        "id_col": "row_id",
        "target_col": "control_success",
        "dropped_ids": ID_DROP,
        "has_game_type": "game_type" in used,
    }
    (out_dir / "feature_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"model_path": str(path), "meta_path": str(out_dir / "feature_meta.json"), "meta": meta, "model": model}


SCRIPT_SINGLE = '''# script.py — EXP-006 single-model inference
from __future__ import annotations
import json, os, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ID_COL, TARGET_COL = "row_id", "control_success"
ROOT = Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR, OUT_DIR = ROOT / "data", ROOT / "model", ROOT / "output"
META_PATH = MODEL_DIR / "feature_meta.json"
OUT_PATH = OUT_DIR / "submission.csv"

def load_meta():
    return json.loads(META_PATH.read_text(encoding="utf-8"))

def prepare_features(df, meta):
    used = list(meta["features_used"])
    nums = list(meta["numeric_cols"])
    cats = list(meta["categorical_cols"])
    medians = meta.get("numeric_medians_full_train_2019_2024") or meta["numeric_medians_primary_train_2019_2023"]
    X = df[used].copy()
    for c in nums:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(float(medians.get(c, 0.0)))
    backend = meta.get("backend", "catboost")
    if backend == "lightgbm":
        for c in cats:
            s = X[c].astype("string").fillna("__NA__").astype(str)
            X[c] = pd.Categorical(s)
    else:
        for c in cats:
            X[c] = X[c].astype("string").fillna("__NA__").astype(str)
    return X

def positive_proba(model, X):
    classes = list(getattr(model, "classes_", [0, 1]))
    pos = classes.index(1) if 1 in classes else classes.index(1.0)
    return np.asarray(model.predict_proba(X), dtype=float)[:, pos]

def main():
    meta = load_meta()
    model = joblib.load(MODEL_DIR / meta["model_filename"])
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    X = prepare_features(test, meta)
    preds = np.clip(positive_proba(model, X), 0, 1)
    pred_map = dict(zip(test[ID_COL].tolist(), preds))
    sub[TARGET_COL] = [float(pred_map.get(r, cur)) for r, cur in zip(sub[ID_COL], sub[TARGET_COL])]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub[[ID_COL, TARGET_COL]].to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUT_PATH} rows={len(sub)}")

if __name__ == "__main__":
    main()
'''

SCRIPT_ENSEMBLE = '''# script.py — EXP-006 ensemble inference
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ID_COL, TARGET_COL = "row_id", "control_success"
ROOT = Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR, OUT_DIR = ROOT / "data", ROOT / "model", ROOT / "output"
META_PATH = MODEL_DIR / "ensemble_meta.json"
OUT_PATH = OUT_DIR / "submission.csv"

def prepare_features(df, meta, backend):
    used = list(meta["features_used"])
    nums = list(meta["numeric_cols"])
    cats = list(meta["categorical_cols"])
    medians = meta["numeric_medians_full_train_2019_2024"]
    X = df[used].copy()
    for c in nums:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(float(medians.get(c, 0.0)))
    if backend == "lightgbm":
        for c in cats:
            X[c] = pd.Categorical(X[c].astype("string").fillna("__NA__").astype(str))
    else:
        for c in cats:
            X[c] = X[c].astype("string").fillna("__NA__").astype(str)
    return X

def positive_proba(model, X):
    classes = list(getattr(model, "classes_", [0, 1]))
    pos = classes.index(1) if 1 in classes else classes.index(1.0)
    return np.asarray(model.predict_proba(X), dtype=float)[:, pos]

def main():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    cb = joblib.load(MODEL_DIR / "catboost_full.joblib")
    lgb = joblib.load(MODEL_DIR / "lightgbm_full.joblib")
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    X_cb = prepare_features(test, meta, "catboost")
    X_lgb = prepare_features(test, meta, "lightgbm")
    w = float(meta["w_catboost"])
    preds = np.clip(w * positive_proba(cb, X_cb) + (1 - w) * positive_proba(lgb, X_lgb), 0, 1)
    pred_map = dict(zip(test[ID_COL].tolist(), preds))
    sub[TARGET_COL] = [float(pred_map.get(r, cur)) for r, cur in zip(sub[ID_COL], sub[TARGET_COL])]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub[[ID_COL, TARGET_COL]].to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUT_PATH} rows={len(sub)} w_cb={w}")

if __name__ == "__main__":
    main()
'''


def write_requirements(path: Path, ensemble: bool) -> None:
    # pin from freeze-ish known good set + lightgbm if needed
    lines = [
        "catboost==1.2.10",
        "joblib==1.5.3",
        "numpy==2.4.4",
        "pandas==3.0.2",
        "scikit-learn==1.8.0",
        "scipy==1.17.1",
    ]
    if ensemble:
        # get local lightgbm version
        import lightgbm as lgb

        lines.append(f"lightgbm=={lgb.__version__}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pack_submit(
    *,
    bundle_dir: Path,
    zip_path: Path,
    mode: str,
    artifacts: dict[str, Any],
) -> list[str]:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    ensure_dir(bundle_dir / "model")
    if mode == "ensemble":
        (bundle_dir / "script.py").write_text(SCRIPT_ENSEMBLE, encoding="utf-8")
        shutil.copy2(artifacts["cb_model"], bundle_dir / "model" / "catboost_full.joblib")
        shutil.copy2(artifacts["lgb_model"], bundle_dir / "model" / "lightgbm_full.joblib")
        (bundle_dir / "model" / "ensemble_meta.json").write_text(
            json.dumps(artifacts["ensemble_meta"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_requirements(bundle_dir / "requirements.txt", ensemble=True)
    else:
        (bundle_dir / "script.py").write_text(SCRIPT_SINGLE, encoding="utf-8")
        shutil.copy2(artifacts["model"], bundle_dir / "model" / Path(artifacts["model"]).name)
        shutil.copy2(artifacts["meta"], bundle_dir / "model" / "feature_meta.json")
        write_requirements(bundle_dir / "requirements.txt", ensemble=(mode == "lightgbm"))
        # ensure meta model_filename matches copied name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in bundle_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(bundle_dir)).replace("\\", "/"))
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def smoke(bundle_dir: Path) -> dict[str, Any]:
    import subprocess
    import sys

    data = ensure_dir(bundle_dir / "data")
    out = ensure_dir(bundle_dir / "output")
    for p in out.glob("*"):
        p.unlink()
    shutil.copy2(resolve_path("data/raw/test.csv"), data / "test.csv")
    shutil.copy2(resolve_path("data/raw/sample_submission.csv"), data / "sample_submission.csv")
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
    sub_path = out / "submission.csv"
    ok = proc.returncode == 0 and sub_path.exists()
    sub = pd.read_csv(sub_path) if ok else None
    return {
        "returncode": proc.returncode,
        "wall_s": wall,
        "ok": ok,
        "n_rows": None if sub is None else len(sub),
        "pred_mean": None if sub is None else float(sub["control_success"].mean()),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-10:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-10:]),
    }


def main() -> None:
    config = load_config("configs/baseline.yaml")
    set_seed(SEED)
    scale = float(resolve_metric_settings(config.get("task", {}))["scale"])
    data_cfg = config["data"]
    feat_cfg = config.get("features", {})
    target_col = data_cfg["target_col"]
    season_col = data_cfg["season_col"]

    train_df = load_competition_data(config)["train"]
    used0, nums0, cats0 = build_feature_lists(
        list(train_df.columns),
        target_col=target_col,
        exclude_cols=list(feat_cfg.get("exclude_cols") or []),
        categorical_cols=list(feat_cfg.get("categorical_cols") or []),
    )
    used = [c for c in used0 if c not in ID_DROP]
    cats = [c for c in cats0 if c not in ID_DROP]
    nums = [c for c in nums0 if c in used]
    assert "game_type" in used and "pitcher_id" not in used

    print("Prepare holdout frames (catboost backend)...", flush=True)
    data_cb = prepare_holdout_frames(
        train_df, used, nums, cats, target_col, season_col, backend="catboost"
    )
    print("Prepare holdout frames (lightgbm backend)...", flush=True)
    data_lgb = prepare_holdout_frames(
        train_df, used, nums, cats, target_col, season_col, backend="lightgbm"
    )

    print(f"=== Optuna CatBoost n_trials={N_TRIALS} ===", flush=True)
    cb = tune_catboost(data_cb, scale, N_TRIALS, SEED)
    print(f"CB best_raw_bss={cb['best_raw_bss']:.6f} refit={cb['refit_raw_bss']:.6f}", flush=True)

    print(f"=== Optuna LightGBM n_trials={N_TRIALS} ===", flush=True)
    lgb = tune_lightgbm(data_lgb, scale, N_TRIALS, SEED)
    print(f"LGB best_raw_bss={lgb['best_raw_bss']:.6f} refit={lgb['refit_raw_bss']:.6f}", flush=True)

    corr, ens_rows = ensemble_grid(data_cb["y_va"], cb["pred"], lgb["pred"], scale)
    # note: preds aligned on same valid rows / order from same fold indices
    # Ensure same y - both folds identical indices
    assert len(cb["pred"]) == len(lgb["pred"])

    candidates = [
        {"name": "tuned_catboost", "raw_brier_skill_score": cb["refit_raw_bss"], "kind": "catboost"},
        {"name": "tuned_lightgbm", "raw_brier_skill_score": lgb["refit_raw_bss"], "kind": "lightgbm"},
        *[{**r, "kind": "ensemble"} for r in ens_rows],
    ]
    best = max(candidates, key=lambda r: r["raw_brier_skill_score"])
    print(f"Best holdout combo: {best}", flush=True)

    models_dir = ensure_dir(resolve_path("outputs/models/EXP-006"))
    pred_dir = ensure_dir(resolve_path("outputs/predictions/EXP-006"))

    # Full retrain based on best
    print("=== Full retrain 2019-2024 ===", flush=True)
    submit_mode = best["kind"]
    artifacts: dict[str, Any] = {}
    if submit_mode == "catboost":
        full_cb = train_full_backend(
            backend="catboost",
            train_df=train_df,
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            target_col=target_col,
            season_col=season_col,
            params={**cb["best_params"], "loss_function": "Logloss"},
            seed=SEED,
            out_dir=ensure_dir(models_dir / "catboost"),
        )
        artifacts = {
            "model": full_cb["model_path"],
            "meta": full_cb["meta_path"],
        }
        shutil.copy2(full_cb["model_path"], models_dir / "catboost_full.joblib")
        shutil.copy2(full_cb["meta_path"], models_dir / "feature_meta.json")
    elif submit_mode == "lightgbm":
        full_lgb = train_full_backend(
            backend="lightgbm",
            train_df=train_df,
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            target_col=target_col,
            season_col=season_col,
            params={k: v for k, v in lgb["best_params"].items() if k not in ("random_state", "n_jobs", "verbosity")},
            seed=SEED,
            out_dir=ensure_dir(models_dir / "lightgbm"),
        )
        artifacts = {"model": full_lgb["model_path"], "meta": full_lgb["meta_path"]}
        shutil.copy2(full_lgb["model_path"], models_dir / Path(full_lgb["model_path"]).name)
        shutil.copy2(full_lgb["meta_path"], models_dir / "feature_meta.json")
    else:
        # ensemble: train both
        full_cb = train_full_backend(
            backend="catboost",
            train_df=train_df,
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            target_col=target_col,
            season_col=season_col,
            params={**cb["best_params"], "loss_function": "Logloss"},
            seed=SEED,
            out_dir=ensure_dir(models_dir / "catboost"),
        )
        full_lgb = train_full_backend(
            backend="lightgbm",
            train_df=train_df,
            used=used,
            numeric_cols=nums,
            categorical_cols=cats,
            target_col=target_col,
            season_col=season_col,
            params={k: v for k, v in lgb["best_params"].items() if k not in ("random_state", "n_jobs", "verbosity")},
            seed=SEED,
            out_dir=ensure_dir(models_dir / "lightgbm"),
        )
        ens_meta = {
            "experiment_id": "EXP-006",
            "mode": "ensemble",
            "w_catboost": best["w_catboost"],
            "w_lightgbm": best["w_lightgbm"],
            "features_used": used,
            "numeric_cols": nums,
            "categorical_cols": cats,
            "numeric_medians_full_train_2019_2024": full_cb["meta"]["numeric_medians_full_train_2019_2024"],
            "sample_weight_by_season": {str(k): float(v) for k, v in sorted(FULL_WEIGHTS.items())},
            "catboost_params": cb["best_params"],
            "lightgbm_params": lgb["best_params"],
            "dropped_ids": ID_DROP,
        }
        (models_dir / "ensemble_meta.json").write_text(
            json.dumps(ens_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        artifacts = {
            "cb_model": full_cb["model_path"],
            "lgb_model": full_lgb["model_path"],
            "ensemble_meta": ens_meta,
        }

    zip_path = resolve_path("outputs/submissions/EXP-006_submit.zip")
    bundle_dir = resolve_path("outputs/submissions/EXP-006_bundle")
    zip_names = pack_submit(
        bundle_dir=bundle_dir,
        zip_path=zip_path,
        mode=submit_mode if submit_mode != "ensemble" else "ensemble",
        artifacts=artifacts,
    )
    smoke_res = smoke(bundle_dir)
    print(f"smoke ok={smoke_res['ok']} rc={smoke_res['returncode']}", flush=True)

    # Save OOF preds
    oof = pd.DataFrame(
        {
            "row_id": train_df.iloc[data_cb["fold"].valid_idx]["row_id"].to_numpy(),
            "y_true": data_cb["y_va"].to_numpy(),
            "pred_catboost": cb["pred"],
            "pred_lightgbm": lgb["pred"],
        }
    )
    oof.to_csv(pred_dir / "holdout_2024_predictions.csv", index=False)

    summary = {
        "experiment_id": "EXP-006",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_trials": N_TRIALS,
        "features_n": len(used),
        "dropped_ids": ID_DROP,
        "holdout": {"train_seasons": HOLD_TRAIN, "valid_seasons": HOLD_VALID},
        "sample_weight_holdout": HOLD_WEIGHTS,
        "sample_weight_full": FULL_WEIGHTS,
        "catboost": {k: v for k, v in cb.items() if k not in ("pred", "model", "trials")},
        "lightgbm": {k: v for k, v in lgb.items() if k not in ("pred", "model", "trials")},
        "catboost_trials": cb["trials"],
        "lightgbm_trials": lgb["trials"],
        "ensemble_corr": corr,
        "ensemble_grid": ens_rows,
        "candidates": candidates,
        "selected": best,
        "submit_zip": str(zip_path),
        "zip_entries": zip_names,
        "smoke": smoke_res,
    }
    (pred_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # markdown
    md = [
        "# EXP-006 Tuning + Ensemble",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- holdout: train `{HOLD_TRAIN}` → valid `{HOLD_VALID}` (single season holdout, not TimeSeriesSplit)",
        f"- features: `{len(used)}` with L (drop {ID_DROP}), game_type kept",
        f"- weights holdout: `{HOLD_WEIGHTS}`; full: `{FULL_WEIGHTS}`",
        f"- Optuna trials: `{N_TRIALS}` each; maximize raw BSS",
        "- **no adopt/reject judgment — results only**",
        "",
        "## CatBoost Optuna",
        "",
        f"- best trial raw BSS: `{cb['best_raw_bss']:.6f}`",
        f"- refit holdout raw BSS: `{cb['refit_raw_bss']:.6f}`",
        f"- best params: `{json.dumps(cb['best_params'], ensure_ascii=False)}`",
        f"- tune elapsed_s: `{cb['elapsed_seconds']:.1f}`",
        "",
        "## LightGBM Optuna",
        "",
        f"- best trial raw BSS: `{lgb['best_raw_bss']:.6f}`",
        f"- refit holdout raw BSS: `{lgb['refit_raw_bss']:.6f}`",
        f"- best params: `{json.dumps({k:v for k,v in lgb['best_params'].items() if k not in ('random_state','n_jobs','verbosity')}, ensure_ascii=False)}`",
        f"- tune elapsed_s: `{lgb['elapsed_seconds']:.1f}`",
        "",
        "## Ensemble (2024 holdout preds)",
        "",
        f"- corr(pred_cb, pred_lgb): `{corr:.6f}`",
        "",
        "| combo | w_cb | w_lgb | raw BSS | official BSS | Brier |",
        "|---|---:|---:|---:|---:|---:|",
        f"| tuned_catboost | 1.0 | 0.0 | {cb['refit_raw_bss']:.6f} | {cb['refit_official_bss']:.6f} | {cb['refit_brier']:.6f} |",
        f"| tuned_lightgbm | 0.0 | 1.0 | {lgb['refit_raw_bss']:.6f} | {lgb['refit_official_bss']:.6f} | {lgb['refit_brier']:.6f} |",
    ]
    for r in ens_rows:
        md.append(
            f"| {r['name']} | {r['w_catboost']:.1f} | {r['w_lightgbm']:.1f} | "
            f"{r['raw_brier_skill_score']:.6f} | {r['official_brier_skill_score']:.6f} | "
            f"{r['brier_score']:.6f} |"
        )
    md.extend(
        [
            "",
            f"- selected for full-train/submit: **`{best['name']}`** "
            f"(raw BSS `{best['raw_brier_skill_score']:.6f}`)",
            "",
            "## Full train + submit",
            "",
            f"- train seasons: 2019–2024 full",
            f"- zip: `{zip_path}`",
            f"- zip entries: `{zip_names}`",
            f"- smoke: ok=`{smoke_res['ok']}` rc=`{smoke_res['returncode']}` "
            f"rows=`{smoke_res['n_rows']}` pred_mean=`{smoke_res['pred_mean']}` "
            f"wall_s=`{smoke_res['wall_s']:.3f}`",
            "",
            f"- summary: `{pred_dir / 'summary.json'}`",
            "",
        ]
    )
    md_path = ensure_dir(resolve_path("outputs/eda")) / "tuning_ensemble.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (resolve_path("outputs/eda") / "tuning_ensemble.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    append_experiment_log(
        config["output"]["experiments_path"],
        {
            "experiment_id": "EXP-006",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "optuna_catboost_lightgbm_ensemble_full_train_L_weights",
            "data_version": config.get("project", {}).get("data_version", ""),
            "features": to_json_text(
                {"n": len(used), "drop_ids": ID_DROP, "game_type": "kept", "selected": best}
            ),
            "model": submit_mode,
            "params": to_json_text(
                {
                    "catboost": cb["best_params"],
                    "lightgbm": {
                        k: v
                        for k, v in lgb["best_params"].items()
                        if k not in ("random_state", "n_jobs", "verbosity")
                    },
                    "sample_weight_by_season": FULL_WEIGHTS,
                    "n_trials": N_TRIALS,
                }
            ),
            "cv_strategy": to_json_text(
                {
                    "tune_holdout": {"train": HOLD_TRAIN, "valid": HOLD_VALID},
                    "final": "full_train_2019_2024",
                }
            ),
            "metric": "brier_skill_score",
            "metric_direction": "maximize",
            "cv_score": best["raw_brier_skill_score"],
            "cv_std": "",
            "holdout_score": best["raw_brier_skill_score"],
            "public_lb_score": "",
            "cv_lb_gap": "",
            "seed": SEED,
            "notes": to_json_text(
                {
                    "corr": corr,
                    "ensemble_grid": ens_rows,
                    "cb_refit": cb["refit_raw_bss"],
                    "lgb_refit": lgb["refit_raw_bss"],
                    "submit_zip": str(zip_path),
                    "decision": "results_only_no_adopt_reject",
                }
            ),
            "next_action": "Optional: submit EXP-006_submit.zip",
            "status": "success",
            "error": "",
        },
    )
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
