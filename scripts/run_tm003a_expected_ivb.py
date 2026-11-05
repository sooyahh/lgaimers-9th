"""TM-003A: add tm_expected_ivb_by_count to the EXP-004 CatBoost recipe.

Row-level prior-history Trackman count x group mean IVB x current-row as-of mix.
Does not overwrite EXP-004 / FE-001A / FE-002A / TM-002A artifacts. No Public LB submit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exp004_catboost_weighted import fit_catboost
from scripts.run_regime_shift_experiments import SEASON_WEIGHTS
from scripts.tree_exp_common import build_feature_lists, run_tree_experiment
from src.metrics import compute_brier_metrics
from src.trackman_profiles import (
    CORE_PITCH_GROUPS,
    LEGAL_BALLS,
    LEGAL_STRIKES,
    MIN_CELL_N,
    MIX_RATE_COLS,
    TM_EXPECTED_IVB_BY_COUNT,
    TM_EXPECTED_IVB_GLOBAL,
    TRACKMAN_IVB_REQUIRED_COLS,
    compute_tm003a_features,
    summarize_cell_n,
)
from src.utils import (
    append_experiment_log,
    collect_run_provenance,
    ensure_dir,
    load_config,
    resolve_path,
    save_json,
    to_json_text,
)

EXPERIMENT_ID = "TM-003A"
SMOKE_EXPERIMENT_ID = "TM-003A-SMOKE"
BASELINE_ID = "EXP-004"
NEW_FEATURE = TM_EXPECTED_IVB_BY_COUNT
DIAGNOSTIC_FEATURE = TM_EXPECTED_IVB_GLOBAL
TRACKMAN_PATH = ROOT / "data/raw/trackman_history.csv"
EXPECTED_BASELINE = {
    "secondary_2022": 2259.3983277146012,
    "secondary_2023": -1366.4810668740479,
    "primary_2024": 578.1844155969828,
}
FOLD_ORDER = ["secondary_2022", "secondary_2023", "primary_2024"]
FOLD_LABELS = {"secondary_2022": "2022", "secondary_2023": "2023", "primary_2024": "2024"}
PROTECTED = [
    ROOT / "outputs/predictions/EXP-004/summary.json",
    ROOT / "outputs/predictions/EXP-004/fold_metrics.csv",
    ROOT / "outputs/predictions/EXP-004/oof_holdout_predictions.csv",
    ROOT / "outputs/predictions/EXP-004/report.md",
    ROOT / "outputs/predictions/FE-001A/summary.json",
    ROOT / "outputs/predictions/FE-001A/oof_holdout_predictions.csv",
    ROOT / "outputs/predictions/FE-001A/fold_metrics.csv",
    ROOT / "outputs/predictions/FE-002A/summary.json",
    ROOT / "outputs/predictions/FE-002A/oof_holdout_predictions.csv",
    ROOT / "outputs/predictions/FE-002A/fold_metrics.csv",
    ROOT / "outputs/predictions/TM-002A/summary.json",
    ROOT / "outputs/predictions/TM-002A/oof_holdout_predictions.csv",
    ROOT / "outputs/predictions/TM-002A/fold_metrics.csv",
]

RUN_KIND_CANONICAL = "canonical"
RUN_KIND_NON_CANONICAL = "non_canonical"
RUN_KIND_PRE_CANONICAL = "pre_canonical"
REPORT_RELPATH = "reports/tm_003a_expected_ivb.md"
CONFIG_RELPATH = "configs/tm003a.yaml"
RERUN_COMMAND = "python -m scripts.run_tm003a_expected_ivb"
CANONICAL_RERUN_COMMAND = "python -m scripts.run_tm003a_expected_ivb --canonical"


def resolve_run_kind(*, canonical_requested: bool) -> tuple[str, bool]:
    """Canonical is never inferred from git_dirty. Only an explicit flag can set it."""
    if canonical_requested:
        return RUN_KIND_CANONICAL, True
    return RUN_KIND_NON_CANONICAL, False


def parse_notes_json(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"_unparsed": text}
    return value if isinstance(value, dict) else {"_unparsed": text}


def aggregate_overall_fallback(fallback_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool fold-level fallback diagnostics over validation rows. No new feature math."""
    n_valid = 0
    n_constructible = 0
    any_num = 0.0
    weight_num = 0.0
    nan_num = 0.0
    for row in fallback_rows:
        nv = int(row["n_valid"])
        nc = int(row["n_constructible"])
        n_valid += nv
        n_constructible += nc
        any_num += float(row["fallback_any_row_rate"]) * nc
        weight_num += float(row["fallback_weight_share"]) * nc
        nan_num += float(row["tm_feature_nan_rate"]) * nv
    return {
        "n_valid": n_valid,
        "n_constructible": n_constructible,
        "fallback_any_row_rate": (any_num / n_constructible) if n_constructible else float("nan"),
        "fallback_weight_share": (weight_num / n_constructible) if n_constructible else float("nan"),
        "tm_feature_nan_rate": (nan_num / n_valid) if n_valid else float("nan"),
        "denominator_note": {
            "fallback_rates": "constructible validation rows",
            "tm_feature_nan_rate": "all validation rows",
        },
    }


def collect_cell_n_diagnostics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Reuse fold_extra.cell_n_valid_profile. Cell n is the full cell row count."""
    by_fold: dict[str, Any] = {}
    for row in fold_metrics:
        extra = row.get("fold_extra") or {}
        profile = extra.get("cell_n_valid_profile") or summarize_cell_n({"cell_n": {}})
        by_fold[str(row["fold_id"])] = {
            "valid_season": extra.get("valid_season"),
            "min": profile.get("min"),
            "p10": profile.get("p10"),
            "median": profile.get("median"),
            "n_cells": profile.get("n_cells"),
        }
    return {
        "definition": "full cell row count, including IVB-missing rows; not the IVB-observed count",
        "source": "fold_extra.cell_n_valid_profile",
        "by_fold": by_fold,
    }


def count_nonfinite_large_cells(profile_cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Count n>=min_n cells whose mean is non-finite. Does not change fallback behavior."""
    n_cells = 0
    details: list[dict[str, Any]] = []
    for season, profile in sorted(profile_cache.items()):
        cell_n = profile.get("cell_n") or {}
        cell_mean = profile.get("cell_mean") or {}
        for key, n_val in cell_n.items():
            mean = cell_mean.get(key, float("nan"))
            if int(n_val) >= MIN_CELL_N and not np.isfinite(float(mean)):
                n_cells += 1
                count_value, group = key if isinstance(key, tuple) else (str(key), "")
                details.append(
                    {
                        "row_season": int(season),
                        "count": count_value,
                        "group": group,
                        "n": int(n_val),
                    }
                )
    return {
        "rule": "n >= min_n and cell mean non-finite -> global fallback",
        "min_n": MIN_CELL_N,
        "n_cells": n_cells,
        "details": details,
    }


def assert_frozen_tm003a_metadata(tm_cfg: dict[str, Any]) -> None:
    """Metadata must match frozen constants. Config is not a tuning interface."""
    meta = tm_cfg.get("tm003a")
    if not isinstance(meta, dict):
        raise AssertionError("configs/tm003a.yaml missing required tm003a metadata block")
    if int(meta.get("min_n")) != MIN_CELL_N:
        raise AssertionError(
            f"tm003a.min_n metadata {meta.get('min_n')} != frozen MIN_CELL_N={MIN_CELL_N}"
        )
    if meta.get("min_n_is_tunable") is not False:
        raise AssertionError("tm003a.min_n_is_tunable must be false; min_n is not tunable")
    if str(meta.get("model_feature")) != TM_EXPECTED_IVB_BY_COUNT:
        raise AssertionError(
            f"tm003a.model_feature {meta.get('model_feature')} != {TM_EXPECTED_IVB_BY_COUNT}"
        )
    if str(meta.get("baseline_experiment")) != BASELINE_ID:
        raise AssertionError(f"tm003a.baseline_experiment must be {BASELINE_ID}")
    if str(meta.get("aggregation")) != "plain_sum":
        raise AssertionError("tm003a.aggregation must remain plain_sum")
    if str(meta.get("temporal_history_rule")) != "trackman.season < row.season":
        raise AssertionError("tm003a.temporal_history_rule must remain trackman.season < row.season")
    if tuple(meta.get("pitch_groups") or ()) != CORE_PITCH_GROUPS:
        raise AssertionError(f"tm003a.pitch_groups must equal {CORE_PITCH_GROUPS}")
    if list(meta.get("excluded_pitch_groups") or []) != ["other"]:
        raise AssertionError("tm003a.excluded_pitch_groups must be [other]")
    count_cell = meta.get("count_cell") or {}
    if tuple(count_cell.get("legal_balls") or ()) != LEGAL_BALLS:
        raise AssertionError(f"tm003a count legal_balls must equal {LEGAL_BALLS}")
    if tuple(count_cell.get("legal_strikes") or ()) != LEGAL_STRIKES:
        raise AssertionError(f"tm003a count legal_strikes must equal {LEGAL_STRIKES}")
    if int(meta.get("seed")) != 42 or int(tm_cfg.get("runtime", {}).get("seed")) != 42:
        raise AssertionError("TM-003A seed metadata/runtime must remain 42")
    meta_tm = ROOT / str(meta.get("trackman_path"))
    if meta_tm.resolve() != TRACKMAN_PATH.resolve():
        raise AssertionError(
            f"tm003a.trackman_path {meta.get('trackman_path')} != {TRACKMAN_PATH}"
        )
    meta_weights = {
        int(k): float(v) for k, v in dict(meta.get("season_weights") or {}).items()
    }
    if meta_weights != {int(k): float(v) for k, v in SEASON_WEIGHTS.items()}:
        raise AssertionError("tm003a.season_weights metadata != locked SEASON_WEIGHTS")
    if meta.get("canonical_requires_explicit_flag") is not True:
        raise AssertionError("canonical must require an explicit runner flag")


def build_run_provenance(
    *,
    experiment_id: str,
    canonical_requested: bool,
    smoke: bool,
    config_path: str | Path,
    artifact_dir: Path,
    report_path: Path | None,
) -> dict[str, Any]:
    git = collect_run_provenance()
    run_kind, canonical = resolve_run_kind(canonical_requested=canonical_requested)
    command = CANONICAL_RERUN_COMMAND if canonical_requested else RERUN_COMMAND
    if smoke:
        command = f"{RERUN_COMMAND} --smoke"
        run_kind = RUN_KIND_NON_CANONICAL
        canonical = False
    cfg_rel = str(config_path).replace("\\", "/")
    if not Path(cfg_rel).is_absolute():
        pass
    else:
        try:
            cfg_rel = str(Path(config_path).resolve().relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            cfg_rel = str(config_path).replace("\\", "/")
    art_rel = str(artifact_dir).replace("\\", "/")
    try:
        art_rel = str(artifact_dir.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        pass
    report_rel = None
    if report_path is not None:
        report_rel = str(report_path).replace("\\", "/")
        try:
            report_rel = str(report_path.resolve().relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            pass
    return {
        "experiment_id": experiment_id,
        "git_commit": git.get("git_commit") or "",
        "git_dirty": git.get("git_dirty") or "",
        "git_branch": git.get("git_branch") or "",
        "author": git.get("author") or "",
        "command": command,
        "config_path": cfg_rel,
        "seed": 42,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline_experiment": BASELINE_ID,
        "feature": NEW_FEATURE,
        "validation_folds": list(FOLD_ORDER),
        "artifact_path": art_rel,
        "report_path": report_rel,
        "run_kind": run_kind,
        "canonical": canonical,
        "canonical_note": (
            "canonical=true only when --canonical is passed explicitly; "
            "git_dirty=false is not sufficient"
        ),
    }


def assert_canonical_run_allowed(provenance: dict[str, Any]) -> None:
    if provenance.get("git_dirty") == "true":
        raise AssertionError("refusing --canonical on a dirty git tree")
    if not str(provenance.get("git_commit") or "").strip():
        raise AssertionError("refusing --canonical without a git commit hash")


def mark_prior_tm003a_rows_superseded(
    path: str | Path,
    *,
    experiment_id: str = EXPERIMENT_ID,
) -> int:
    """Mark earlier TM-003A rows superseded. Call only after a canonical row is saved."""
    path = resolve_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    tm_indices = [i for i, row in enumerate(rows) if row.get("experiment_id") == experiment_id]
    if len(tm_indices) < 2:
        return 0
    keep_i = tm_indices[-1]
    keep_notes = parse_notes_json(rows[keep_i].get("notes", ""))
    if keep_notes.get("canonical") is not True:
        raise AssertionError(
            "refusing to supersede prior TM-003A rows: last row is not canonical"
        )
    n_updated = 0
    for i in tm_indices[:-1]:
        notes = parse_notes_json(rows[i].get("notes", ""))
        if notes.get("superseded") is True:
            continue
        notes["superseded"] = True
        notes["superseded_reason"] = "later_canonical_tm003a_run"
        rows[i]["notes"] = to_json_text(notes)
        n_updated += 1
    if n_updated:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return n_updated


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    st = path.stat()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": True,
        "nbytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime_ns": st.st_mtime_ns,
    }


def assert_configs_match(baseline_cfg: dict[str, Any], tm_cfg: dict[str, Any]) -> None:
    for key in ("cv", "task", "runtime"):
        if baseline_cfg[key] != tm_cfg[key]:
            raise AssertionError(f"config mismatch in {key}")
    if baseline_cfg["models"]["params"]["catboost"] != tm_cfg["models"]["params"]["catboost"]:
        raise AssertionError("CatBoost params mismatch vs baseline.yaml")
    if baseline_cfg["features"]["exclude_cols"] != tm_cfg["features"]["exclude_cols"]:
        raise AssertionError("exclude_cols mismatch; TM-003A must not drop EXP-004 features")
    if baseline_cfg["features"]["categorical_cols"] != tm_cfg["features"]["categorical_cols"]:
        raise AssertionError("categorical_cols mismatch")


def load_baseline_folds(path: Path) -> dict[str, dict[str, Any]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in summary["fold_metrics"]:
        out[row["fold_id"]] = row
        logged = EXPECTED_BASELINE.get(row["fold_id"])
        if logged is None:
            continue
        got = float(row["raw_brier_skill_score"])
        if abs(got - logged) > 1e-8:
            raise AssertionError(
                f"EXP-004 {row['fold_id']} raw BSS {got} != logged {logged}."
            )
    return out


def quantile_10_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    if n == 0:
        raise ValueError("empty arrays for ECE")
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        ps = p_sorted[lo:hi]
        ys = y_sorted[lo:hi]
        ece += (len(ps) / n) * abs(float(ps.mean()) - float(ys.mean()))
    return float(ece)


def pearson_finite(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    pair = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1)
    pair.columns = ["a", "b"]
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(pair))
    if n < 3:
        return float("nan"), n
    corr = float(pair["a"].corr(pair["b"], method="pearson"))
    return corr, n


def evaluate_predeclared_rule(
    deltas: dict[str, float],
    *,
    leakage_ok: bool,
    max_abs_corrs: dict[str, float],
) -> dict[str, Any]:
    """Mechanical §7.11.C / §7.12.A signals. Not a team ADOPT/HOLD/REJECT decision."""
    dup_fail = any(
        np.isfinite(v) and abs(float(v)) >= 0.98 for v in max_abs_corrs.values()
    )
    dmean = float(deltas["dmean"])
    d22, d23, d24 = float(deltas["d2022"]), float(deltas["d2023"]), float(deltas["d2024"])
    n_pos = int(sum(d > 0 for d in (d22, d23, d24)))
    adopt = (
        leakage_ok
        and dmean > 0
        and n_pos >= 2
        and d23 >= 0
        and not dup_fail
    )
    reject = (not leakage_ok) or dup_fail or (dmean < 0 and d23 < 0)
    if adopt:
        bucket = "ADOPT_CANDIDATE"
    elif reject:
        bucket = "REJECT"
    else:
        bucket = "HOLD"
    return {
        "leakage_ok": leakage_ok,
        "mean_delta_positive": dmean > 0,
        "at_least_two_folds_positive": n_pos >= 2,
        "fold_2023_nonnegative": d23 >= 0,
        "duplicate_gate_pass": not dup_fail,
        "n_positive_folds": n_pos,
        "max_abs_corrs": max_abs_corrs,
        "predeclared_rule_bucket": bucket,
        "note": (
            "Mechanical application of the frozen pre-run rule. "
            "This is not a team ADOPT/HOLD/REJECT decision."
        ),
    }


def make_fold_extra_features(
    tm: pd.DataFrame,
    profile_cache: dict[int, dict[str, Any]],
    diag_store: list[dict[str, Any]],
    corr_store: list[dict[str, Any]],
    fallback_store: list[dict[str, Any]],
):
    def fold_extra_features(
        train_df: pd.DataFrame, spec: dict[str, Any]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if len(spec["valid_seasons"]) != 1:
            raise AssertionError("TM-003A expects a single valid season per fold")
        valid_season = int(spec["valid_seasons"][0])
        extra = compute_tm003a_features(train_df, tm, profile_cache=profile_cache)
        if extra is train_df:
            raise AssertionError("TM-003A extra features reused train_df")
        if NEW_FEATURE not in extra.columns:
            raise AssertionError("TM-003A model feature missing")
        if DIAGNOSTIC_FEATURE not in extra.columns:
            raise AssertionError("TM-003A diagnostic feature missing")

        train_mask = train_df["season"].isin(spec["train_seasons"])
        valid_mask = train_df["season"].isin(spec["valid_seasons"])
        feat = extra[NEW_FEATURE]
        for split, mask in (("train", train_mask), ("valid", valid_mask)):
            s = pd.to_numeric(feat.loc[mask], errors="coerce")
            n2019 = int((train_df.loc[mask, "season"] == 2019).sum()) if split == "train" else 0
            diag_store.append(
                {
                    "fold_id": spec["fold_id"],
                    "valid_season": valid_season,
                    "split": split,
                    "n": int(mask.sum()),
                    "missing_rate": float(s.isna().mean()),
                    "n_nan": int(s.isna().sum()),
                    "n_inf": int(np.isinf(s.fillna(0)).sum()),
                    "dtype": str(s.dtype),
                    "n_2019": n2019,
                    "share_2019": float(n2019 / mask.sum()) if int(mask.sum()) else float("nan"),
                    "mean": float(s.dropna().mean()) if s.notna().any() else float("nan"),
                }
            )

        valid_extra = extra.loc[valid_mask]
        valid_main = train_df.loc[valid_mask]
        cons = valid_extra["tm003a_constructible"].astype(bool)
        n_valid = int(valid_mask.sum())
        n_cons = int(cons.sum())
        fallback_any = float(valid_extra.loc[cons, "tm003a_fallback_any"].mean()) if n_cons else float("nan")
        fallback_w = (
            float(valid_extra.loc[cons, "tm003a_fallback_weight_share"].mean()) if n_cons else float("nan")
        )
        contextual_w = (
            float((1.0 - valid_extra.loc[cons, "tm003a_fallback_weight_share"]).mean())
            if n_cons
            else float("nan")
        )
        fallback_store.append(
            {
                "fold_id": spec["fold_id"],
                "valid_season": valid_season,
                "n_valid": n_valid,
                "n_constructible": n_cons,
                "fallback_any_row_rate": fallback_any,
                "fallback_weight_share": fallback_w,
                "contextual_weight_share": contextual_w,
                "tm_feature_nan_rate": float(valid_extra[NEW_FEATURE].isna().mean()),
            }
        )
        corr_targets = {
            DIAGNOSTIC_FEATURE: valid_extra[DIAGNOSTIC_FEATURE],
            MIX_RATE_COLS[0]: valid_main[MIX_RATE_COLS[0]],
            MIX_RATE_COLS[1]: valid_main[MIX_RATE_COLS[1]],
            MIX_RATE_COLS[2]: valid_main[MIX_RATE_COLS[2]],
        }
        for name, series in corr_targets.items():
            corr, n = pearson_finite(valid_extra[NEW_FEATURE], series)
            corr_store.append(
                {
                    "fold_id": spec["fold_id"],
                    "valid_season": valid_season,
                    "target": name,
                    "pearson": corr,
                    "n": n,
                }
            )

        fold_seasons = [int(s) for s in list(spec["train_seasons"]) + list(spec["valid_seasons"])]
        used_by_season = {
            int(season): list(profile_cache[int(season)]["tm_seasons"])
            for season in sorted(profile_cache)
            if int(season) in set(fold_seasons)
        }
        for row_season, tm_seasons in used_by_season.items():
            if any(int(s) >= int(row_season) for s in tm_seasons):
                raise AssertionError(
                    f"{spec['fold_id']} row-level TM leak: season={row_season} used {tm_seasons}"
                )
            if any(int(s) >= valid_season for s in tm_seasons) or 2025 in {int(s) for s in tm_seasons}:
                raise AssertionError(
                    f"{spec['fold_id']} fold-level TM leak: row_season={row_season} used {tm_seasons}"
                )
        valid_used = list(used_by_season.get(valid_season) or ())
        cell_summary = summarize_cell_n(profile_cache.get(valid_season, {"cell_n": {}}))
        meta = {
            "valid_season": valid_season,
            "tm_cutoff_rule": "trackman.season < row.season",
            "tm_seasons_valid_rows": valid_used,
            "tm_seasons_by_row_season": used_by_season,
            "min_n": MIN_CELL_N,
            "formula": "plain_sum",
            "model_feature": NEW_FEATURE,
            "diagnostic_feature": DIAGNOSTIC_FEATURE,
            "diagnostic_is_model_input": False,
            "feature_missing_rate_train": float(feat.loc[train_mask].isna().mean()),
            "feature_missing_rate_valid": float(feat.loc[valid_mask].isna().mean()),
            "fallback_any_row_rate": fallback_any,
            "fallback_weight_share": fallback_w,
            "contextual_weight_share": contextual_w,
            "cell_n_valid_profile": cell_summary,
            "n_train": int(train_mask.sum()),
            "n_valid": int(valid_mask.sum()),
        }
        return extra, meta

    return fold_extra_features


def write_smoke_config(base_cfg: dict[str, Any], train_path: Path, iterations: int) -> Path:
    cfg = {k: v for k, v in base_cfg.items() if k != "_config_path"}
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["train_path"] = str(train_path.relative_to(ROOT)).replace("\\", "/")
    cfg["models"] = json.loads(json.dumps(cfg["models"]))
    cfg["models"]["params"]["catboost"]["iterations"] = int(iterations)
    cfg["project"] = dict(cfg["project"])
    cfg["project"]["purpose"] = "tm003a_smoke"
    out = ROOT / "data/processed/tm003a_smoke/tm003a_smoke.yaml"
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out


def subsample_train(path: Path, rows_per_season: int, seed: int) -> pd.DataFrame:
    train = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    parts = []
    for _, part in train.groupby("season", dropna=False):
        n = min(int(len(part)), int(rows_per_season))
        parts.append(part.sample(n=n, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if not out["row_id"].is_unique:
        raise AssertionError("Smoke subsample lost row_id uniqueness")
    return out


def write_report(**kwargs: Any) -> None:
    baseline_folds = kwargs["baseline_folds"]
    tm_folds = kwargs["tm_folds"]
    diag_df = kwargs["diag_df"]
    corr_df = kwargs["corr_df"]
    fallback_df = kwargs["fallback_df"]
    n_base_feat = kwargs["n_base_feat"]
    n_tm_feat = kwargs["n_tm_feat"]
    feature_assert = kwargs["feature_assert"]
    rule = kwargs["rule"]
    leakage_ok = kwargs["leakage_ok"]
    report_path = kwargs["report_path"]
    seed = kwargs["seed"]
    provenance = kwargs["provenance"]
    overall_fallback = kwargs["overall_fallback"]
    cell_n = kwargs["cell_n"]
    nonfinite = kwargs["nonfinite"]

    fmap = {r["fold_id"]: r for r in tm_folds}
    b_raw = [float(baseline_folds[k]["raw_brier_skill_score"]) for k in FOLD_ORDER]
    t_raw = [float(fmap[k]["raw_brier_skill_score"]) for k in FOLD_ORDER]
    b_mean, t_mean = float(np.mean(b_raw)), float(np.mean(t_raw))
    b_std, t_std = float(np.std(b_raw, ddof=0)), float(np.std(t_raw, ddof=0))
    deltas = [t - b for t, b in zip(t_raw, b_raw)]
    directions = ["+" if d > 0 else ("-" if d < 0 else "0") for d in deltas]
    run_kind = str(provenance.get("run_kind") or RUN_KIND_NON_CANONICAL)
    canonical = bool(provenance.get("canonical"))
    git_commit = provenance.get("git_commit") or ""
    git_dirty = provenance.get("git_dirty") or ""

    if canonical:
        status_lines = [
            "**Run status: CANONICAL.** This report is from the explicit `--canonical` rerun.",
            "",
        ]
    elif run_kind == RUN_KIND_PRE_CANONICAL:
        status_lines = [
            "**Run status: PRE-CANONICAL.** Numbers below are from the dirty working-tree run.",
            "They are **not** the canonical committed rerun.",
            "",
        ]
    else:
        status_lines = [
            "**Run status: NON-CANONICAL.** `git_dirty=false` does not make a run canonical.",
            "Canonical status is granted only by an explicit `--canonical` rerun.",
            "",
        ]

    lines = [
        "# TM-003A Expected IVB by Count",
        "",
        *status_lines,
        f"- generated_at: `{provenance.get('timestamp') or datetime.now().isoformat(timespec='seconds')}`",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        f"- baseline: `{BASELINE_ID}` (existing artifacts reused; not overwritten)",
        f"- feature: `{NEW_FEATURE}`",
        f"- seed: `{seed}`",
        f"- run_kind: `{run_kind}`",
        f"- canonical: `{canonical}`",
        f"- git_commit: `{git_commit}`",
        f"- git_dirty: `{git_dirty}`",
        "",
        "## Experiment definition",
        "",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        f"- baseline: `{BASELINE_ID}`",
        f"- feature: `{NEW_FEATURE}`",
        f"- seed: `{seed}`",
        "- model/controls: CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss;",
        "  EXP-004 season weights 2019:1.0 … 2023:2.0; fold-local median impute",
        "- temporal cutoff: `trackman.season < row.season`",
        f"- code commit: `{git_commit or 'unknown'}`",
        f"- config path: `{provenance.get('config_path') or CONFIG_RELPATH}`",
        f"- min_n: `{MIN_CELL_N}` (frozen, not tunable)",
        "- aggregation: plain sum `Σ_g asof_mix_g × IVB_hist(count, g)`",
        "- pitch groups: fastball / breaking / offspeed (`other` excluded)",
        "- count cells: `balls_before-strikes_before`, legal 0-0 … 3-2",
        "- sparse fallback: cell `n < 1000` → same-group count-agnostic mean",
        "- non-finite defensive fallback: `n >= 1000` but non-finite cell mean → global fallback",
        "",
        "## 1. Hypothesis",
        "",
        "A prior-history Trackman `count × pitch_type_group` IVB profile combined with the",
        "current row's official as-of pitch mix adds movement information that EXP-004 does",
        "not have. Exactly one model feature is added.",
        "",
        "## 2. What changed",
        "",
        f"- Added exactly `{NEW_FEATURE}`",
        f"- Feature count: {n_base_feat} → {n_tm_feat}",
        f"- TM-003A − EXP-004 = `{feature_assert['tm_minus_baseline']}`",
        f"- EXP-004 − TM-003A = `{feature_assert['baseline_minus_tm']}`",
        f"- `{DIAGNOSTIC_FEATURE}` is diagnostic-only and is **not** a model input",
        "- Trackman cutoff: row-level `trackman.season < row.season`",
        f"- Sparse fallback: `min_n = {MIN_CELL_N}` to same-group count-agnostic mean",
        "- Combination: plain sum `Σ_g asof_mix_g × IVB_hist(count, g)`",
        "",
        "## 3. What stayed fixed",
        "",
        "- CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss",
        "- seed 42",
        "- season weights 2019:1.0 … 2023:2.0",
        "- expanding 2022/2023/2024 holdouts",
        "- `src.metrics` unchanged",
        "- fold-local median impute + native cats",
        "- no current-pitch Trackman, no player-ID join, no 2025 TM, no test aggregates",
        "",
        "## CV comparison",
        "",
        "| Fold | n_train | n_valid | EXP-004 raw BSS | TM-003A raw BSS | Δ raw BSS | TM-003A official | TM-003A Brier | runtime_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in zip(FOLD_ORDER, ["2022", "2023", "2024"]):
        b = baseline_folds[key]
        t = fmap[key]
        lines.append(
            f"| {label} | {int(t['n_train']):,} | {int(t['n_valid']):,} | "
            f"{float(b['raw_brier_skill_score']):.6f} | {float(t['raw_brier_skill_score']):.6f} | "
            f"{float(t['raw_brier_skill_score']) - float(b['raw_brier_skill_score']):.6f} | "
            f"{float(t['official_brier_skill_score']):.6f} | {float(t['brier_score']):.6f} | "
            f"{float(t['elapsed_seconds']):.1f} |"
        )
    lines.extend(
        [
            "",
            f"- TM-003A CV mean (ddof=0): `{t_mean:.6f}` (EXP-004 `{b_mean:.6f}`)",
            f"- TM-003A CV std (ddof=0): `{t_std:.6f}` (EXP-004 `{b_std:.6f}`)",
            f"- TM-003A worst fold: `{min(t_raw):.6f}`",
            f"- TM-003A best fold: `{max(t_raw):.6f}`",
            f"- mean Δ raw BSS: `{t_mean - b_mean:.6f}`",
            f"- fold direction: `{', '.join(directions)}` (2022/2023/2024)",
            "",
            "## Fallback diagnostics",
            "",
            "Fold-level rates keep the constructible-row denominator from §7.12.",
            "Overall rates are pooled from those fold counts; they are not hard-coded.",
            "",
            "| Fold | n_valid | n_constructible | fallback_any_row_rate | fallback_weight_share | contextual_weight_share | nan_rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fallback_df.itertuples():
        n_valid = int(getattr(row, "n_valid", 0) or 0)
        n_cons = int(getattr(row, "n_constructible", 0) or 0)
        lines.append(
            f"| {FOLD_LABELS.get(row.fold_id, row.fold_id)} | {n_valid:,} | {n_cons:,} | "
            f"{row.fallback_any_row_rate:.6f} | {row.fallback_weight_share:.6f} | "
            f"{row.contextual_weight_share:.6f} | {row.tm_feature_nan_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Overall | n_valid | n_constructible | fallback_any_row_rate | fallback_weight_share | tm_feature_nan_rate |",
            "|---|---:|---:|---:|---:|---:|",
            f"| all folds | {int(overall_fallback['n_valid']):,} | "
            f"{int(overall_fallback['n_constructible']):,} | "
            f"{float(overall_fallback['fallback_any_row_rate']):.8f} | "
            f"{float(overall_fallback['fallback_weight_share']):.8f} | "
            f"{float(overall_fallback['tm_feature_nan_rate']):.8f} |",
            "",
        ]
    )
    nf_count = nonfinite.get("n_cells")
    if nf_count is None:
        lines.append(
            f"- non-finite `n >= {MIN_CELL_N}` cell-mean fallback cells: `not stored in this artifact`"
        )
    else:
        lines.append(
            f"- non-finite `n >= {MIN_CELL_N}` cell-mean fallback cells: `{int(nf_count)}`"
        )
    lines.extend(
        [
            "- that branch is defensive only; it does not change the frozen sparse rule",
            "",
            "Train-side NaN rate includes structural 2019 rows (no prior-history Trackman).",
            "",
        ]
    )
    train_nan = diag_df.loc[diag_df["split"] == "train"] if len(diag_df) else diag_df
    if len(train_nan):
        lines.append("| Fold | train nan_rate | train n_2019 | train share_2019 |")
        lines.append("|---|---:|---:|---:|")
        for row in train_nan.itertuples():
            lines.append(
                f"| {FOLD_LABELS.get(row.fold_id, row.fold_id)} | {row.missing_rate:.6f} | "
                f"{int(row.n_2019):,} | {row.share_2019:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Cell-n diagnostics",
            "",
            "Cell `n` is the **full cell row count**, including IVB-missing rows.",
            "Values are reused from `fold_extra.cell_n_valid_profile` (no second aggregator).",
            "",
            "| Fold | min | p10 | median | n_cells |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold_id, label in zip(FOLD_ORDER, ["2022", "2023", "2024"]):
        rec = (cell_n.get("by_fold") or {}).get(fold_id) or {}
        lines.append(
            f"| {label} | {rec.get('min')} | {rec.get('p10')} | {rec.get('median')} | "
            f"{rec.get('n_cells')} |"
        )
    lines.extend(
        [
            "",
            "## Duplicate-signal diagnostics (validation, Pearson, per fold)",
            "",
            "| Fold | target | pearson | n |",
            "|---|---|---:|---:|",
        ]
    )
    for row in corr_df.itertuples():
        lines.append(
            f"| {FOLD_LABELS.get(row.fold_id, row.fold_id)} | {row.target} | {row.pearson:.6f} | {int(row.n):,} |"
        )
    lines.extend(
        [
            "",
            "## Predeclared rule signals",
            "",
            "These are mechanical signals from the frozen pre-run rule. They are **not** a",
            "team ADOPT / HOLD / REJECT decision.",
            "",
            f"- leakage_ok: `{leakage_ok}`",
            f"- predeclared_rule_bucket: `{rule['predeclared_rule_bucket']}`",
            f"- duplicate max |corr|: `{to_json_text(rule['max_abs_corrs'])}`",
            "",
            "## Integrity/reproducibility",
            "",
            f"- git_commit: `{git_commit}`",
            f"- git_dirty: `{git_dirty}`",
            f"- git_branch: `{provenance.get('git_branch') or ''}`",
            f"- config path: `{provenance.get('config_path') or CONFIG_RELPATH}`",
            f"- command: `{provenance.get('command') or RERUN_COMMAND}`",
            f"- artifact path: `{provenance.get('artifact_path') or f'outputs/predictions/{EXPERIMENT_ID}'}`",
            f"- report path: `{REPORT_RELPATH}`",
            f"- run_kind: `{run_kind}`",
            f"- canonical: `{canonical}`",
            "",
            "## Re-run",
            "",
            "Default rerun is **not** canonical:",
            "",
            "```powershell",
            RERUN_COMMAND,
            "```",
            "",
            "Canonical rerun (only after an explicit instruction, on a committed tree):",
            "",
            "```powershell",
            CANONICAL_RERUN_COMMAND,
            "```",
            "",
            f"- summary: `outputs/predictions/{EXPERIMENT_ID}/summary.json`",
            f"- OOF: `outputs/predictions/{EXPERIMENT_ID}/oof_holdout_predictions.csv`",
            f"- comparison: `outputs/predictions/{EXPERIMENT_ID}/comparison_vs_exp004.json`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TM-003A vs EXP-004.")
    parser.add_argument("--smoke", action="store_true", help="Small end-to-end pipeline check.")
    parser.add_argument("--smoke-rows-per-season", type=int, default=800)
    parser.add_argument("--smoke-iterations", type=int, default=20)
    parser.add_argument(
        "--canonical",
        action="store_true",
        help=(
            "Mark this successful run as the canonical committed TM-003A rerun. "
            "Never inferred from git_dirty=false."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke = bool(args.smoke)
    canonical_requested = bool(args.canonical)
    if canonical_requested and smoke:
        raise AssertionError("refusing --canonical with --smoke")
    experiment_id = SMOKE_EXPERIMENT_ID if smoke else EXPERIMENT_ID
    baseline_cfg = load_config("configs/baseline.yaml")
    tm_cfg = load_config("configs/tm003a.yaml")
    assert_frozen_tm003a_metadata(tm_cfg)
    if not smoke:
        assert_configs_match(baseline_cfg, tm_cfg)
    if canonical_requested:
        assert_canonical_run_allowed(collect_run_provenance())

    fp_before = [file_fingerprint(p) for p in PROTECTED]
    exp004_summary = ROOT / "outputs/predictions/EXP-004/summary.json"
    if not smoke:
        baseline_folds = load_baseline_folds(exp004_summary)
        exp004_used = json.loads(exp004_summary.read_text(encoding="utf-8"))["features_used"]
    else:
        baseline_folds = {}
        exp004_used = json.loads(exp004_summary.read_text(encoding="utf-8"))["features_used"] if exp004_summary.exists() else []

    dummy_cols = list(
        dict.fromkeys(list(exp004_used) + list(baseline_cfg["features"]["exclude_cols"]) + ["control_success"])
    ) if exp004_used else []
    if dummy_cols:
        used_b, _, _ = build_feature_lists(
            dummy_cols,
            target_col="control_success",
            exclude_cols=list(baseline_cfg["features"]["exclude_cols"]),
            categorical_cols=list(baseline_cfg["features"]["categorical_cols"]),
        )
    else:
        used_b = []
    used_tm = list(used_b) + [NEW_FEATURE]
    feature_assert = {
        "n_baseline": len(used_b),
        "n_tm003a": len(used_tm),
        "baseline_minus_tm": sorted(set(used_b) - set(used_tm)),
        "tm_minus_baseline": sorted(set(used_tm) - set(used_b)),
    }
    if used_b:
        if set(used_tm) - set(used_b) != {NEW_FEATURE} or set(used_b) - set(used_tm):
            raise AssertionError(f"feature set assert failed: {feature_assert}")
        if set(exp004_used) != set(used_b):
            raise AssertionError("Reconstructed baseline features != EXP-004 features_used")
        if len(used_b) != 44:
            raise AssertionError(f"Expected EXP-004 44 features, found {len(used_b)}")

    if not TRACKMAN_PATH.exists():
        raise FileNotFoundError(f"Trackman history not found: {TRACKMAN_PATH}")
    tm = pd.read_csv(
        TRACKMAN_PATH,
        encoding="utf-8-sig",
        usecols=list(TRACKMAN_IVB_REQUIRED_COLS),
        low_memory=False,
    )
    if set(tm["pitch_type_group"].astype(str).str.lower().str.strip().unique()) - {
        "fastball",
        "breaking",
        "offspeed",
        "other",
    }:
        raise AssertionError("Unexpected pitch_type_group values in Trackman")

    config_path: str | Path = "configs/tm003a.yaml"
    if smoke:
        raw_train = ROOT / tm_cfg["data"]["train_path"]
        smoke_dir = ensure_dir(ROOT / "data/processed/tm003a_smoke")
        smoke_train = subsample_train(raw_train, args.smoke_rows_per_season, seed=42)
        smoke_train_path = smoke_dir / "train.csv"
        smoke_train.to_csv(smoke_train_path, index=False)
        config_path = write_smoke_config(tm_cfg, smoke_train_path, args.smoke_iterations)

    profile_cache: dict[int, dict[str, Any]] = {}
    diag_store: list[dict[str, Any]] = []
    corr_store: list[dict[str, Any]] = []
    fallback_store: list[dict[str, Any]] = []
    result = run_tree_experiment(
        experiment_id=experiment_id,
        model_name="catboost",
        backend="catboost",
        config_path=config_path,
        params_key="catboost",
        fit_model=fit_catboost,
        model_filename="catboost_primary.joblib",
        sample_weight_by_season=dict(SEASON_WEIGHTS),
        extra_exclude_cols=None,
        extra_numeric_cols=[NEW_FEATURE],
        fold_extra_features=make_fold_extra_features(
            tm, profile_cache, diag_store, corr_store, fallback_store
        ),
        skip_experiment_log=True,
        log_overrides={
            "purpose": "tm003a_smoke" if smoke else "tm003a_expected_ivb_by_count_vs_exp004",
            "next_action": "GPT/Codex review of TM-003A; no TM bundle; no Public LB; stop.",
        },
    )

    if DIAGNOSTIC_FEATURE in result["features_used"]:
        raise AssertionError("Diagnostic tm_expected_ivb_global leaked into model features")
    if used_b:
        if set(result["features_used"]) - set(exp004_used) != {NEW_FEATURE}:
            raise AssertionError("Trained feature delta is not exactly +tm_expected_ivb_by_count")
        if set(exp004_used) - set(result["features_used"]):
            raise AssertionError("TM-003A dropped baseline features; aborting")
        if int(result["n_features_used"]) != 45:
            raise AssertionError(f"Expected 45 features, got {result['n_features_used']}")

    fp_after = [file_fingerprint(p) for p in PROTECTED]
    if fp_before != fp_after:
        raise AssertionError("Protected EXP-004/FE/TM-002A artifacts changed during TM-003A.")

    oof_tm = pd.read_csv(ROOT / "outputs/predictions" / experiment_id / "oof_holdout_predictions.csv")
    if oof_tm["prediction"].isna().any() or np.isinf(oof_tm["prediction"].to_numpy()).any():
        raise AssertionError("TM-003A predictions contain NaN/inf")
    if ((oof_tm["prediction"] < 0) | (oof_tm["prediction"] > 1)).any():
        raise AssertionError("TM-003A predictions outside [0, 1]")
    if not oof_tm["row_id"].is_unique:
        raise AssertionError("TM-003A row_id not unique")

    leakage_ok = True
    for row in result["fold_metrics"]:
        extra = row.get("fold_extra") or {}
        valid_season = int(extra.get("valid_season") or row["valid_seasons"][0])
        valid_tm = extra.get("tm_seasons_valid_rows") or []
        if any(int(s) >= valid_season for s in valid_tm) or 2025 in {int(s) for s in valid_tm}:
            leakage_ok = False
            raise AssertionError(f"{row['fold_id']} Trackman leakage: seasons={valid_tm}")
        by_season = extra.get("tm_seasons_by_row_season") or {}
        for row_season, seasons in by_season.items():
            if any(int(s) >= int(row_season) for s in seasons):
                leakage_ok = False
                raise AssertionError(
                    f"row-level leak season={row_season} used TM {seasons}"
                )

    out_dir = ensure_dir(ROOT / "outputs" / "predictions" / experiment_id)
    diag_df = pd.DataFrame(diag_store)
    corr_df = pd.DataFrame(corr_store)
    fallback_df = pd.DataFrame(fallback_store)
    diag_df.to_csv(out_dir / "feature_diagnostics.csv", index=False)
    corr_df.to_csv(out_dir / "duplicate_correlations.csv", index=False)
    fallback_df.to_csv(out_dir / "fallback_diagnostics.csv", index=False)
    overall_fallback = aggregate_overall_fallback(fallback_store)
    cell_n = collect_cell_n_diagnostics(result["fold_metrics"])
    nonfinite = count_nonfinite_large_cells(profile_cache)
    save_json(overall_fallback, out_dir / "fallback_overall.json")
    save_json(cell_n, out_dir / "cell_n_diagnostics.json")
    save_json(nonfinite, out_dir / "nonfinite_large_cell_fallback.json")

    fmap = {r["fold_id"]: r for r in result["fold_metrics"]}
    max_abs_corrs: dict[str, float] = {}
    if len(corr_df):
        for target, part in corr_df.groupby("target"):
            max_abs_corrs[str(target)] = float(part["pearson"].abs().max())

    smoke_integrity = {
        "exit_gate": "ok",
        "n_oof": int(len(oof_tm)),
        "n_features": int(result["n_features_used"]),
        "tm003a_feature": NEW_FEATURE,
        "duplicate_row_id": False,
        "prediction_min": float(oof_tm["prediction"].min()),
        "prediction_max": float(oof_tm["prediction"].max()),
        "prediction_nan": int(oof_tm["prediction"].isna().sum()),
        "fold_n_train": {k: int(fmap[k]["n_train"]) for k in FOLD_ORDER if k in fmap},
        "fold_n_valid": {k: int(fmap[k]["n_valid"]) for k in FOLD_ORDER if k in fmap},
        "artifact_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "rerun_command": (
            "python -m scripts.run_tm003a_expected_ivb --smoke"
            if smoke
            else (
                CANONICAL_RERUN_COMMAND
                if canonical_requested
                else RERUN_COMMAND
            )
        ),
    }

    if smoke:
        save_json(
            {
                "experiment_id": experiment_id,
                "smoke": True,
                "seed": 42,
                "min_n": MIN_CELL_N,
                "integrity": smoke_integrity,
                "fold_metrics": result["fold_metrics"],
                "fallback": fallback_store,
                "fallback_overall": overall_fallback,
                "cell_n": cell_n,
                "nonfinite_large_cell_fallback": nonfinite,
                "leakage_ok": leakage_ok,
                "protected_fingerprints_unchanged": True,
                "canonical": False,
                "run_kind": RUN_KIND_NON_CANONICAL,
            },
            out_dir / "smoke_summary.json",
        )
        print(json.dumps({"experiment_id": experiment_id, "smoke": True, **smoke_integrity}, indent=2))
        return

    oof_base = pd.read_csv(ROOT / "outputs/predictions/EXP-004/oof_holdout_predictions.csv")
    if set(oof_tm["row_id"]) != set(oof_base["row_id"]):
        raise AssertionError("TM-003A row_id set != EXP-004")
    ece_by_fold = {}
    for fold_id, part in oof_tm.groupby("fold_id"):
        ece_by_fold[str(fold_id)] = quantile_10_ece(
            part["control_success"].to_numpy(float),
            part["prediction"].to_numpy(float),
        )

    b_raw = np.array([float(baseline_folds[k]["raw_brier_skill_score"]) for k in FOLD_ORDER])
    t_raw = np.array([float(fmap[k]["raw_brier_skill_score"]) for k in FOLD_ORDER])
    deltas = {
        "d2022": float(t_raw[0] - b_raw[0]),
        "d2023": float(t_raw[1] - b_raw[1]),
        "d2024": float(t_raw[2] - b_raw[2]),
        "dmean": float(np.mean(t_raw) - np.mean(b_raw)),
        "dstd": float(np.std(t_raw, ddof=0) - np.std(b_raw, ddof=0)),
        "dworst": float(np.min(t_raw) - np.min(b_raw)),
        "dbest": float(np.max(t_raw) - np.max(b_raw)),
    }
    rule = evaluate_predeclared_rule(deltas, leakage_ok=leakage_ok, max_abs_corrs=max_abs_corrs)
    report_path = ROOT / REPORT_RELPATH
    if canonical_requested:
        assert_canonical_run_allowed(collect_run_provenance())
    provenance = build_run_provenance(
        experiment_id=EXPERIMENT_ID,
        canonical_requested=canonical_requested,
        smoke=False,
        config_path=CONFIG_RELPATH,
        artifact_dir=out_dir,
        report_path=report_path,
    )
    comparison = {
        "experiment_id": EXPERIMENT_ID,
        "baseline": BASELINE_ID,
        "new_feature": NEW_FEATURE,
        "diagnostic_feature": DIAGNOSTIC_FEATURE,
        "diagnostic_is_model_input": False,
        "min_n": MIN_CELL_N,
        "formula": "plain_sum",
        "cutoff": "trackman.season < row.season",
        "feature_assert": feature_assert,
        "deltas": deltas,
        "fold_raw_bss": {k: float(fmap[k]["raw_brier_skill_score"]) for k in FOLD_ORDER},
        "fold_official_bss": {k: float(fmap[k]["official_brier_skill_score"]) for k in FOLD_ORDER},
        "fold_brier": {k: float(fmap[k]["brier_score"]) for k in FOLD_ORDER},
        "fold_quantile_10_ece": ece_by_fold,
        "cv_mean": float(np.mean(t_raw)),
        "cv_std": float(np.std(t_raw, ddof=0)),
        "worst_fold": float(np.min(t_raw)),
        "best_fold": float(np.max(t_raw)),
        "fallback": fallback_store,
        "fallback_overall": overall_fallback,
        "cell_n": cell_n,
        "nonfinite_large_cell_fallback": nonfinite,
        "max_abs_corrs": max_abs_corrs,
        "predeclared_rule": rule,
        "protected_fingerprints_unchanged": True,
        "leakage_ok": leakage_ok,
        "public_lb": "not_submitted",
        "seed": 42,
        "integrity": smoke_integrity,
        "run_provenance": provenance,
    }
    save_json(comparison, out_dir / "comparison_vs_exp004.json")
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_provenance"] = provenance
        summary["fallback_overall"] = overall_fallback
        summary["cell_n"] = cell_n
        summary["nonfinite_large_cell_fallback"] = nonfinite
        save_json(summary, summary_path)
    write_report(
        baseline_folds=baseline_folds,
        tm_folds=result["fold_metrics"],
        diag_df=diag_df,
        corr_df=corr_df,
        fallback_df=fallback_df,
        n_base_feat=len(used_b),
        n_tm_feat=len(result["features_used"]),
        feature_assert=feature_assert,
        rule=rule,
        leakage_ok=leakage_ok,
        report_path=report_path,
        seed=42,
        provenance=provenance,
        overall_fallback=overall_fallback,
        cell_n=cell_n,
        nonfinite=nonfinite,
    )
    log_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_experiment_log(
        tm_cfg["output"]["experiments_path"],
        {
            "experiment_id": EXPERIMENT_ID,
            "date": log_date,
            "purpose": "tm003a_expected_ivb_by_count_vs_exp004",
            "data_version": tm_cfg.get("project", {}).get("data_version", ""),
            "features": to_json_text(
                {
                    "n_before": len(used_b),
                    "n_after": len(result["features_used"]),
                    "new_feature": NEW_FEATURE,
                    "diagnostic_feature": DIAGNOSTIC_FEATURE,
                    "delta": feature_assert,
                }
            ),
            "model": "catboost",
            "params": to_json_text(
                {
                    **tm_cfg["models"]["params"]["catboost"],
                    "sample_weight_by_season": {
                        str(k): float(v) for k, v in sorted(SEASON_WEIGHTS.items())
                    },
                }
            ),
            "cv_strategy": to_json_text(tm_cfg["cv"]),
            "metric": "brier_skill_score",
            "metric_direction": "maximize",
            "cv_score": result["fold_raw_bss_mean"],
            "cv_std": result["fold_raw_bss_std"],
            "holdout_score": fmap["primary_2024"]["official_brier_skill_score"],
            "public_lb_score": "not_submitted",
            "cv_lb_gap": "",
            "seed": 42,
            "notes": to_json_text(
                {
                    "baseline": BASELINE_ID,
                    "new_feature": NEW_FEATURE,
                    "min_n": MIN_CELL_N,
                    "formula": "plain_sum",
                    "cutoff": "trackman.season < row.season",
                    "fold_scores": {k: float(fmap[k]["raw_brier_skill_score"]) for k in FOLD_ORDER},
                    "cv_mean": result["fold_raw_bss_mean"],
                    "cv_std": result["fold_raw_bss_std"],
                    "worst_fold": float(np.min(t_raw)),
                    "best_fold": float(np.max(t_raw)),
                    "delta_vs_exp004": deltas,
                    "fallback": fallback_store,
                    "fallback_overall": overall_fallback,
                    "cell_n": cell_n,
                    "nonfinite_large_cell_fallback": nonfinite,
                    "max_abs_corrs": max_abs_corrs,
                    "predeclared_rule_bucket": rule["predeclared_rule_bucket"],
                    "predeclared_rule_note": rule["note"],
                    "leakage_check": "row_level_tm_season_lt_row_season_no_id_join_no_current_pitch_no_test_agg",
                    "public_lb": "not_submitted",
                    "run_kind": provenance["run_kind"],
                    "canonical": provenance["canonical"],
                    "command": provenance["command"],
                    "config_path": provenance["config_path"],
                    "artifact_path": provenance["artifact_path"],
                    "report_path": provenance["report_path"],
                    "validation_folds": provenance["validation_folds"],
                }
            ),
            "next_action": (
                "Canonical TM-003A run logged; do not auto-run other Trackman features; no Public LB."
                if provenance["canonical"]
                else "GPT/Codex review of TM-003A; canonical rerun only on explicit instruction; no Public LB."
            ),
            "status": "success",
            "error": "",
        },
    )
    if provenance["canonical"]:
        mark_prior_tm003a_rows_superseded(tm_cfg["output"]["experiments_path"])
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "predeclared_rule_bucket": rule["predeclared_rule_bucket"],
                "run_kind": provenance["run_kind"],
                "canonical": provenance["canonical"],
                **deltas,
                "n_features": result["n_features_used"],
                "leakage_ok": leakage_ok,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
