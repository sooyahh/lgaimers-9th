from __future__ import annotations

"""VAL-AUDIT-001: expanding-window target-mean diagnosis (no model training)."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import load_competition_data
from src.evaluate import build_season_holdout, evaluate_predictions, resolve_metric_settings
from src.utils import ensure_dir, load_config, resolve_path, save_json


AUDIT_ID = "VAL-AUDIT-001"

FOLDS = [
    {
        "fold_id": "fold_2022",
        "name": "expanding_to_2022",
        "train_seasons": [2019, 2020, 2021],
        "valid_seasons": [2022],
    },
    {
        "fold_id": "fold_2023",
        "name": "expanding_to_2023",
        "train_seasons": [2019, 2020, 2021, 2022],
        "valid_seasons": [2023],
    },
    {
        "fold_id": "fold_2024",
        "name": "expanding_to_2024",
        "train_seasons": [2019, 2020, 2021, 2022, 2023],
        "valid_seasons": [2024],
    },
]


def _run_fold(
    train_df: pd.DataFrame,
    *,
    season_col: str,
    target_col: str,
    fold_spec: dict[str, Any],
    scale: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    fold = build_season_holdout(
        train_df,
        season_col,
        fold_spec["train_seasons"],
        fold_spec["valid_seasons"],
        name=fold_spec["name"],
    )
    y_train = train_df.iloc[fold.train_idx][target_col]
    y_valid = train_df.iloc[fold.valid_idx][target_col]
    train_mean = float(y_train.mean())
    valid_mean = float(y_valid.mean())
    pred = np.full(len(fold.valid_idx), train_mean, dtype=float)
    metrics = evaluate_predictions(y_valid, pred, scale=scale)
    elapsed = float(time.perf_counter() - started)
    return {
        "fold_id": fold_spec["fold_id"],
        "fold_name": fold_spec["name"],
        "train_seasons": list(fold.train_seasons),
        "valid_seasons": list(fold.valid_seasons),
        "n_train": int(len(fold.train_idx)),
        "n_valid": int(len(fold.valid_idx)),
        "train_target_mean": train_mean,
        "valid_target_mean": valid_mean,
        "mean_difference_valid_minus_train": float(valid_mean - train_mean),
        "prediction_mean": float(pred.mean()),
        "brier_score": float(metrics["brier_score"]),
        "raw_brier_skill_score": float(metrics["raw_brier_skill_score"]),
        "official_brier_skill_score": float(metrics["official_brier_skill_score"]),
        "elapsed_seconds": elapsed,
    }


def _season_rates(train_df: pd.DataFrame, season_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    for season, part in train_df.groupby(season_col, sort=True):
        rows.append(
            {
                "season": int(season),
                "n_rows": int(len(part)),
                "target_mean": float(part[target_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_season_validation_report(
    path: Path,
    *,
    fold_df: pd.DataFrame,
    season_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Season Validation Audit (VAL-AUDIT-001)")
    lines.append("")
    lines.append(f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("- model: training-split **target mean** only (EXP-000 logic reused)")
    lines.append("- no Logistic Regression / GBDT / Trackman / feature engineering")
    lines.append("")
    lines.append("## Per-season target rate (full train.csv)")
    lines.append("")
    lines.append("| season | n_rows | target_mean |")
    lines.append("|---|---:|---:|")
    for row in season_df.itertuples(index=False):
        lines.append(f"| {row.season} | {row.n_rows} | {row.target_mean:.6f} |")
    lines.append("")
    lines.append("## Expanding-window target-mean folds")
    lines.append("")
    lines.append(
        "| fold | train seasons | valid | n_train | n_valid | train mean | valid mean | "
        "Δ(valid−train) | pred mean | Brier | raw BSS | official BSS | elapsed_s |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in fold_df.itertuples(index=False):
        trains = "-".join(str(s) for s in row.train_seasons)
        valids = ",".join(str(s) for s in row.valid_seasons)
        lines.append(
            f"| {row.fold_id} | {trains} | {valids} | {row.n_train} | {row.n_valid} | "
            f"{row.train_target_mean:.6f} | {row.valid_target_mean:.6f} | "
            f"{row.mean_difference_valid_minus_train:+.6f} | {row.prediction_mean:.6f} | "
            f"{row.brier_score:.6f} | {row.raw_brier_skill_score:.3f} | "
            f"{row.official_brier_skill_score:.3f} | {row.elapsed_seconds:.3f} |"
        )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- mean raw BSS across folds: `{summary['mean_raw_bss']:.6f}`")
    lines.append(f"- std raw BSS across folds: `{summary['std_raw_bss']:.6f}`")
    lines.append(f"- mean official BSS across folds: `{summary['mean_official_bss']:.6f}`")
    lines.append(f"- folds with official BSS == 0: `{summary['n_folds_official_zero']}` / `{summary['n_folds']}`")
    lines.append("")
    lines.append("## Interpretation (evidence-separated)")
    lines.append("")
    lines.append("### 1) Year-to-year target drift (observed)")
    lines.append("")
    lines.append(
        "Per-season `control_success` means are not constant. "
        f"Observed season means: "
        + ", ".join(f"{int(r.season)}={r.target_mean:.4f}" for r in season_df.itertuples(index=False))
        + ". This supports a **seasonal target distribution shift** hypothesis for constant-mean predictors."
    )
    lines.append("")
    lines.append("### 2) Specific-season anomaly vs repeated pattern")
    lines.append("")
    deltas = fold_df["mean_difference_valid_minus_train"].to_numpy()
    raws = fold_df["raw_brier_skill_score"].to_numpy()
    lines.append(
        f"- Fold mean gaps (valid−train): "
        + ", ".join(f"{fid}={d:+.4f}" for fid, d in zip(fold_df["fold_id"], deltas))
        + "."
    )
    # Evidence-only statements
    all_negative_gap = bool((deltas < 0).all())
    all_raw_negative = bool((raws < 0).all())
    if all_negative_gap and all_raw_negative:
        lines.append(
            "- Across all three expanding folds, valid mean was **below** train mean and raw BSS was **negative**. "
            "This is a **repeated pattern** under the target-mean baseline, not a 2024-only artifact."
        )
    else:
        lines.append(
            "- Gaps / raw BSS signs are mixed across folds; treat 2024 as one observation among several, "
            "not automatically as a unique anomaly."
        )
    # Highlight relative magnitude without claiming causation
    abs_deltas = np.abs(deltas)
    max_i = int(np.argmax(abs_deltas))
    lines.append(
        f"- Largest |valid−train| gap among these folds: `{fold_df.iloc[max_i]['fold_id']}` "
        f"(`{deltas[max_i]:+.6f}`)."
    )
    lines.append("")
    lines.append("### 3) Fold-wise raw BSS stability")
    lines.append("")
    lines.append(
        f"- raw BSS values: "
        + ", ".join(f"{fid}={v:.3f}" for fid, v in zip(fold_df["fold_id"], raws))
        + f"; mean=`{summary['mean_raw_bss']:.3f}`, std=`{summary['std_raw_bss']:.3f}`."
    )
    lines.append(
        "- Official BSS is clipped with `max(0, ·)`, so when raw BSS is negative on every fold, "
        "official BSS collapses to 0 and **cannot rank** constant-mean folds. Diagnostics must use raw BSS / Brier / mean gap."
    )
    lines.append("")
    lines.append("### 4) Risk of using 2024 holdout alone as primary")
    lines.append("")
    lines.append(
        "- EXP-000 already showed 2019–2023 mean → 2024 valid yields official BSS = 0 with negative raw BSS."
    )
    lines.append(
        "- This audit shows earlier expanding folds also produce negative raw BSS under the same constant baseline, "
        "so 2024 is consistent with a broader seasonal-mean mismatch problem."
    )
    lines.append(
        "- **Risk:** selecting models only on 2024 holdout can overfit to one season’s base rate. "
        "A single holdout cannot distinguish true skill from luck in matching that season’s rate."
    )
    lines.append("")
    lines.append("### 5) Need for expanding-window average as an auxiliary criterion")
    lines.append("")
    lines.append(
        "- Recommendation (non-final validation policy): keep 2024 season holdout as a **primary stress test**, "
        "and report **expanding-window mean ± std of raw BSS / Brier / mean-gap** as auxiliary diagnostics."
    )
    lines.append(
        "- Do not average official BSS alone while it is floor-clipped at 0 for weak baselines."
    )
    lines.append(
        "- Validation strategy is still **not finalized**; game-id absence and empty `order_cols` remain unchanged."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{summary['fold_metrics_path']}`")
    lines.append(f"- `{summary['summary_json_path']}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    metric_settings = resolve_metric_settings(config.get("task", {}))
    scale = float(metric_settings["scale"])
    data_cfg = config["data"]
    target_col = data_cfg["target_col"]
    season_col = data_cfg["season_col"]

    loaded = load_competition_data(config)
    train_df = loaded["train"]

    fold_rows = [
        _run_fold(
            train_df,
            season_col=season_col,
            target_col=target_col,
            fold_spec=spec,
            scale=scale,
        )
        for spec in FOLDS
    ]
    fold_df = pd.DataFrame(fold_rows)
    season_df = _season_rates(train_df, season_col, target_col)

    out_dir = ensure_dir(resolve_path(config["output"]["predictions_dir"]) / AUDIT_ID)
    fold_path = out_dir / "fold_metrics.csv"
    # Serialize list columns as JSON strings for CSV readability
    fold_csv = fold_df.copy()
    fold_csv["train_seasons"] = fold_csv["train_seasons"].map(lambda x: json.dumps(x))
    fold_csv["valid_seasons"] = fold_csv["valid_seasons"].map(lambda x: json.dumps(x))
    fold_csv.to_csv(fold_path, index=False)

    raw_vals = fold_df["raw_brier_skill_score"].to_numpy(dtype=float)
    official_vals = fold_df["official_brier_skill_score"].to_numpy(dtype=float)
    summary = {
        "audit_id": AUDIT_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "target_mean",
        "n_folds": int(len(fold_df)),
        "mean_raw_bss": float(np.mean(raw_vals)),
        "std_raw_bss": float(np.std(raw_vals, ddof=0)),
        "mean_official_bss": float(np.mean(official_vals)),
        "n_folds_official_zero": int((official_vals == 0.0).sum()),
        "season_target_rates": season_df.to_dict(orient="records"),
        "folds": fold_rows,
        "fold_metrics_path": str(fold_path),
        "notes": [
            "Valid predictions use each fold's training-split target mean only.",
            "Full-data target mean is never used for validation predictions.",
            "order_cols remain empty; row_id not treated as a time key.",
        ],
    }
    summary_path = out_dir / "summary.json"
    summary["summary_json_path"] = str(summary_path)
    save_json(summary, summary_path)

    report_path = resolve_path("reports/season_validation_audit.md")
    ensure_dir(report_path.parent)
    _write_season_validation_report(
        report_path, fold_df=fold_df, season_df=season_df, summary=summary
    )
    summary["report_path"] = str(report_path)
    # refresh summary with report path
    save_json(summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    result = run_audit(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
