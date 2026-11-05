from __future__ import annotations

"""EXP-002: LightGBM baseline with native categorical features."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tree_exp_common import run_tree_experiment
from src.models import build_model

EXPERIMENT_ID = "EXP-002"


def fit_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    categorical_cols: list[str],
    params: dict[str, Any],
    seed: int,
) -> Any:
    model = build_model("lightgbm", params, seed)
    # Prefer column-name categorical_feature when X is a DataFrame.
    model.fit(X_tr, y_tr, categorical_feature=categorical_cols)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP-002 LightGBM baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    result = run_tree_experiment(
        experiment_id=EXPERIMENT_ID,
        model_name="lightgbm",
        backend="lightgbm",
        config_path=args.config,
        params_key="lightgbm",
        fit_model=fit_lightgbm,
        model_filename="lightgbm_primary.joblib",
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "primary_2024": result["primary_2024"],
                "oof": {
                    "brier_score": result["oof"]["brier_score"],
                    "raw_brier_skill_score": result["oof"]["raw_brier_skill_score"],
                    "official_brier_skill_score": result["oof"]["official_brier_skill_score"],
                },
                "fold_raw_bss_mean": result["fold_raw_bss_mean"],
                "fold_raw_bss_std": result["fold_raw_bss_std"],
                "comparison_primary_2024": result["comparison_primary_2024"],
                "elapsed_seconds_wall": result["elapsed_seconds_wall"],
                "summary_path": result.get("summary_path"),
                "report_path": result.get("report_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
