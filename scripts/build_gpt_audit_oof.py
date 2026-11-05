"""Build reports/gpt_audit/oof enriched prediction tables from existing artifacts only.

No model training. Joins train context columns by row_id onto already-saved predictions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "gpt_audit" / "oof"
INV = ROOT / "reports" / "gpt_audit" / "oof_inventory.csv"

CONTEXT_COLS = [
    "row_id",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "game_month",
    "game_dayofweek",
    "game_type",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "num_runners_on",
    "score_diff_pitcher_team",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_batter_success_rate",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv", usecols=CONTEXT_COLS)
    assert train["row_id"].is_unique

    inventory: list[dict] = []

    # EXP-000 holdout only
    src0 = ROOT / "outputs" / "predictions" / "EXP-000" / "holdout_predictions.csv"
    df0 = pd.read_csv(src0)
    df0 = df0.rename(columns={"control_success": "target", "split": "fold_id"})
    df0["experiment_id"] = "EXP-000"
    df0["model"] = "target_mean"
    merged0 = df0.merge(train, on="row_id", how="left", validate="many_to_one")
    assert merged0["pitcher_id"].notna().all()
    out0 = OUT / "EXP-000_holdout_enriched.csv"
    merged0.to_csv(out0, index=False)
    inventory.append(
        {
            "experiment_id": "EXP-000",
            "artifact_kind": "holdout_only",
            "source_path": str(src0.relative_to(ROOT)).replace("\\", "/"),
            "enriched_path": str(out0.relative_to(ROOT)).replace("\\", "/"),
            "n_rows": len(merged0),
            "source_columns": ",".join(df0.columns),
            "has_pitcher_batter_in_source": False,
            "enrichment": "left_join_train_context_by_row_id",
            "folds_present": "season_holdout_2024",
            "notes": "Primary 2024 only; no secondary OOF file for EXP-000",
        }
    )

    # EXP-001A .. EXP-004 full oof+holdout
    for exp, model in [
        ("EXP-001A", "logistic_regression"),
        ("EXP-002", "lightgbm"),
        ("EXP-003", "catboost"),
        ("EXP-004", "catboost"),
    ]:
        src = ROOT / "outputs" / "predictions" / exp / "oof_holdout_predictions.csv"
        df = pd.read_csv(src)
        df = df.rename(columns={"control_success": "target"})
        df["experiment_id"] = exp
        df["model"] = model
        merged = df.merge(train, on="row_id", how="left", validate="many_to_one")
        assert merged["pitcher_id"].notna().all()
        out = OUT / f"{exp}_oof_holdout_enriched.csv"
        merged.to_csv(out, index=False)
        inventory.append(
            {
                "experiment_id": exp,
                "artifact_kind": "oof_plus_primary_holdout",
                "source_path": str(src.relative_to(ROOT)).replace("\\", "/"),
                "enriched_path": str(out.relative_to(ROOT)).replace("\\", "/"),
                "n_rows": len(merged),
                "source_columns": "row_id,season,control_success,prediction,fold_id,split_type",
                "has_pitcher_batter_in_source": False,
                "enrichment": "left_join_train_context_by_row_id",
                "folds_present": "secondary_2022;secondary_2023;primary_2024",
                "notes": "Concatenated expanding valid rows; one prediction per valid row per fold set",
            }
        )

    # EXP-005: no row-level predictions
    inventory.append(
        {
            "experiment_id": "EXP-005",
            "artifact_kind": "missing_row_predictions",
            "source_path": "outputs/predictions/EXP-005/summary.json",
            "enriched_path": "",
            "n_rows": 0,
            "source_columns": "",
            "has_pitcher_batter_in_source": False,
            "enrichment": "none",
            "folds_present": "",
            "notes": "Only summary.json; no OOF/holdout prediction CSV was generated",
        }
    )

    # EXP-006 holdout CB/LGB preds
    src6 = ROOT / "outputs" / "predictions" / "EXP-006" / "holdout_2024_predictions.csv"
    df6 = pd.read_csv(src6)
    df6 = df6.rename(columns={"y_true": "target"})
    df6["experiment_id"] = "EXP-006"
    df6["fold_id"] = "primary_2024"
    df6["season"] = 2024
    # selected submit model = catboost; keep both preds for error-pattern comparison
    long_rows = []
    for model_col, model_name in [
        ("pred_catboost", "tuned_catboost"),
        ("pred_lightgbm", "tuned_lightgbm"),
    ]:
        part = df6[["row_id", "target", "season", "fold_id", "experiment_id"]].copy()
        part["model"] = model_name
        part["prediction"] = df6[model_col]
        long_rows.append(part)
    long6 = pd.concat(long_rows, ignore_index=True)
    # also wide enriched for convenience
    wide6 = df6.merge(train, on="row_id", how="left", validate="many_to_one")
    assert wide6["pitcher_id"].notna().all()
    out6w = OUT / "EXP-006_holdout_2024_enriched_wide.csv"
    out6l = OUT / "EXP-006_holdout_2024_enriched_long.csv"
    wide6.to_csv(out6w, index=False)
    long6.merge(train, on="row_id", how="left", validate="many_to_one").to_csv(
        out6l, index=False
    )
    inventory.append(
        {
            "experiment_id": "EXP-006",
            "artifact_kind": "holdout_only_multi_model",
            "source_path": str(src6.relative_to(ROOT)).replace("\\", "/"),
            "enriched_path": str(out6w.relative_to(ROOT)).replace("\\", "/")
            + ";"
            + str(out6l.relative_to(ROOT)).replace("\\", "/"),
            "n_rows": len(wide6),
            "source_columns": "row_id,y_true,pred_catboost,pred_lightgbm",
            "has_pitcher_batter_in_source": False,
            "enrichment": "left_join_train_context_by_row_id",
            "folds_present": "primary_2024",
            "notes": "No secondary OOF; wide keeps both preds; long stacks models",
        }
    )

    pd.DataFrame(inventory).to_csv(INV, index=False)
    print(f"Wrote inventory {INV}")
    print(f"Wrote enriched OOF under {OUT}")


if __name__ == "__main__":
    main()
