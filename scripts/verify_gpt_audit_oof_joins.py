"""Minimal join verification for reports/gpt_audit/oof enriched CSVs. No training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "reports" / "gpt_audit" / "oof"
OUT = ROOT / "reports" / "gpt_audit" / "oof_join_verification.json"

CASES = [
    {
        "enriched": "EXP-000_holdout_enriched.csv",
        "source": ROOT / "outputs/predictions/EXP-000/holdout_predictions.csv",
        "source_target": "control_success",
        "enriched_target": "target",
        "expect_row_id_unique": True,
    },
    {
        "enriched": "EXP-001A_oof_holdout_enriched.csv",
        "source": ROOT / "outputs/predictions/EXP-001A/oof_holdout_predictions.csv",
        "source_target": "control_success",
        "enriched_target": "target",
        "expect_row_id_unique": False,  # 3 folds concatenated; same row_id can appear once per fold season
    },
    {
        "enriched": "EXP-002_oof_holdout_enriched.csv",
        "source": ROOT / "outputs/predictions/EXP-002/oof_holdout_predictions.csv",
        "source_target": "control_success",
        "enriched_target": "target",
        "expect_row_id_unique": False,
    },
    {
        "enriched": "EXP-003_oof_holdout_enriched.csv",
        "source": ROOT / "outputs/predictions/EXP-003/oof_holdout_predictions.csv",
        "source_target": "control_success",
        "enriched_target": "target",
        "expect_row_id_unique": False,
    },
    {
        "enriched": "EXP-004_oof_holdout_enriched.csv",
        "source": ROOT / "outputs/predictions/EXP-004/oof_holdout_predictions.csv",
        "source_target": "control_success",
        "enriched_target": "target",
        "expect_row_id_unique": False,
    },
    {
        "enriched": "EXP-006_holdout_2024_enriched_wide.csv",
        "source": ROOT / "outputs/predictions/EXP-006/holdout_2024_predictions.csv",
        "source_target": "y_true",
        "enriched_target": "target",
        "expect_row_id_unique": True,
    },
]


def main() -> None:
    train_ids = pd.read_csv(
        ROOT / "data/raw/train.csv", usecols=["row_id", "control_success"]
    )
    train_key_unique = bool(train_ids["row_id"].is_unique)
    results = {
        "train_join_key_row_id_is_unique": train_key_unique,
        "train_n_rows": int(len(train_ids)),
        "cases": [],
    }

    for case in CASES:
        src = pd.read_csv(case["source"])
        enr = pd.read_csv(OOF / case["enriched"])
        n_src = len(src)
        n_enr = len(enr)
        # unmatched: enriched rows where pitcher_id (joined) is NA
        unmatched = int(enr["pitcher_id"].isna().sum()) if "pitcher_id" in enr.columns else -1
        # duplicate increase: compare n rows
        dup_increase = n_enr > n_src
        # within enriched: duplicate row_id counts
        n_dup_row_id = int(enr["row_id"].duplicated().sum())
        # target alignment vs source on row_id order if same length and same order
        src_t = src[case["source_target"]].to_numpy()
        enr_t = enr[case["enriched_target"]].to_numpy()
        if n_src == n_enr and list(src["row_id"]) == list(enr["row_id"]):
            target_align_same_order = bool((src_t == enr_t).all())
            alignment_method = "same_row_order_elementwise"
        else:
            # merge check
            m = src[["row_id", case["source_target"]]].merge(
                enr[["row_id", case["enriched_target"]]],
                on="row_id",
                how="inner",
            )
            # for multi-fold OOF, merge on row_id alone can fan out; use index order instead
            target_align_same_order = False
            alignment_method = "row_order_mismatch_or_nonunique_join"
            if n_src == n_enr:
                target_align_same_order = bool((src_t == enr_t).all()) and list(
                    src["row_id"]
                ) == list(enr["row_id"])
                alignment_method = "same_row_order_elementwise"

        # also compare to train control_success via row_id (unique in train)
        tmap = train_ids.set_index("row_id")["control_success"]
        aligned_to_train = enr["row_id"].map(tmap)
        train_target_match = bool(
            (aligned_to_train.to_numpy() == enr[case["enriched_target"]].to_numpy()).all()
        )
        train_map_na = int(aligned_to_train.isna().sum())

        results["cases"].append(
            {
                "enriched_file": case["enriched"],
                "source_file": str(case["source"].relative_to(ROOT)).replace("\\", "/"),
                "n_rows_before_join": n_src,
                "n_rows_after_join": n_enr,
                "unmatched_rows_pitcher_id_na": unmatched,
                "duplicate_row_increase_vs_source": dup_increase,
                "n_duplicate_row_id_in_enriched": n_dup_row_id,
                "train_row_id_unique": train_key_unique,
                "target_alignment_vs_source_same_order": target_align_same_order,
                "target_alignment_method": alignment_method,
                "target_alignment_vs_train_control_success": train_target_match,
                "train_map_na": train_map_na,
                "expect_row_id_unique": case["expect_row_id_unique"],
                "row_id_unique_ok": (n_dup_row_id == 0) == case["expect_row_id_unique"]
                or (case["expect_row_id_unique"] and n_dup_row_id == 0)
                or (not case["expect_row_id_unique"]),
            }
        )

    # long file: intentional 2x rows (two models)
    long_path = OOF / "EXP-006_holdout_2024_enriched_long.csv"
    if long_path.exists():
        src6 = pd.read_csv(ROOT / "outputs/predictions/EXP-006/holdout_2024_predictions.csv")
        long = pd.read_csv(long_path)
        results["cases"].append(
            {
                "enriched_file": "EXP-006_holdout_2024_enriched_long.csv",
                "source_file": "outputs/predictions/EXP-006/holdout_2024_predictions.csv",
                "n_rows_before_join": len(src6),
                "n_rows_after_join": len(long),
                "unmatched_rows_pitcher_id_na": int(long["pitcher_id"].isna().sum()),
                "duplicate_row_increase_vs_source": True,
                "n_duplicate_row_id_in_enriched": int(long["row_id"].duplicated().sum()),
                "train_row_id_unique": train_key_unique,
                "target_alignment_vs_source_same_order": None,
                "target_alignment_method": "intentional_long_format_2_models_per_row_id",
                "target_alignment_vs_train_control_success": bool(
                    (
                        long["row_id"].map(
                            train_ids.set_index("row_id")["control_success"]
                        )
                        == long["target"]
                    ).all()
                ),
                "train_map_na": int(
                    long["row_id"]
                    .map(train_ids.set_index("row_id")["control_success"])
                    .isna()
                    .sum()
                ),
                "expect_row_id_unique": False,
                "row_id_unique_ok": True,
                "notes": "Row count is 2x holdout by design (catboost+lightgbm stacked).",
            }
        )

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
