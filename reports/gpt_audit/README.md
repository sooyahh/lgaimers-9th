# GPT Data Analysis — Audit Package

- prepared_at: 2026-08-07
- updated_at: 2026-08-13 (VAL-EXP006-SECONDARY-001 logged; EXP-006 secondary verdict = hold)
- scope: EXP-000 ~ EXP-006 + VAL-EXP006-SECONDARY-001
- constraint: no new training in this documentation update; VAL scores copied from existing artifacts

## Contents

| file | role |
|---|---|
| `experiment_summary.csv` | EXP-level metrics, deltas, status |
| `experiment_handoff.md` | narrative for independent review |
| `config_manifest.csv` | config / script linkage per EXP |
| `data_column_inventory.csv` | train/test column list for analysis |
| `data_paths.md` | raw data + existing artifact paths |
| `oof_inventory.csv` | which OOF/holdout preds exist and columns |
| `oof/` | enriched prediction tables (joined context from train by `row_id` only) |
| `SOURCE_INDEX.md` | pointers to original reports/outputs (not copied) |

## Audit checklist (intended)

1. validation reliability
2. EXP performance trajectory
3. fold stability
4. leakage risk
5. model error patterns
6. temporal / generalization issues
7. next-experiment priority

## Hard rules for the analyst

- Prefer **raw BSS** over official BSS when official is clipped at 0.
- Do not treat missing Public LB as 0; it is **unrecorded**.
- Do not invent adopt/reject except where recorded. EXP-006 vs EXP-005 secondary is **hold** (`VAL-EXP006-SECONDARY-001`).
- EXP-005/006 **2022/2023** scores come from `VAL-EXP006-SECONDARY-001`, not from original EXP-005/006 OOF files. 2024 remains the EXP-006 tune holdout.
- Use `is_comparable_cv` / `score_scope` in `experiment_summary.csv`: only rows with `is_comparable_cv=true` have expanding 3-fold CV means.
- EXP-006 is still the **highest observed primary 2024 score**; secondary validation **did not** confirm adopt.
- EXP-000 secondary scores: see `source` column (VAL-AUDIT-001), not EXP-000 prediction files.
)
