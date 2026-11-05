# Source index (read originals; do not modify)

## Metric / validation

- `src/metrics.py`
- `src/evaluate.py`
- `configs/baseline.yaml`
- `reports/validation_policy.md`
- `reports/validation_structure_audit.md`
- `reports/feature_safety_audit.md`
- `reports/season_validation_audit.md`

## Experiment registry

- `outputs/experiments.csv`
- `docs/GPT_HANDOFF.md` (prior narrative handoff)

## Per-EXP artifacts

| EXP | summary / report | predictions | models |
|---|---|---|---|
| EXP-000 | `outputs/predictions/EXP-000/report.md`, `metrics.json` | `holdout_predictions.csv` | `outputs/models/EXP-000/` |
| EXP-001A | `summary.json`, `report.md`, `fold_metrics.csv` | `oof_holdout_predictions.csv` | `outputs/models/EXP-001A/` |
| EXP-002 | same pattern | `oof_holdout_predictions.csv` | `outputs/models/EXP-002/` |
| EXP-003 | same pattern | `oof_holdout_predictions.csv` | `outputs/models/EXP-003/` |
| EXP-004 | same pattern | `oof_holdout_predictions.csv` | `outputs/models/EXP-004/` |
| EXP-005 | `summary.json`; `outputs/eda/full_train_experiments.md|json` | **no row-level OOF file** | `outputs/models/EXP-005/R_L_*` |
| EXP-006 | `summary.json`; `outputs/eda/tuning_ensemble.md|json` | `holdout_2024_predictions.csv` only | inside `EXP-006_submit.zip` |
| VAL-AUDIT-001 | `outputs/predictions/VAL-AUDIT-001/summary.json` | — | — |
| VAL-EXP006-SECONDARY-001 | `outputs/predictions/VAL-EXP006-SECONDARY-001/summary.json` | `oof_predictions.parquet` (local; gitignored) | — |

## Diagnostics (not formal EXP IDs)

- `outputs/eda/leakage_check.md`
- `outputs/eda/gap_diagnosis.md|json`
- `outputs/eda/unseen_id_experiments.md|json`
- `outputs/eda/regime_shift_experiments.md|json`
- `outputs/eda/season_extrapolation_experiments.md|json`
- `outputs/eda/target_regime_check.md`
- `outputs/eda/fold_comparison.md`
- `outputs/eda/data_integrity.md`

## Submissions (smoke-verified locally; Public LB unknown)

- `outputs/submissions/EXP-004_submit.zip`
- `outputs/submissions/EXP-005_submit.zip`
- `outputs/submissions/EXP-006_submit.zip`
)
