# Season Validation Audit (VAL-AUDIT-001)

- generated_at: `2026-08-06T19:10:17`
- model: training-split **target mean** only (EXP-000 logic reused)
- no Logistic Regression / GBDT / Trackman / feature engineering

## Per-season target rate (full train.csv)

| season | n_rows | target_mean |
|---|---:|---:|
| 2019 | 237413 | 0.564670 |
| 2020 | 244087 | 0.532712 |
| 2021 | 247088 | 0.532762 |
| 2022 | 247472 | 0.528920 |
| 2023 | 245525 | 0.499957 |
| 2024 | 253507 | 0.486105 |

## Expanding-window target-mean folds

| fold | train seasons | valid | n_train | n_valid | train mean | valid mean | Δ(valid−train) | pred mean | Brier | raw BSS | official BSS | elapsed_s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fold_2022 | 2019-2020-2021 | 2022 | 728588 | 247472 | 0.543142 | 0.528920 | -0.014222 | 0.543142 | 0.249366 | -81.177 | 0.000 | 0.462 |
| fold_2023 | 2019-2020-2021-2022 | 2023 | 976060 | 245525 | 0.539537 | 0.499957 | -0.039579 | 0.539537 | 0.251567 | -626.607 | 0.000 | 0.565 |
| fold_2024 | 2019-2020-2021-2022-2023 | 2024 | 1221585 | 253507 | 0.531582 | 0.486105 | -0.045477 | 0.531582 | 0.251875 | -827.887 | 0.000 | 0.730 |

## Aggregate

- mean raw BSS across folds: `-511.890527`
- std raw BSS across folds: `315.451259`
- mean official BSS across folds: `0.000000`
- folds with official BSS == 0: `3` / `3`

## Interpretation (evidence-separated)

### 1) Year-to-year target drift (observed)

Per-season `control_success` means are not constant. Observed season means: 2019=0.5647, 2020=0.5327, 2021=0.5328, 2022=0.5289, 2023=0.5000, 2024=0.4861. This supports a **seasonal target distribution shift** hypothesis for constant-mean predictors.

### 2) Specific-season anomaly vs repeated pattern

- Fold mean gaps (valid−train): fold_2022=-0.0142, fold_2023=-0.0396, fold_2024=-0.0455.
- Across all three expanding folds, valid mean was **below** train mean and raw BSS was **negative**. This is a **repeated pattern** under the target-mean baseline, not a 2024-only artifact.
- Largest |valid−train| gap among these folds: `fold_2024` (`-0.045477`).

### 3) Fold-wise raw BSS stability

- raw BSS values: fold_2022=-81.177, fold_2023=-626.607, fold_2024=-827.887; mean=`-511.891`, std=`315.451`.
- Official BSS is clipped with `max(0, ·)`, so when raw BSS is negative on every fold, official BSS collapses to 0 and **cannot rank** constant-mean folds. Diagnostics must use raw BSS / Brier / mean gap.

### 4) Risk of using 2024 holdout alone as primary

- EXP-000 already showed 2019–2023 mean → 2024 valid yields official BSS = 0 with negative raw BSS.
- This audit shows earlier expanding folds also produce negative raw BSS under the same constant baseline, so 2024 is consistent with a broader seasonal-mean mismatch problem.
- **Risk:** selecting models only on 2024 holdout can overfit to one season’s base rate. A single holdout cannot distinguish true skill from luck in matching that season’s rate.

### 5) Need for expanding-window average as an auxiliary criterion

- Recommendation (non-final validation policy): keep 2024 season holdout as a **primary stress test**, and report **expanding-window mean ± std of raw BSS / Brier / mean-gap** as auxiliary diagnostics.
- Do not average official BSS alone while it is floor-clipped at 0 for weak baselines.
- Validation strategy is still **not finalized**; game-id absence and empty `order_cols` remain unchanged.

## Artifacts

- `outputs\predictions\VAL-AUDIT-001\fold_metrics.csv`
- `outputs\predictions\VAL-AUDIT-001\summary.json`

