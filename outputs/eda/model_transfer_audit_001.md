# MODEL-TRANSFER-AUDIT-001 — FE transferability across model families

- generated_at: `2026-08-19T00:41:17`
- feature set: `L42` (42 base features)
- backends: `['catboost', 'lightgbm']`
- seeds: `[42, 1, 7]`
- folds: `['secondary_2022', 'secondary_2023', 'primary_2024']`
- season sample weights: `{'2019': 1.0, '2020': 1.2, '2021': 1.4, '2022': 1.6, '2023': 2.0}`
- params per backend: `{"catboost": {"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}, "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31}}`
- fits executed / reused: `72` / `0`
- **results only — no adopt/reject; no submission; not written to `outputs/experiments.csv`**

## 1. Baseline model gap under this feature set

Answers: how much of the CatBoost-over-LightGBM lead survives once the ID-encoding confound is removed?

| fold | catboost | lightgbm | gap (cb − lgb) |
|---|---:|---:|---:|
| primary_2024 | 732.363 | 544.198 | +188.165 |
| secondary_2022 | 2321.307 | 2224.493 | +96.813 |
| secondary_2023 | -1308.953 | -1579.985 | +271.033 |

## 2. Transfer verdict per variant

Paired Δ = variant raw BSS − BASE raw BSS at the **same** backend, fold and seed.

| variant | change | Δ mean (catboost) | Δ mean (lightgbm) | sign agree | fold sign agree | verdict |
|---|---|---:|---:|---|---|---|
| FE-001A | drop asof_pitcher_ball_rate | -5.835 | -15.553 | yes | 2/3 | **INCONCLUSIVE_NOISE** |
| FE-002A | drop asof_pitcher_prev1_game_success_rate | -31.525 | -25.841 | yes | 3/3 | **PARTIAL_lightgbm** |
| FE-002 | row-local prev1 shrinkage toward long-term rate, k=500 | -2.694 | -10.545 | yes | 3/3 | **INCONCLUSIVE_NOISE** |

## 3. Noise reference and per-fold breakdown

### FE-001A — drop asof_pitcher_ball_rate

| backend | cells | Δ mean | Δ std (folds) | 2·sem | seed spread | materiality | Δ secondary_2022 | Δ secondary_2023 | Δ primary_2024 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| catboost | 9 | -5.835 | 13.946 | 16.104 | 11.434 | WITHIN_NOISE | -6.316 | -19.535 | +8.345 |
| lightgbm | 9 | -15.553 | 17.419 | 20.113 | 36.651 | WITHIN_NOISE | -9.405 | -35.212 | -2.042 |

- reading: No backend moved beyond its own paired-delta noise; this is a measurement result, not a feature result.

### FE-002A — drop asof_pitcher_prev1_game_success_rate

| backend | cells | Δ mean | Δ std (folds) | 2·sem | seed spread | materiality | Δ secondary_2022 | Δ secondary_2023 | Δ primary_2024 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| catboost | 9 | -31.525 | 36.853 | 42.554 | 16.130 | WITHIN_NOISE | -6.567 | -73.853 | -14.156 |
| lightgbm | 9 | -25.841 | 21.992 | 25.394 | 31.763 | MATERIAL | -2.221 | -45.726 | -29.575 |

- reading: Material in some backends only; treat as weak evidence and add seeds before deciding.

### FE-002 — row-local prev1 shrinkage toward long-term rate, k=500

| backend | cells | Δ mean | Δ std (folds) | 2·sem | seed spread | materiality | Δ secondary_2022 | Δ secondary_2023 | Δ primary_2024 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| catboost | 9 | -2.694 | 13.119 | 15.149 | 13.974 | WITHIN_NOISE | -13.770 | +11.795 | -6.106 |
| lightgbm | 9 | -10.545 | 13.748 | 15.875 | 29.088 | WITHIN_NOISE | -6.082 | +0.418 | -25.969 |

- reading: No backend moved beyond its own paired-delta noise; this is a measurement result, not a feature result.

## 4. Auxiliary — CatBoost vs LightGBM agreement on BASE

_single seed per fold; ensemble input only, not an adopt signal_

| fold | seed | Pearson corr | catboost raw BSS | lightgbm raw BSS | 50/50 blend raw BSS |
|---|---:|---:|---:|---:|---:|
| primary_2024 | 42 | 0.8579 | 739.970 | 482.022 | 666.280 |

## How to read this

- `TRANSFERS` — the FE moves both backends the same way beyond noise. The feature work is model-agnostic.
- `MODEL_SPECIFIC` — both backends move materially but in opposite directions. That is the CatBoost-specialisation risk.
- `PARTIAL_*` — material in one backend only. Weak; add seeds before acting.
- `INCONCLUSIVE_NOISE` — the delta is inside its own paired-delta noise. This says the **measurement** was not sensitive enough; it is not evidence the feature is bad.
- `materiality` is a descriptive heuristic (`|Δ mean| > 2·sem`, `sem = std(fold means)/√n_folds`), not a hypothesis test. Fold means are the unit of observation because seeds are nested inside folds.
- `seed spread (deterministic)` means that backend produced identical predictions across seeds under the configured params. LightGBM is deterministic here because `configs/baseline.yaml` sets no `subsample` / `colsample_bytree`; add them if you want a seed-noise estimate for it.
- Validation, metric and season weights are unchanged (`reports/validation_policy.md`, `src/metrics.py`).
