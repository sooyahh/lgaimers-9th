# Regime Shift Experiments (EXP-003 CatBoost / primary 2024)

- generated_at: `2026-08-07T01:08:39`
- common eval: valid=`2024` (primary fold)
- model params: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`; seed=`42`
- **no adopt/reject judgment — results only**

## Summary: raw / official BSS on 2024

| experiment | train seasons | n_train | raw BSS | official BSS | Brier |
|---|---|---:|---:|---:|---:|
| EXP-003 baseline | 2019–2023 | 1,221,585 | 562.278018 | 562.278018 | 0.248402 |
| A sample_weight | 2019–2023 (weighted) | 1,221,585 | 578.184416 | 578.184416 | 0.248363 |
| B recent-only | 2022–2023 | 492,997 | 540.828824 | 540.828824 | 0.248456 |
| C season feature | 2019–2023 (same as EXP-003) | 1,221,585 | 562.278018 | 562.278018 | 0.248402 |

| Δ A − EXP-003 (raw BSS) | +15.906398 |
| Δ B − EXP-003 (raw BSS) | -21.449194 |

## Experiment A — sample_weight

Weights: `2019=1.0, 2020=1.2, 2021=1.4, 2022=1.6, 2023=2.0`.

- train/valid: `[2019, 2020, 2021, 2022, 2023]` → `[2024]`
- n_train / n_valid: `1,221,585` / `253,507`
- train target mean: `0.531582`
- valid target rate: `0.486105`
- raw BSS: `578.184416`
- official BSS: `578.184416`
- Brier: `0.248363`
- elapsed_s: `199.0`

## Experiment B — drop 2019–2021

- train/valid: `[2022, 2023]` → `[2024]`
- n_train / n_valid: `492,997` / `253,507`
- train target mean: `0.514496`
- valid target rate: `0.486105`
- raw BSS: `540.828824`
- official BSS: `540.828824`
- Brier: `0.248456`
- elapsed_s: `86.7`

### Side-by-side with EXP-003 (full 2019–2023 train)

| | EXP-003 | B (2022–2023 train) |
|---|---:|---:|
| n_train | 1,221,585 | 492,997 |
| raw BSS | 562.278018 | 540.828824 |
| official BSS | 562.278018 | 540.828824 |
| Brier | 0.248402 | 0.248456 |

## Experiment C — season feature / SHAP

- `season` in the 44 input features? **`yes`**
- because it is already included: **no retrain**; metrics identical to EXP-003 primary_2024
- SHAP method: `CatBoost get_feature_importance(type='ShapValues')` on valid-2024 subsample n=`30,000` (of `253,507`)
- **`season` mean|SHAP| rank: `1` / `44`** (mean_abs_shap=`0.0789308`)

### Top-15 features by mean |SHAP|

| rank | feature | mean_abs_shap |
|---:|---|---:|
| 1 | `season` | 0.0789308 |
| 2 | `asof_pitcher_success_rate` | 0.039837 |
| 3 | `game_type` | 0.0350946 |
| 4 | `pitcher_id` | 0.0265592 |
| 5 | `asof_pitcher_ball_rate` | 0.0210794 |
| 6 | `asof_pitcher_reverse_rate` | 0.0166071 |
| 7 | `batter_team_id` | 0.0160014 |
| 8 | `asof_pitcher_prev1_game_success_rate` | 0.0155293 |
| 9 | `batter_id` | 0.0154608 |
| 10 | `asof_pitcher_prev3_game_success_rate` | 0.0144005 |
| 11 | `asof_batter_success_rate` | 0.0130057 |
| 12 | `pitcher_team_id` | 0.0128056 |
| 13 | `balls_before` | 0.0108221 |
| 14 | `strikes_before` | 0.00929837 |
| 15 | `batter_hand` | 0.00814751 |

- machine-readable: `outputs\eda\regime_shift_experiments.json`

## Follow-up: season extrapolation D / E / F

See `season_extrapolation_experiments.md` for full detail.

| experiment | raw BSS | official BSS | Δ raw vs EXP-003 |
|---|---:|---:|---:|
| EXP-003 baseline | 562.278018 | 562.278018 | +0.000000 |
| A sample_weight | 578.184416 | 578.184416 | +15.906398 |
| D drop season | -61.627806 | 0.000000 | -623.905824 |
| E season_since_start | 562.278018 | 562.278018 | +0.000000 |
| F no season + weights | 116.035290 | 116.035290 | -446.242727 |

## Follow-up: Experiment G (season as categorical; H skipped)

| experiment | raw BSS | official BSS | Δ raw vs EXP-003 |
|---|---:|---:|---:|
| EXP-003 baseline | 562.278018 | 562.278018 | +0.000000 |
| A sample_weight | 578.184416 | 578.184416 | +15.906398 |
| G season categorical | -357.891912 | 0.000000 | -920.169929 |
| H season clip | skipped | — | — |

Details: `season_categorical_experiment_G.md`
