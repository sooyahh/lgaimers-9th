# FE-001A Ball Rate Ablation

- generated_at: `2026-08-16T02:21:08`
- experiment_id: `FE-001A`
- baseline: `EXP-004` (existing artifacts reused; not retrained, not overwritten)

## 1. Hypothesis

FEATURE-STABILITY-AUDIT labeled `asof_pitcher_ball_rate` as **SIGN_FLIP**. If EXP-004’s 2023 collapse is partly caused by trusting that flipped association, removing only this column should improve 2023 (and/or worst-fold) raw BSS without large 2022/2024 damage.

## 2. What changed

- Removed exactly one feature: `asof_pitcher_ball_rate`
- Feature count: 44 → 43 (delta -1)
- Assert baseline − FE-001A = `['asof_pitcher_ball_rate']`
- Assert FE-001A − baseline = `[]`

## 3. What stayed fixed

- Model: CatBoost native cats, `iterations=300`, `learning_rate=0.05`, `depth=6`, `Logloss`
- Seed: 42
- Season weights: `{2019:1.0, 2020:1.2, 2021:1.4, 2022:1.6, 2023:2.0}`
- Validation: expanding 2019–21→2022, 2019–22→2023, 2019–23→2024
- Metric: `src.metrics.compute_brier_metrics` / official BSS clip-at-zero
- Preprocessing: fold-local numeric median impute + native cat strings (`tree_exp_common`)
- No tuning, no ensemble, no Trackman, no prev* edits, no test.csv aggregates

## 4. Baseline reproduction

EXP-004 was **not re-run** (would overwrite protected artifacts). Logged fold raw BSS were checked against `outputs/predictions/EXP-004/summary.json` and match the expected constants to 1e-8.

| Fold | n_valid | target mean | pred mean | raw BSS | official BSS | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 247,472 | 0.528920 | 0.531923 | 2259.398328 | 2259.398328 | 0.243534 |
| 2023 | 245,525 | 0.499957 | 0.520392 | -1366.481067 | 0.000000 | 0.253416 |
| 2024 | 253,507 | 0.486105 | 0.496214 | 578.184416 | 578.184416 | 0.248363 |

- EXP-004 fold raw BSS mean (ddof=0): `490.367225`
- EXP-004 fold raw BSS std (ddof=0): `1481.560943`
- EXP-004 worst fold: `-1366.481067` (2023)

## 5. Fold results (FE-001A)

| Fold | n_train | n_valid | target mean | pred mean | raw BSS | official BSS | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 728,588 | 247,472 | 0.528920 | 0.530113 | 2222.012531 | 2222.012531 | 0.243627 |
| 2023 | 976,060 | 245,525 | 0.499957 | 0.520528 | -1393.668603 | 0.000000 | 0.253484 |
| 2024 | 1,221,585 | 253,507 | 0.486105 | 0.496331 | 582.122112 | 582.122112 | 0.248353 |

- FE-001A fold raw BSS mean (ddof=0): `470.155347`
- FE-001A fold raw BSS std (ddof=0): `1478.217380`
- FE-001A worst fold: `-1393.668603`

## 6. Delta vs EXP-004

| Fold | EXP-004 raw BSS | FE-001A raw BSS | Delta |
|---|---:|---:|---:|
| 2022 | 2259.398328 | 2222.012531 | -37.385797 |
| 2023 | -1366.481067 | -1393.668603 | -27.187536 |
| 2024 | 578.184416 | 582.122112 | +3.937697 |
| Mean | 490.367225 | 470.155347 | -20.211879 |
| Std | 1481.560943 | 1478.217380 | -3.343563 |
| Worst fold | -1366.481067 | -1393.668603 | -27.187536 |

Prediction mean vs target mean:

| Fold | target | EXP-004 pred | EXP-004 gap | FE-001A pred | FE-001A gap | Δ gap |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 0.528920 | 0.531923 | +0.003002 | 0.530113 | +0.001193 | -0.001810 |
| 2023 | 0.499957 | 0.520392 | +0.020435 | 0.520528 | +0.020571 | +0.000136 |
| 2024 | 0.486105 | 0.496214 | +0.010109 | 0.496331 | +0.010226 | +0.000117 |

## 7. Error slice comparison

Minimal slices from ERR-AUDIT-001. Concatenated OOF (2022–2024 valid rows).

| slice | level | n | EXP-004 raw BSS | FE-001A raw BSS | Δ BSS | EXP-004 gap | FE-001A gap |
|---|---|---:|---:|---:|---:|---:|---:|
| season | 2022 | 247,472 | 2259.4 | 2222.0 | -37.4 | +0.0030 | +0.0012 |
| season | 2023 | 245,525 | -1366.5 | -1393.7 | -27.2 | +0.0204 | +0.0206 |
| season | 2024 | 253,507 | 578.2 | 582.1 | +3.9 | +0.0101 | +0.0102 |
| asof_pitcher_n | 0 | 232 | -2658.3 | -3056.2 | -397.9 | +0.0962 | +0.0992 |
| asof_pitcher_n | 1-100 | 21,442 | 349.5 | 346.8 | -2.7 | +0.0352 | +0.0358 |
| asof_pitcher_n | 101-500 | 68,670 | 815.6 | 798.8 | -16.9 | +0.0206 | +0.0198 |
| asof_pitcher_n | 2000+ | 461,576 | 460.2 | 451.5 | -8.7 | +0.0079 | +0.0073 |
| asof_pitcher_n | 501-2000 | 194,584 | 955.9 | 906.6 | -49.2 | +0.0128 | +0.0123 |
| prev1 | prev1_<=0.35 | 78,607 | -184.2 | -228.7 | -44.5 | +0.0229 | +0.0234 |
| count | 0-2 | 46,276 | 78.2 | 100.9 | +22.7 | +0.0105 | +0.0100 |

## 8. Leakage/reproducibility check

- EXP-004 artifact fingerprints unchanged: **True**
- test.csv was loaded by `load_competition_data` for schema checks only; not used in features/aggregates
- Fold train seasons are strictly earlier than valid seasons (`build_season_holdout`)
- Seed 42; CatBoost `random_seed` from `build_model`
- Public LB: **not submitted / pending**

## 9. Interpretation

Removing `asof_pitcher_ball_rate` did **not** help temporal generalization.

- **2023 got worse** (−27.2 raw BSS). Overprediction is essentially unchanged (gap +0.0204 → +0.0206). The SIGN_FLIP column was not the main driver of the 2023 collapse.
- **2022 paid a larger tax** (−37.4). That fold is where ball_rate’s 2019–2022 association was usable; dropping it hurts the “good” year more than it helps the bad year.
- **2024 is a wash** (+3.9), far smaller than 2022/2023 losses.
- CV mean declined (−20.2). Std shrank trivially (−3.3) only because the 2022 peak fell; worst fold also declined (−27.2).
- Failure slices did not recover: `prev1_<=0.35` −44.5 BSS; low `asof_pitcher_n` bins mixed-to-worse. Count 0-2 improved slightly (+22.7) but that does not offset season-level losses.

So: a flipped *marginal* correlation is not the same as a feature the tree was relying on for 2023. Do not chain another single asof-rate drop (`strike_rate`) without a new hypothesis.

## 10. ADOPT / HOLD / REJECT

**REJECT**

2023 raw BSS and worst fold both deteriorated; 2022 loss is larger than the 2024 bump; 2023 calibration gap did not shrink. Single seed 42; not a Public LB decision.

## 11. Next experiment recommendation

Do **not** auto-run `asof_pitcher_strike_rate` ablation. Stop for GPT review.

Recommended next single experiment (unchanged from the audit decision, now with extra negative evidence against asof-rate ablation): **FE-002** row-local prev1 shrinkage toward `asof_pitcher_success_rate`. `prev1_<=0.35` remains the largest actionable non-season failure and got slightly worse here.

- FE-001A summary: `outputs/predictions/FE-001A/summary.json`
- FE-001A OOF: `outputs/predictions/FE-001A/oof_holdout_predictions.csv`
