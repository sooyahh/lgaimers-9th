# TM-002A Expected Velocity

- generated_at: `2026-08-16T03:33:15`
- experiment_id: `TM-002A`
- baseline: `EXP-004` (existing artifacts reused; not overwritten)

## 1. Hypothesis

Main as-of pitch mix rates do not carry Trackman physical velocity scale. A fold-safe expected rel_speed from prior-season Trackman group means combined with the current row's as-of mix may add orthogonal information for temporal CV. This experiment does **not** require fixing 2023 to succeed.

## 2. What changed

- Added exactly `tm_expected_rel_speed`
- Feature count: 44 → 45
- TM-002A − EXP-004 = `['tm_expected_rel_speed']`
- EXP-004 − TM-002A = `[]`
- Trackman summary: `mean(rel_speed)` per core group; `other` excluded

## 3. What stayed fixed

- CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss
- seed 42
- season weights 2019:1.0 … 2023:2.0
- expanding 2022/2023/2024 holdouts
- `src.metrics` unchanged
- fold-local median impute + native cats
- no current-pitch Trackman, no player-ID join, no 2025 TM, no test aggregates
- no spin / IVB / HB / extension / cluster bundle

## 4. Formula

```
numerator = r_fastball*v_fastball + r_breaking*v_breaking + r_offspeed*v_offspeed
denominator = r_fastball + r_breaking + r_offspeed
tm_expected_rel_speed = numerator / denominator   if denominator > 0
tm_expected_rel_speed = NaN                        otherwise
```

Observed as-of mix rates already sum to 1 (float noise ~1e-6), so the denominator is a numerical guard, not a hidden `other` residual. Cold-start mix NA (`asof_pitcher_pitchmix_n==0`) stays NaN until fold-local median impute.

## 5. Baseline verification

EXP-004 was not retrained. Logged fold raw BSS match expected constants to 1e-8.

| Fold | n_valid | target | pred | raw BSS | official | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 247,472 | 0.528920 | 0.531923 | 2259.398328 | 2259.398328 | 0.243534 |
| 2023 | 245,525 | 0.499957 | 0.520392 | -1366.481067 | 0.000000 | 0.253416 |
| 2024 | 253,507 | 0.486105 | 0.496214 | 578.184416 | 578.184416 | 0.248363 |

## 6. Fold-specific Trackman profiles

| Valid | TM seasons used | fastball mean | breaking mean | offspeed mean |
|---|---|---:|---:|---:|
| 2022 | 2019,2020,2021 | 141.4812 | 126.6941 | 129.0195 |
| 2023 | 2019,2020,2021,2022 | 142.0660 | 127.0706 | 129.8509 |
| 2024 | 2019,2020,2021,2022,2023 | 142.4178 | 127.3264 | 130.2316 |

## 7. Fold results

| Fold | n_train | n_valid | target | pred | gap | raw BSS | official | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 728,588 | 247,472 | 0.528920 | 0.531095 | +0.002175 | 2261.564769 | 2261.564769 | 0.243529 |
| 2023 | 976,060 | 245,525 | 0.499957 | 0.520915 | +0.020958 | -1391.536084 | 0.000000 | 0.253479 |
| 2024 | 1,221,585 | 253,507 | 0.486105 | 0.496087 | +0.009982 | 596.056730 | 596.056730 | 0.248318 |

- TM-002A mean (ddof=0): `488.695138` (EXP-004 `490.367225`)
- TM-002A std (ddof=0): `1493.303127` (EXP-004 `1481.560943`)
- TM-002A worst: `-1391.536084` (EXP-004 `-1366.481067`)

## 8. Delta vs EXP-004

| Fold | EXP-004 | TM-002A | Delta |
|---|---:|---:|---:|
| 2022 | 2259.398328 | 2261.564769 | +2.166441 |
| 2023 | -1366.481067 | -1391.536084 | -25.055017 |
| 2024 | 578.184416 | 596.056730 | +17.872314 |
| Mean | 490.367225 | 488.695138 | -1.672087 |
| Std | 1481.560943 | 1493.303127 | +11.742184 |
| Worst fold | -1366.481067 | -1391.536084 | -25.055017 |

## 9. Calibration / 2023 gap

| Fold | target | EXP-004 pred | EXP-004 gap | TM-002A pred | TM-002A gap | Δ gap |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 0.528920 | 0.531923 | +0.003002 | 0.531095 | +0.002175 | -0.000828 |
| 2023 | 0.499957 | 0.520392 | +0.020435 | 0.520915 | +0.020958 | +0.000523 |
| 2024 | 0.486105 | 0.496214 | +0.010109 | 0.496087 | +0.009982 | -0.000127 |

- overall `tm_expected_rel_speed` missing rate (valid OOF rows, pre-impute): `0.000311`

## 10. Feature distribution (valid split, pre-impute)

| Fold | n | missing | mean | std | p10 | median | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 247,472 | 0.000356 | 135.2671 | 1.3245 | 133.6800 | 135.1456 | 136.9630 |
| 2023 | 245,525 | 0.000257 | 135.6901 | 1.2476 | 134.2469 | 135.5895 | 137.2293 |
| 2024 | 253,507 | 0.000320 | 136.0931 | 1.3296 | 134.5075 | 136.0134 | 137.7246 |

## 11. Correlation with existing mix rates (valid rows)

| Fold | fastball | breaking | offspeed |
|---|---:|---:|---:|
| 2022 | +0.9867 | -0.5641 | -0.3530 |
| 2023 | +0.9777 | -0.6119 | -0.2268 |
| 2024 | +0.9786 | -0.6569 | -0.2069 |

High correlation is diagnostic only; it is not an automatic reject.

## 12. Core error slices

### season

| slice | n | EXP-004 raw BSS | TM-002A raw BSS | Δ BSS | EXP-004 gap | TM-002A gap |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 247,472 | 2259.4 | 2261.6 | +2.2 | +0.0030 | +0.0022 |
| 2023 | 245,525 | -1366.5 | -1391.5 | -25.1 | +0.0204 | +0.0210 |
| 2024 | 253,507 | 578.2 | 596.1 | +17.9 | +0.0101 | +0.0100 |

### prev1_<=0.35

| slice | n | EXP-004 raw BSS | TM-002A raw BSS | Δ BSS | EXP-004 gap | TM-002A gap |
|---|---:|---:|---:|---:|---:|---:|
| prev1_<=0.35 | 78,607 | -184.2 | -197.0 | -12.8 | +0.0229 | +0.0232 |

### asof_pitcher_n

| slice | n | EXP-004 raw BSS | TM-002A raw BSS | Δ BSS | EXP-004 gap | TM-002A gap |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 232 | -2658.3 | -3596.0 | -937.7 | +0.0962 | +0.1022 |
| 1-100 | 21,442 | 349.5 | 340.7 | -8.7 | +0.0352 | +0.0353 |
| 101-500 | 68,670 | 815.6 | 826.6 | +10.9 | +0.0206 | +0.0202 |
| 501-2000 | 194,584 | 955.9 | 926.4 | -29.5 | +0.0128 | +0.0133 |
| 2000+ | 461,576 | 460.2 | 469.5 | +9.3 | +0.0079 | +0.0075 |

## 13. Leakage and reproducibility

- EXP-004 / FE-001A / FE-002A fingerprints unchanged: **True**
- Trackman temporal safety assertions passed: **True**
- test.csv loaded only for schema checks; no test-row aggregates
- no current-pitch Trackman join; TM columns used: season, pitch_type_group, rel_speed
- seed 42; same CatBoost params and season weights as EXP-004
- Public LB: **not submitted**

## 14. What this experiment tested / did not test

Tested: one global historical pitch-group rel_speed profile × as-of mix feature.

Not tested: spin, movement, extension, release location, clustering, archetypes, hand mapping, player-level TM, current-pitch physics, tuning, multi-seed, ensemble.

A negative result does **not** mean Trackman is useless. A positive result does **not** authorize adding a TM bundle.

## 15. Verdict

**REJECT**

CV mean raw BSS worsened versus EXP-004.

Single seed 42. Not a Public LB decision.

## 16. Next experiment

Stop. Do not auto-run expected spin/movement/extension, TM bundle, clustering, hand mapping, tuning, multi-seed, ensemble, or Public LB.

- summary: `outputs/predictions/TM-002A/summary.json`
- OOF: `outputs/predictions/TM-002A/oof_holdout_predictions.csv`
