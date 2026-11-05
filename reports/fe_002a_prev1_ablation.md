# FE-002A Prev1 Ablation

- generated_at: `2026-08-16T02:40:49`
- experiment_id: `FE-002A`
- baseline: `EXP-004` (existing artifacts reused; not overwritten)

## 1. Hypothesis

If `prev1 <= 0.35` fails because `asof_pitcher_prev1_game_success_rate` is a noisy/unstable signal, dropping **only** that column should improve EXP-004 temporal CV, especially 2023 and the prev1-low slice.

If dropping it **hurts**, prev1 itself is useful and the low-prev1 cohort is simply hard.

## 2. Why this experiment

ERR-AUDIT-001: `prev1_<=0.35` was the largest non-season failure (n=78,607, raw BSS −184). A failure **slice** is not the same as a bad **feature**. FE-002A is a single ablation to separate those. No shrinkage/smoothing is added.

## 3. What changed

- Removed exactly `asof_pitcher_prev1_game_success_rate`
- Feature count: 44 → 43
- baseline − FE-002A = `['asof_pitcher_prev1_game_success_rate']`
- FE-002A − baseline = `[]`

## 4. What stayed fixed

- CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss
- seed 42
- season weights 2019:1.0 … 2023:2.0
- expanding 2022/2023/2024 holdouts
- `src.metrics` unchanged
- fold-local median impute + native cats
- prev3/prev5, Trackman, other asof rates untouched
- no shrinkage feature

## 5. Baseline verification

EXP-004 was not retrained. Logged fold raw BSS match expected constants to 1e-8.

| Fold | n_valid | target | pred | raw BSS | official | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 247,472 | 0.528920 | 0.531923 | 2259.398328 | 2259.398328 | 0.243534 |
| 2023 | 245,525 | 0.499957 | 0.520392 | -1366.481067 | 0.000000 | 0.253416 |
| 2024 | 253,507 | 0.486105 | 0.496214 | 578.184416 | 578.184416 | 0.248363 |

## 6. Fold results

| Fold | n_train | n_valid | target | pred | raw BSS | official | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 728,588 | 247,472 | 0.528920 | 0.531751 | 2241.880343 | 2241.880343 | 0.243578 |
| 2023 | 976,060 | 245,525 | 0.499957 | 0.520166 | -1360.862125 | 0.000000 | 0.253402 |
| 2024 | 1,221,585 | 253,507 | 0.486105 | 0.497955 | 568.947266 | 568.947266 | 0.248386 |

- FE-002A mean (ddof=0): `483.321828` (EXP-004 `490.367225`)
- FE-002A std (ddof=0): `1472.059127` (EXP-004 `1481.560943`)
- FE-002A worst: `-1360.862125` (EXP-004 `-1366.481067`)

## 7. Delta vs EXP-004

| Fold | EXP-004 | FE-002A | Delta |
|---|---:|---:|---:|
| 2022 | 2259.398328 | 2241.880343 | -17.517984 |
| 2023 | -1366.481067 | -1360.862125 | +5.618942 |
| 2024 | 578.184416 | 568.947266 | -9.237149 |
| Mean | 490.367225 | 483.321828 | -7.045397 |
| Std | 1481.560943 | 1472.059127 | -9.501816 |
| Worst fold | -1366.481067 | -1360.862125 | +5.618942 |

## 8. Prev1 slice results

Bins match ERR-AUDIT-001 (`prev1_<=0.35` / mid / `>=0.65` / NA).

| slice | n | EXP-004 raw BSS | FE-002A raw BSS | Δ BSS | EXP-004 gap | FE-002A gap |
|---|---:|---:|---:|---:|---:|---:|
| prev1_<=0.35 | 78,607 | -184.2 | -276.0 | -91.8 | +0.0229 | +0.0272 |
| prev1_mid | 566,329 | 242.0 | 240.8 | -1.2 | +0.0101 | +0.0105 |
| prev1_>=0.65 | 92,267 | 2030.3 | 2049.2 | +18.9 | +0.0056 | +0.0031 |
| prev1_NA | 9,301 | -104.7 | -12.6 | +92.1 | +0.0304 | +0.0332 |
| not_prev1_<=0.35 | 667,897 | 622.9 | 625.8 | +2.8 | +0.0098 | +0.0098 |

## 9. Season × prev1 analysis

`prev1_<=0.35` inside each OOF season:

| slice | n | EXP-004 raw BSS | FE-002A raw BSS | Δ BSS | EXP-004 gap | FE-002A gap |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 21,628 | 1724.3 | 1717.6 | -6.7 | +0.0092 | +0.0108 |
| 2023 | 26,442 | -2434.0 | -2593.6 | -159.7 | +0.0409 | +0.0455 |
| 2024 | 30,537 | 248.2 | 154.5 | -93.6 | +0.0170 | +0.0229 |

## 10. Low-history interaction

Canonical `asof_pitcher_n` bins, then the same bins restricted to `prev1_<=0.35`.

### asof_pitcher_n

| slice | n | EXP-004 raw BSS | FE-002A raw BSS | Δ BSS | EXP-004 gap | FE-002A gap |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 232 | -2658.3 | -2777.4 | -119.0 | +0.0962 | +0.1011 |
| 1-100 | 21,442 | 349.5 | 459.4 | +110.0 | +0.0352 | +0.0367 |
| 101-500 | 68,670 | 815.6 | 809.7 | -5.9 | +0.0206 | +0.0225 |
| 501-2000 | 194,584 | 955.9 | 933.0 | -22.8 | +0.0128 | +0.0148 |
| 2000+ | 461,576 | 460.2 | 454.2 | -6.0 | +0.0079 | +0.0074 |

### asof_pitcher_n × prev1_<=0.35

| slice | n | EXP-004 raw BSS | FE-002A raw BSS | Δ BSS | EXP-004 gap | FE-002A gap |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | missing | missing | missing | missing | missing |
| 1-100 | 2,197 | -2538.9 | -2449.7 | +89.2 | +0.0546 | +0.0567 |
| 101-500 | 8,954 | -1634.6 | -1706.4 | -71.7 | +0.0346 | +0.0388 |
| 501-2000 | 24,655 | 354.7 | 178.6 | -176.1 | +0.0250 | +0.0306 |
| 2000+ | 42,801 | -99.7 | -156.4 | -56.8 | +0.0176 | +0.0212 |

### count 0-2 (auxiliary, from FE-001A)

| slice | n | EXP-004 raw BSS | FE-002A raw BSS | Δ BSS | EXP-004 gap | FE-002A gap |
|---|---:|---:|---:|---:|---:|---:|
| 0-2 | 46,276 | 78.2 | 116.3 | +38.2 | +0.0105 | +0.0106 |

## 11. Calibration / prediction mean

| Fold | target | EXP-004 pred | EXP-004 gap | FE-002A pred | FE-002A gap | Δ gap |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 0.528920 | 0.531923 | +0.003002 | 0.531751 | +0.002831 | -0.000172 |
| 2023 | 0.499957 | 0.520392 | +0.020435 | 0.520166 | +0.020209 | -0.000226 |
| 2024 | 0.486105 | 0.496214 | +0.010109 | 0.497955 | +0.011850 | +0.001741 |

## 12. Leakage and reproducibility

- EXP-004 and FE-001A fingerprints unchanged: **True**
- test.csv loaded only for schema checks; no test-row aggregates
- seed 42; same CatBoost params and season weights as EXP-004
- Public LB: **not submitted**

## 13. Interpretation

Fold-level deltas are small: 2023 +5.6, 2022 −17.5, 2024 −9.2, CV mean −7.0. That is **not** CASE A (noisy feature) and **not** CASE C (large opposing fold moves). 2023 overprediction barely moved (+0.0204 → +0.0202).

The experiment’s target slice moved the other way. `prev1_<=0.35` (n=78,607) raw BSS went −184.2 → −276.0 (Δ −91.8) and the calibration gap widened (+0.0229 → +0.0272). Season × prev1-low: 2023 −159.7, 2024 −93.6, 2022 −6.7. Dropping the column made the failure slice worse, especially in 2023.

Low-history overlap is weak as an explanation of that slice: most prev1-low rows are `501-2000` and `2000+`. Those cells got worse without prev1 (−176 and −57). The `1-100` bin improved overall, but it is a small share of prev1-low (n=2,197). So prev1-low failure is not mainly a cold-start pitcher problem, and the prev1 column is useful once history exists.

`prev1_NA` improved (+92) and count 0-2 improved (+38). Those side effects do not overturn the hypothesis test: the feature is not what *causes* the prev1-low miss; removing it hurts that cohort.

No shrinkage was added, so this is not mixed with a smoothing hypothesis.

## 14. Verdict

**REJECT**

Removing prev1 did not improve CV mean or the `prev1_<=0.35` slice. The 2023 fold change (+5.6) is not a material recovery. The low-prev1 cohort remains hard, and the raw prev1 feature looks useful rather than noisy.

Conclusion label: **B** (A=prev1 feature noisy; B=cohort hard / feature useful; C=cannot separate).

Single seed 42. Not a Public LB decision. Do **not** auto-start prev1 shrinkage from this run.

## 15. Next experiment

Stop. Do not auto-run prev1 shrinkage, prev3/prev5 ablation, strike_rate ablation, TM-002, or Public LB.

Single follow-up candidate after GPT review: **TM-002** (prior-season Trackman rates). FE-001A and FE-002A both left the 2023 fold essentially unfixed; TM-002 was already leakage-PASS in TM-AUDIT-001B and is a different mechanism than asof rate ablations.

- summary: `outputs/predictions/FE-002A/summary.json`
- OOF: `outputs/predictions/FE-002A/oof_holdout_predictions.csv`
