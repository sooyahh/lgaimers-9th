# ERR-AUDIT-001

- baseline model: **EXP-004** CatBoost + season sample weights
- OOF: `outputs/predictions/EXP-004/oof_holdout_predictions.csv`
- validation: expanding secondary 2022 / 2023 + primary 2024 (`is_comparable_cv=true`)
- metric: `src.metrics.compute_brier_metrics` (unchanged); quantile-10 ECE copied from VAL-EXP006-SECONDARY-001 for diagnostics only
- no retraining; existing EXP artifacts were not overwritten

## Why EXP-004, not EXP-006

EXP-004 is the latest CatBoost with is_comparable_cv=true, expanding 3-fold OOF covering 2022/2023/2024, and passed gpt_audit join verification. EXP-006 is hold (not adopt); its 2024 file is the Optuna tune holdout; EXP-005 has no row-level OOF.

- EXP-006 vs EXP-005 secondary verdict = **hold** (`VAL-EXP006-SECONDARY-001`).
- EXP-006 2024 predictions are the Optuna tune holdout, so 2024 error slices would be optimistically biased.
- EXP-005 has no row-level OOF.
- VAL-EXP006 parquet exists locally for 2022/2023 only; it is not a comparable 3-fold OOF and was not used as the primary error table.

## 2-1. OOF integrity

| check | value |
|---|---|
| OOF rows | 746,504 |
| row_id unique | True |
| duplicate row_id count | 0 |
| rows before join | 746504 |
| rows after join | 746504 |
| unmatched (pitcher_id NA) | 0 |
| duplicate increase vs source | 0 |
| prediction NaN | 0 |
| prediction inf | 0 |
| prediction min | 0.328365 |
| prediction max | 0.799734 |
| prediction outside [0,1] | 0 |
| target alignment vs train | True |
| folds | primary_2024, secondary_2022, secondary_2023 |
| seasons | 2022, 2023, 2024 |

Fold raw BSS recomputed with `evaluate_predictions` vs logged EXP-004 summary:

| fold | n | recomputed raw BSS | logged raw BSS | match |
|---|---|---|---|---|
| primary_2024 | 253,507 | 578.184416 | 578.184416 | True |
| secondary_2022 | 247,472 | 2259.398328 | 2259.398328 | True |
| secondary_2023 | 245,525 | -1366.481067 | -1366.481067 | True |

Integrity: **PASS**. OOF is suitable for comparable-CV error analysis.

## 2-2 / 2-3. Slice metrics

Raw BSS is the ranking diagnostic when official BSS is clipped at 0. Groups with n < 2,000 are flagged.

### Season

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | 2023 | 245,525 | 0.329 | 0.5000 | 0.5204 | 0.0204 | -1366.5 | 0.0 | 0.25342 | 0.0272 | 0.4987 |  |
| all_oof | 2024 | 253,507 | 0.340 | 0.4861 | 0.4962 | 0.0101 | 578.2 | 578.2 | 0.24836 | 0.0101 | 0.4968 |  |
| all_oof | 2022 | 247,472 | 0.332 | 0.5289 | 0.5319 | 0.0030 | 2259.4 | 2259.4 | 0.24353 | 0.0072 | 0.4878 |  |

### Pitcher sample size (`asof_pitcher_n`)

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | 0 | 232 | 0.000 | 0.4655 | 0.5617 | 0.0962 | -2658.3 | 0.0 | 0.25543 | 0.1026 | 0.4946 | small_n_interpret_with_caution |
| all_oof | 1-100 | 21,442 | 0.029 | 0.4969 | 0.5321 | 0.0352 | 349.5 | 349.5 | 0.24912 | 0.0366 | 0.4907 |  |
| all_oof | 2000+ | 461,576 | 0.618 | 0.5067 | 0.5146 | 0.0079 | 460.2 | 460.2 | 0.24880 | 0.0081 | 0.4960 |  |
| all_oof | 101-500 | 68,670 | 0.092 | 0.5049 | 0.5255 | 0.0206 | 815.6 | 815.6 | 0.24794 | 0.0212 | 0.4909 |  |
| all_oof | 501-2000 | 194,584 | 0.261 | 0.5013 | 0.5142 | 0.0128 | 955.9 | 955.9 | 0.24761 | 0.0128 | 0.4923 |  |

2023 only:

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| secondary_2023 | 0 | 63 | 0.000 | 0.4286 | 0.5851 | 0.1565 | -11645.8 | 0.0 | 0.27342 | 0.1845 | 0.5087 | small_n_interpret_with_caution |
| secondary_2023 | 1-100 | 5,953 | 0.024 | 0.4640 | 0.5564 | 0.0924 | -6891.6 | 0.0 | 0.26584 | 0.1023 | 0.5042 |  |
| secondary_2023 | 101-500 | 19,911 | 0.081 | 0.4773 | 0.5376 | 0.0602 | -5067.9 | 0.0 | 0.26213 | 0.0659 | 0.5025 |  |
| secondary_2023 | 501-2000 | 60,116 | 0.245 | 0.4847 | 0.5162 | 0.0315 | -2228.1 | 0.0 | 0.25533 | 0.0345 | 0.4995 |  |
| secondary_2023 | 2000+ | 159,482 | 0.650 | 0.5099 | 0.5185 | 0.0085 | -493.7 | 0.0 | 0.25114 | 0.0207 | 0.4977 |  |

### Batter sample size (`asof_batter_n`)

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | 2000+ | 506,426 | 0.678 | 0.4995 | 0.5054 | 0.0059 | 561.4 | 561.4 | 0.24860 | 0.0059 | 0.4964 |  |
| all_oof | 501-2000 | 161,542 | 0.216 | 0.5139 | 0.5296 | 0.0157 | 627.1 | 627.1 | 0.24824 | 0.0157 | 0.4921 |  |
| all_oof | 101-500 | 58,002 | 0.078 | 0.5192 | 0.5502 | 0.0310 | 647.0 | 647.0 | 0.24802 | 0.0310 | 0.4879 |  |
| all_oof | 1-100 | 20,287 | 0.027 | 0.5249 | 0.5738 | 0.0488 | 1032.9 | 1032.9 | 0.24680 | 0.0492 | 0.4819 |  |
| all_oof | 0 | 247 | 0.000 | 0.5587 | 0.5769 | 0.0182 | 1108.0 | 1108.0 | 0.24382 | 0.0847 | 0.4775 | small_n_interpret_with_caution |

### Count

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | 0-2 | 46,276 | 0.062 | 0.5052 | 0.5158 | 0.0105 | 78.2 | 78.2 | 0.24978 | 0.0199 | 0.4963 |  |
| all_oof | 1-2 | 71,595 | 0.096 | 0.5103 | 0.5150 | 0.0047 | 280.4 | 280.4 | 0.24919 | 0.0137 | 0.4958 |  |
| all_oof | 3-0 | 8,881 | 0.012 | 0.4915 | 0.4984 | 0.0069 | 330.3 | 330.3 | 0.24910 | 0.0203 | 0.4945 |  |
| all_oof | 0-1 | 92,890 | 0.124 | 0.5178 | 0.5203 | 0.0025 | 572.2 | 572.2 | 0.24826 | 0.0122 | 0.4942 |  |
| all_oof | 0-0 | 192,539 | 0.258 | 0.5068 | 0.5196 | 0.0128 | 589.8 | 589.8 | 0.24848 | 0.0128 | 0.4943 |  |
| all_oof | 2-2 | 61,260 | 0.082 | 0.5048 | 0.5126 | 0.0078 | 633.2 | 633.2 | 0.24839 | 0.0086 | 0.4948 |  |
| all_oof | 1-0 | 77,422 | 0.104 | 0.5012 | 0.5180 | 0.0168 | 640.5 | 640.5 | 0.24840 | 0.0168 | 0.4942 |  |
| all_oof | 3-2 | 35,473 | 0.048 | 0.4748 | 0.4965 | 0.0217 | 648.5 | 648.5 | 0.24775 | 0.0217 | 0.4939 |  |
| all_oof | 1-1 | 75,831 | 0.102 | 0.5091 | 0.5192 | 0.0100 | 657.0 | 657.0 | 0.24827 | 0.0112 | 0.4942 |  |
| all_oof | 2-0 | 26,866 | 0.036 | 0.4967 | 0.5144 | 0.0178 | 803.3 | 803.3 | 0.24798 | 0.0178 | 0.4936 |  |
| all_oof | 2-1 | 40,039 | 0.054 | 0.4992 | 0.5157 | 0.0164 | 1018.5 | 1018.5 | 0.24745 | 0.0168 | 0.4932 |  |
| all_oof | 3-1 | 17,432 | 0.023 | 0.4824 | 0.4989 | 0.0166 | 1163.6 | 1163.6 | 0.24678 | 0.0194 | 0.4925 |  |

### LI

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | 3+ | 26,548 | 0.036 | 0.5017 | 0.5167 | 0.0150 | 471.2 | 471.2 | 0.24882 | 0.0159 | 0.4949 |  |
| all_oof | 1-2 | 198,455 | 0.266 | 0.5112 | 0.5194 | 0.0083 | 583.1 | 583.1 | 0.24842 | 0.0090 | 0.4943 |  |
| all_oof | <=0.5 | 242,202 | 0.324 | 0.4978 | 0.5106 | 0.0128 | 616.4 | 616.4 | 0.24845 | 0.0128 | 0.4945 |  |
| all_oof | 0.5-1 | 228,494 | 0.306 | 0.5071 | 0.5182 | 0.0111 | 634.1 | 634.1 | 0.24837 | 0.0113 | 0.4944 |  |
| all_oof | 2-3 | 50,805 | 0.068 | 0.5054 | 0.5179 | 0.0124 | 643.1 | 643.1 | 0.24836 | 0.0138 | 0.4943 |  |

### Inning

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | 10+ | 12,153 | 0.016 | 0.4934 | 0.5023 | 0.0089 | 97.3 | 97.3 | 0.24971 | 0.0175 | 0.4977 |  |
| all_oof | 1-3 | 250,830 | 0.336 | 0.5115 | 0.5205 | 0.0091 | 574.9 | 574.9 | 0.24843 | 0.0096 | 0.4944 |  |
| all_oof | 4-6 | 250,128 | 0.335 | 0.5049 | 0.5172 | 0.0122 | 606.5 | 606.5 | 0.24846 | 0.0123 | 0.4943 |  |
| all_oof | 7-9 | 233,393 | 0.313 | 0.4983 | 0.5106 | 0.0123 | 674.7 | 674.7 | 0.24831 | 0.0124 | 0.4944 |  |

### Base / score / hands / recent history

num_runners_on:

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | 1 | 225,615 | 0.302 | 0.5094 | 0.5170 | 0.0075 | 577.4 | 577.4 | 0.24847 | 0.0092 | 0.4944 |  |
| all_oof | 0 | 392,175 | 0.525 | 0.5030 | 0.5151 | 0.0121 | 608.3 | 608.3 | 0.24847 | 0.0121 | 0.4946 |  |
| all_oof | 2 | 104,534 | 0.140 | 0.5039 | 0.5172 | 0.0133 | 668.2 | 668.2 | 0.24831 | 0.0136 | 0.4941 |  |
| all_oof | 3 | 24,180 | 0.032 | 0.4967 | 0.5171 | 0.0204 | 903.4 | 903.4 | 0.24773 | 0.0235 | 0.4930 |  |

score_diff_pitcher_team bins:

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | -2to0 | 334,985 | 0.449 | 0.5090 | 0.5191 | 0.0101 | 569.6 | 569.6 | 0.24850 | 0.0101 | 0.4945 |  |
| all_oof | >=4 | 92,670 | 0.124 | 0.5004 | 0.5134 | 0.0131 | 607.0 | 607.0 | 0.24848 | 0.0134 | 0.4943 |  |
| all_oof | 1to3 | 189,832 | 0.254 | 0.5065 | 0.5177 | 0.0112 | 637.1 | 637.1 | 0.24837 | 0.0115 | 0.4943 |  |
| all_oof | <=-3 | 129,017 | 0.173 | 0.4949 | 0.5073 | 0.0124 | 676.4 | 676.4 | 0.24828 | 0.0138 | 0.4944 |  |

pitcher_hand × batter_hand:

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | 1x1 | 92,907 | 0.124 | 0.4711 | 0.4953 | 0.0242 | 345.7 | 345.7 | 0.24830 | 0.0242 | 0.4941 |  |
| all_oof | 2x2 | 273,195 | 0.366 | 0.5003 | 0.5181 | 0.0177 | 437.9 | 437.9 | 0.24891 | 0.0177 | 0.4948 |  |
| all_oof | 1x2 | 110,252 | 0.148 | 0.5155 | 0.5185 | 0.0030 | 628.2 | 628.2 | 0.24819 | 0.0141 | 0.4940 |  |
| all_oof | 2x1 | 270,150 | 0.362 | 0.5167 | 0.5200 | 0.0033 | 659.6 | 659.6 | 0.24807 | 0.0082 | 0.4943 |  |

prev1 missing vs observed:

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | prev1_NA | 9,301 | 0.012 | 0.4996 | 0.5300 | 0.0304 | -104.7 | 0.0 | 0.25026 | 0.0327 | 0.4938 |  |
| all_oof | prev1_observed | 737,203 | 0.988 | 0.5049 | 0.5158 | 0.0109 | 630.0 | 630.0 | 0.24840 | 0.0109 | 0.4944 |  |

prev1 extreme bins:

| scope | level | n | share | y_mean | p_mean | gap | raw BSS | official BSS | Brier | ECE10q | MAE | warn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_oof | prev1_<=0.35 | 78,607 | 0.105 | 0.4628 | 0.4857 | 0.0229 | -184.2 | 0.0 | 0.24908 | 0.0229 | 0.4954 |  |
| all_oof | prev1_NA | 9,301 | 0.012 | 0.4996 | 0.5300 | 0.0304 | -104.7 | 0.0 | 0.25026 | 0.0327 | 0.4938 |  |
| all_oof | prev1_mid | 566,329 | 0.759 | 0.5023 | 0.5124 | 0.0101 | 242.0 | 242.0 | 0.24939 | 0.0102 | 0.4965 |  |
| all_oof | prev1_>=0.65 | 92,267 | 0.124 | 0.5569 | 0.5625 | 0.0056 | 2030.3 | 2030.3 | 0.24176 | 0.0101 | 0.4809 |  |

## 2-4. Failure patterns

### FAILURE-01 — 2023 OOF is globally worse than climatology, not a tiny subgroup

- 현상: 2023 OOF is globally worse than climatology, not a tiny subgroup
- 근거: 2023 raw BSS=-1366.5 (official 0); 2022=2259.4; 2024=578.2. This reproduces the logged EXP-004 secondary_2023 failure.
- 영향받는 행 비율: 0.329 of concatenated OOF (the entire 2023 fold)
- 가능한 원인: seasonal target-rate shift plus feature–target sign changes (ball/strike asof rates) documented in FEATURE-STABILITY-AUDIT / target_regime_check
- feature 가설: prefer features whose 2019–2024 association does not flip; avoid leaning on SIGN_FLIP asof rates
- validation 문제 가능성: validation itself is the intended stress test; do not retune validation to hide 2023
- 추가 검증 필요 여부: yes — confirm whether 2023 failure is uniform across count/LI/cold-start (tables above)

### FAILURE-02 — Pitcher cold-start / very low asof_n

- 현상: Pitcher cold-start / very low asof_n
- 근거: asof_pitcher_n==0 raw BSS=-2658.3 n=232; 2000+ raw BSS=460.2. pred std at n==0=0.0838, frac extreme=0.0474.
- 영향받는 행 비율: 0.0003 of OOF
- 가능한 원인: asof rates missing when n=0; model must fall back to global/context features
- feature 가설: explicit cold-start flag or stronger shrinkage toward league/count prior when asof_pitcher_n is tiny
- validation 문제 가능성: low — this is a feature-support issue, not a split bug
- 추가 검증 필요 여부: yes if the n==0 slice is large enough in 2025; sample share is reported above

### FAILURE-03 — Batter cold-start is **not** a primary failure

- 현상: asof_batter_n==0 is tiny and **better** than the OOF average (raw BSS +1108, n=247)
- 근거: batter n==0 share=0.0003; bins 1-100 through 2000+ are all positive raw BSS on concatenated OOF
- 영향받는 행 비율: 0.0003
- 가능한 원인: not applicable as a failure; batter asof is a weak/unstable associate in FEATURE-STABILITY-AUDIT but the model does not collapse on new batters
- feature 가설: do **not** prioritize batter cold-start FE
- validation 문제 가능성: none observed
- 추가 검증 필요 여부: no, unless 2025 has a much larger new-batter mass (not measurable here)

### FAILURE-04 — Count-level miscalibration

- 현상: Count-level miscalibration
- 근거: See count table: compare prediction mean vs target mean especially 3-0 / 3-2 / 0-2. Systematic gap (not just noise) is the failure.
- 영향받는 행 비율: count is defined on 100% of rows; the badly calibrated cells are the ones with large |gap| in the table
- 가능한 원인: count is a stable context feature (stability audit) but the model may under-use it relative to drifting asof rates
- feature 가설: count × hand interaction or count-specific calibration; keep it fold-safe (row-local, no test aggregates)
- validation 문제 가능성: low
- 추가 검증 필요 여부: yes — rank counts by |calibration_gap| and raw BSS

### FAILURE-05 — High LI / innings 7-9 are **not** the failure; extra innings are weak

- 현상: LI 3+ raw BSS=471 is only mildly below concat OOF (~621). Innings 7-9 (674.7) are **better** than 1-3 (574.9). Extra innings (10+, n=12,153, share=0.016) drop to raw BSS 97.3.
- 근거: slice tables above
- 영향받는 행 비율: extra innings 1.6%; LI 3+ 3.6%
- 가능한 원인: extra-inning samples are small and possibly a different game-type mix; not a general “late game” effect
- feature 가설: do **not** add `li`/`WE`. Extra-inning FE is low priority (small n)
- validation 문제 가능성: none for 7-9; extra innings are a small-slice warning
- 추가 검증 필요 여부: no for LI/7-9; extra innings only if 2025 extras are common (unknown)

### FAILURE-06 — Prev-game asof missing / extreme

- 현상: Prev-game asof missing / extreme
- 근거: prev1_NA raw BSS=-104.7 vs observed=630.0.
- 영향받는 행 비율: prev1_NA share=0.012
- 가능한 원인: prev* features are sparse at pitcher-season start; extremes of prev1 success may be small-sample noise
- feature 가설: missing indicator + shrink prev rates; do not treat prev1 as a current-pitch signal
- validation 문제 가능성: low (asof construction already audited as no mid-season NA re-entry)
- 추가 검증 필요 여부: connect to FEATURE-STABILITY-AUDIT SPARSE/COLD_START labels

## Answers to the required questions

1. **Is 2023 degradation concentrated in a subgroup?** No single tiny group explains it. The **entire 2023 fold** is worse than climatology (raw BSS −1366.5, 32.9% of OOF). It is **worse still** when `asof_pitcher_n` is small (2023 1–100: −6891, n=5,953; 101–500: −5068, n=19,911), but 2023 2000+ is still negative (−494). So: season-wide overprediction (p=0.520 vs y=0.500) **plus** extra damage in low pitcher-history cells.
2. **Cold-start pitcher?** `n==0` is too small to matter (share 0.0003). The material pitcher-history problem is **low-n, not zero-n**: 2023 1–500 is ~10.5% of that fold and severely negative. Concatenated OOF `n==0` also overpredicts (gap +0.096) with slightly more extreme probabilities (frac 0.047 vs 0.018 on 2000+; pred range still inside [0.44, 0.71], not collapsed to 0/1).
3. **Cold-start batter?** No. `asof_batter_n==0` is tiny and **better** than average (raw BSS +1108).
4. **Systematic count miscalibration?** Mild yes. Every count overpredicts slightly. Worst large cell is **0-2** (raw BSS 78, n=46,276, 6.2%). 3-2 has the largest gap among big cells (+0.022) but still positive BSS (649).
5. **High LI worse?** Only mildly (LI 3+ raw BSS 471 vs concat ~621). **Not** a primary failure. Do not add `li`.
6. **Late game worse?** **No** for innings 7–9 (better than 1–3). Extra innings 10+ are weak (BSS 97, 1.6%).
7. **Extreme predictions at small asof_n?** Partially. `n==0` pred std is higher (0.084 vs 0.051) but predictions are not degenerate at 0/1 (min 0.44, max 0.71). The failure is **miscalibration / overprediction**, not exploding probabilities.
8. **Link to feature instability?** Yes, circumstantial: 2023 is the SIGN_FLIP year for ball/strike asof rates **and** the failed fold. Independently, `prev1_<=0.35` (10.5% of OOF, raw BSS −184) is the worst large non-season slice; prev1 success is UNSTABLE in the stability audit. Same-hand 1x1 (gap +0.024, BSS 346) is the worst large hand cell.

Lowest raw BSS slices (all_oof, n≥2000, excluding overall/season):

| family | level | n | share | raw BSS | gap |
|---|---|---|---|---|---|
| prev1_extreme | prev1_<=0.35 | 78,607 | 0.105 | -184.2 | 0.0229 |
| prev1_missing | prev1_NA | 9,301 | 0.012 | -104.7 | 0.0304 |
| prev5_missing | prev5_NA | 9,301 | 0.012 | -104.7 | 0.0304 |
| prev1_extreme | prev1_NA | 9,301 | 0.012 | -104.7 | 0.0304 |
| inning | 11 | 3,186 | 0.004 | -56.6 | 0.0034 |
| count | 0-2 | 46,276 | 0.062 | 78.2 | 0.0105 |
| inning_bin | 10+ | 12,153 | 0.016 | 97.3 | 0.0089 |
| inning | 10 | 7,123 | 0.010 | 199.2 | 0.0142 |
| prev1_extreme | prev1_mid | 566,329 | 0.759 | 242.0 | 0.0101 |
| count | 1-2 | 71,595 | 0.096 | 280.4 | 0.0047 |
| count | 3-0 | 8,881 | 0.012 | 330.3 | 0.0069 |
| hand_matchup | 1x1 | 92,907 | 0.124 | 345.7 | 0.0242 |

## [FEATURE HYPOTHESES]

Maximum five. None implemented in this audit.

### FEH-01

- **hypothesis_id**: FEH-01
- **관찰**: `prev1_<=0.35` is the worst large non-season slice (n=78,607, 10.5% of OOF, raw BSS −184, gap +0.023). prev1 success is UNSTABLE; 2023 low `asof_pitcher_n` is also overpredicted.
- **제안 feature**: Row-local shrinkage of `asof_pitcher_prev1_game_success_rate` toward the same row's `asof_pitcher_success_rate`, with weight rising in `asof_pitcher_n` (no other-row / test aggregates).
- **예상 효과**: Less overprediction when last-game success is extreme or based on thin history; may help 2023 low-n cells without touching SIGN_FLIP ball/strike rates.
- **leakage 위험**: low if only current-row official asof fields are used
- **구현 난이도**: low
- **검증 방법**: EXP-004 recipe, same expanding folds; report overall 2022/2023/2024 raw BSS **and** `prev1_<=0.35` / pitcher_n_bin slices
- **우선순위**: high

### FEH-02

- **hypothesis_id**: FEH-02
- **관찰**: `balls_before`/`strikes_before` are numeric in EXP-004; 0-2 is the weakest large count (raw BSS 78, 6.2%). Categorical `balls_before` ranks are STABLE.
- **제안 feature**: Add one native categorical `count_code = balls_before-strikes_before` (do not drop the numeric columns in the same experiment).
- **예상 효과**: Non-monotonic count effects (0-2 vs 3-2) without leakage.
- **leakage 위험**: low (pre-pitch fields on the same row)
- **구현 난이도**: low
- **검증 방법**: same EXP-004 CV; inspect count-slice BSS/gaps
- **우선순위**: high

### FEH-03

- **hypothesis_id**: FEH-03
- **관찰**: Same-hand 1x1 (n=92,907, gap +0.024, raw BSS 346) and 2x2 (n=273,195, gap +0.018, BSS 438) lag platoon matchups (~628–660).
- **제안 feature**: Add one categorical `hand_matchup = pitcher_hand x batter_hand` (hands are already separate categoricals; this is an explicit interaction).
- **예상 효과**: Reduce same-hand overprediction.
- **leakage 위험**: low
- **구현 난이도**: low
- **검증 방법**: same CV; hand_matchup slices
- **우선순위**: medium (trees can already split the two hand columns)

### FEH-04

- **hypothesis_id**: FEH-04
- **관찰**: 2023 fails even at pitcher 2000+ (raw BSS −494). Ball/strike asof rates SIGN_FLIP. M/N season-relative asof already hurt 2023.
- **제안 feature**: Do **not** add more raw `asof_pitcher_ball_rate` / `strike_rate`. No new feature in this hypothesis — it is a negative result.
- **예상 효과**: Avoid wasting a slot on flipped rates.
- **leakage 위험**: n/a
- **구현 난이도**: n/a
- **검증 방법**: n/a (do not implement)
- **우선순위**: reject as an add; keep as a constraint

### FEH-05

- **hypothesis_id**: FEH-05
- **관찰**: Trackman physics (IVB/HB) and mix shifted 2022→2023; player IDs do not join. TM-002 is fold-safe (prior-season TM × current-row asof mix).
- **제안 feature**: `tm_expected_*` from TM-002 (see TM-AUDIT-001B). TM-001 as specified is HOLD (hand codes 1/2 vs Left/Right).
- **예상 효과**: Tell the model that the same mix implies different physics after the 2023 regime.
- **leakage 위험**: low if TM `season < valid_season` is enforced; never current-pitch TM / never test aggregates
- **구현 난이도**: medium
- **검증 방법**: TM-002 only, EXP-004 CV, no other FE in the same run
- **우선순위**: next after a cheaper row-local FE, not simultaneous

## Appendix

- slices CSV: `outputs/analysis/error_slices_001.csv`
- plots under `outputs/analysis/error_audit_001/`
- ECE is diagnostic only and was not used to select a model

```json
{
  "experiment_id": "EXP-004",
  "oof_path": "outputs/predictions/EXP-004/oof_holdout_predictions.csv",
  "oof_n_rows": 746504,
  "oof_columns": [
    "row_id",
    "season",
    "control_success",
    "prediction",
    "fold_id",
    "split_type"
  ],
  "row_id_unique": true,
  "n_duplicate_row_id": 0,
  "folds_present": [
    "primary_2024",
    "secondary_2022",
    "secondary_2023"
  ],
  "seasons_present": [
    2022,
    2023,
    2024
  ],
  "prediction_nan": 0,
  "prediction_inf": 0,
  "prediction_min": 0.3283647120351847,
  "prediction_max": 0.7997341571413175,
  "prediction_outside_01": 0,
  "target_nan": 0,
  "comparable_cv": true,
  "baseline_choice_reason": "EXP-004 is the latest CatBoost with is_comparable_cv=true, expanding 3-fold OOF covering 2022/2023/2024, and passed gpt_audit join verification. EXP-006 is hold (not adopt); its 2024 file is the Optuna tune holdout; EXP-005 has no row-level OOF.",
  "n_before_join": 746504,
  "n_after_join": 746504,
  "unmatched_pitcher_id_na": 0,
  "duplicate_row_increase": 0,
  "target_alignment_vs_train": true,
  "train_map_na": 0,
  "prior_join_verification_exp004": {
    "enriched_file": "EXP-004_oof_holdout_enriched.csv",
    "source_file": "outputs/predictions/EXP-004/oof_holdout_predictions.csv",
    "n_rows_before_join": 746504,
    "n_rows_after_join": 746504,
    "unmatched_rows_pitcher_id_na": 0,
    "duplicate_row_increase_vs_source": false,
    "n_duplicate_row_id_in_enriched": 0,
    "train_row_id_unique": true,
    "target_alignment_vs_source_same_order": true,
    "target_alignment_method": "same_row_order_elementwise",
    "target_alignment_vs_train_control_success": true,
    "train_map_na": 0,
    "expect_row_id_unique": false,
    "row_id_unique_ok": true
  },
  "fold_score_crosscheck": {
    "primary_2024": {
      "n": 253507,
      "raw_brier_skill_score": 578.1844155969828,
      "official_brier_skill_score": 578.1844155969828,
      "brier_score": 0.24836258203719846,
      "logged_raw_bss": 578.1844155969828,
      "matches_logged": true
    },
    "secondary_2022": {
      "n": 247472,
      "raw_brier_skill_score": 2259.3983277146012,
      "official_brier_skill_score": 2259.3983277146012,
      "brier_score": 0.24353400955510912,
      "logged_raw_bss": 2259.3983277146012,
      "matches_logged": true
    },
    "secondary_2023": {
      "n": 245525,
      "raw_brier_skill_score": -1366.4810668740479,
      "official_brier_skill_score": 0.0,
      "brier_score": 0.25341620081330546,
      "logged_raw_bss": -1366.4810668740479,
      "matches_logged": true
    }
  }
}
```
