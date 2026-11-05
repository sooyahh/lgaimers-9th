# FE-002 Prev1 Shrinkage (k=500)

- generated_at: `2026-08-18T22:36:42`
- experiment_id: `FE-002`
- baseline: `EXP-004` (existing artifacts reused; not overwritten)
- prev1 column: `asof_pitcher_prev1_game_success_rate`
- prev1 values in the model: `row_local_shrinkage_k500` (not raw prev1)
- shrinkage_k: `500`

## 1. Hypothesis

Extreme / thin `asof_pitcher_prev1_game_success_rate` overstates last-game noise. A row-local shrink toward the same row's `asof_pitcher_success_rate`, with `w = n / (n + 500)`, should improve EXP-004 raw BSS on the original `prev1 <= 0.35` slice without dropping a temporal fold.

## 2. What changed

- In-place replacement of `asof_pitcher_prev1_game_success_rate` with k=500 shrinkage values
- Feature **names** unchanged: 44 → 44
- name delta baseline − FE-002 = `[]`
- name delta FE-002 − baseline = `[]`
- Raw prev1 is **not** a model input
- Not FE-002A: the column is not dropped
- Not FE-001: `count_code` is not added
- No Trackman columns

## 3. What stayed fixed

- CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss
- seed 42
- season weights 2019:1.0 … 2023:2.0
- expanding 2022/2023/2024 holdouts
- `src.metrics` unchanged
- fold-local median impute + native cats
- prev3/prev5, count numerics, other asof rates untouched

## 4. Frozen verdict rule

- ADOPT_CANDIDATE: original prev1<=0.35 raw BSS up, 3-fold mean raw BSS up, no fold delta < -50
- REJECT: prev1<=0.35 not improved, or mean down without temporal-stability gain
- HOLD: slice vs overall CV / folds conflict
- 2023 and 2023 × n<=500 are diagnostics only; they do not enter `decide_verdict`

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
| 2022 | 728,588 | 247,472 | 0.528920 | 0.532486 | 2251.442006 | 2251.442006 | 0.243554 |
| 2023 | 976,060 | 245,525 | 0.499957 | 0.519831 | -1340.347907 | 0.000000 | 0.253351 |
| 2024 | 1,221,585 | 253,507 | 0.486105 | 0.497804 | 576.586121 | 576.586121 | 0.248367 |

- FE-002 mean (ddof=0): `495.893407` (EXP-004 `490.367225`)
- FE-002 std (ddof=0): `1467.451801` (EXP-004 `1481.560943`)
- FE-002 worst: `-1340.347907` (EXP-004 `-1366.481067`)

## 7. Delta vs EXP-004

| Fold | EXP-004 | FE-002 | Delta |
|---|---:|---:|---:|
| 2022 | 2259.398328 | 2251.442006 | -7.956321 |
| 2023 | -1366.481067 | -1340.347907 | +26.133160 |
| 2024 | 578.184416 | 576.586121 | -1.598295 |
| Mean | 490.367225 | 495.893407 | +5.526181 |
| Std | 1481.560943 | 1467.451801 | -14.109142 |
| Worst fold | -1366.481067 | -1340.347907 | +26.133160 |

## 8. Prev1 slice results (original raw prev1 labels)

Bins match ERR-AUDIT-001. Slice membership uses raw `train.csv` prev1, not the shrunk modeling values.

| slice | n | EXP-004 raw BSS | FE-002 raw BSS | Δ BSS | EXP-004 gap | FE-002 gap |
|---|---:|---:|---:|---:|---:|---:|
| prev1_<=0.35 | 78,607 | -184.2 | -209.8 | -25.6 | +0.0229 | +0.0235 |
| prev1_mid | 566,329 | 242.0 | 246.3 | +4.3 | +0.0101 | +0.0104 |
| prev1_>=0.65 | 92,267 | 2030.3 | 2069.4 | +39.2 | +0.0056 | +0.0075 |
| prev1_NA | 9,301 | -104.7 | -101.0 | +3.7 | +0.0304 | +0.0325 |
| not_prev1_<=0.35 | 667,897 | 622.9 | 632.0 | +9.1 | +0.0098 | +0.0103 |

## 9. Season × prev1<=0.35

| slice | n | EXP-004 raw BSS | FE-002 raw BSS | Δ BSS | EXP-004 gap | FE-002 gap |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 21,628 | 1724.3 | 1683.2 | -41.0 | +0.0092 | +0.0096 |
| 2023 | 26,442 | -2434.0 | -2439.9 | -6.0 | +0.0409 | +0.0404 |
| 2024 | 30,537 | 248.2 | 216.6 | -31.5 | +0.0170 | +0.0188 |

## 10. Low-history diagnostics (not used in verdict)

### asof_pitcher_n bins

| slice | n | EXP-004 raw BSS | FE-002 raw BSS | Δ BSS | EXP-004 gap | FE-002 gap |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 232 | -2658.3 | -3352.8 | -694.5 | +0.0962 | +0.1013 |
| 1-100 | 21,442 | 349.5 | 458.6 | +109.1 | +0.0352 | +0.0369 |
| 101-500 | 68,670 | 815.6 | 838.3 | +22.7 | +0.0206 | +0.0224 |
| 501-2000 | 194,584 | 955.9 | 941.1 | -14.8 | +0.0128 | +0.0139 |
| 2000+ | 461,576 | 460.2 | 467.1 | +6.9 | +0.0079 | +0.0080 |

### n<=500 vs n>500

| slice | n | EXP-004 raw BSS | FE-002 raw BSS | Δ BSS | EXP-004 gap | FE-002 gap |
|---|---:|---:|---:|---:|---:|---:|
| n<=500 | 90,344 | 702.1 | 743.5 | +41.4 | +0.0242 | +0.0261 |
| n>500 | 656,160 | 609.6 | 610.1 | +0.5 | +0.0093 | +0.0097 |

### season × n<=500 (includes required 2023 × n<=500)

| slice | n | EXP-004 raw BSS | FE-002 raw BSS | Δ BSS | EXP-004 gap | FE-002 gap |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 32,327 | 3994.9 | 4014.9 | +20.1 | -0.0021 | -0.0016 |
| 2023 | 25,927 | -5485.9 | -5284.4 | +201.4 | +0.0679 | +0.0660 |
| 2024 | 32,090 | 197.8 | 131.8 | -66.0 | +0.0156 | +0.0216 |

## 11. Season folds

| slice | n | EXP-004 raw BSS | FE-002 raw BSS | Δ BSS | EXP-004 gap | FE-002 gap |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 247,472 | 2259.4 | 2251.4 | -8.0 | +0.0030 | +0.0036 |
| 2023 | 245,525 | -1366.5 | -1340.3 | +26.1 | +0.0204 | +0.0199 |
| 2024 | 253,507 | 578.2 | 576.6 | -1.6 | +0.0101 | +0.0117 |

## 12. Leakage and reproducibility

- EXP-004 / FE-001A / FE-002A / TM fingerprints unchanged: **True**
- shrinkage is a function of the current row's prev1, longterm, and n only
- test.csv loaded only for schema checks; no test-row aggregates
- seed 42; same CatBoost params and season weights as EXP-004
- Public LB: **not submitted**

## 13. Verdict

**REJECT**

Core prev1<=0.35 slice did not improve, or 3-fold mean worsened without temporal-stability gain (worst-fold up or std down, no collapse).

Single seed 42. Not a Public LB decision.

## 14. Next experiment

Stop. Do not auto-run another FE or TM experiment or Public LB.

- summary: `outputs/predictions/FE-002/summary.json`
- OOF: `outputs/predictions/FE-002/oof_holdout_predictions.csv`

## 15. Confirmed final verdict (post-review)

**REJECT** (confirmed). The frozen rule was not changed after seeing scores.

Confirmed grounds, copied from canonical artifacts:

- Original `prev1 <= 0.35` slice (raw `data/raw/train.csv` prev1, n=78,607): EXP-004 −184.202 → FE-002 −209.752, **Δ −25.550**.
- The same slice worsened in every temporal fold: 2022 −41.0, 2023 −6.0, 2024 −31.5.
- 3-fold mean raw BSS improved **+5.526** (490.367 → 495.893). That does not override the failed core slice condition.
- No fold delta was below −50 (2022 −7.956, 2023 +26.133, 2024 −1.598). Collapse was not the reject reason.
- Diagnostic only, **not** an adopt argument: `n<=500` Δ **+41.4**; 2023 × `n<=500` Δ **+201.4**.

Interpretation:

k=500 global row-local prev1 shrinkage는 low-prev1 문제 해결에는 실패했다.
다만 low-history(n<=500) 집단에서 개선 신호가 관찰되었으며,
이는 별도의 후속 가설일 뿐 FE-002 채택 근거가 아니다.
