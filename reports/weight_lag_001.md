# WEIGHT-LAG-001

- generated_at: `2026-08-19T23:15:34`
- smoke: `false`
- seed: `42`
- weight_policy: `season_lag_geometric`
- lag_definition: `lag = valid_season - train_season; w(lag) = m ** (1 - (lag - 1) / (L - 1)); L = max lag in fold; L==1 -> w=m`
- feature_set: L42 (`42` features; drop `['pitcher_id', 'batter_id']`; keep `game_type`)
- CatBoost params: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`
- decision: `results_only_no_adopt_reject`

## Objective

RECENCY-AUDIT-001 full-history training window를 유지한 채, validation 시즌에
가까운 학습 시즌에 더 큰 sample weight를 주는 것이 다음 시즌 raw BSS를
바꾸는지 측정한다. 바뀌는 핵심 변수는 multiplier `m` 하나뿐이다.

## Experimental contract

- 5 multipliers × 3 folds = 15 CatBoost L42 fits
- multipliers: `[1.0, 1.5, 2.0, 3.0, 4.0]`
- `m = 1.0` is CONTROL (all training weights = 1.0)
- 바뀌는 것: validation-relative season-lag sample weight
- 바뀌지 않는 것: full-history window, L42, CatBoost hyperparameters, seed 42, metric, preprocessing
- Public LB / 신규 submission 없음
- ADOPT / REJECT / HOLD 판정 없음

## Canonical L42 controls

- backend: `catboost`
- params lock: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`
- seed lock: `42`
- dropped IDs: `['pitcher_id', 'batter_id']`
- `game_type` kept as categorical
- missing numeric: fold-local median (`prepare_native_frames`)
- categorical: CatBoost native string/`__NA__`
- metric: `src.evaluate.evaluate_predictions` → `src.metrics.compute_brier_metrics`
- comparison metric: **raw BSS** (official clipped BSS is recorded only)

## Weight definition

```text
lag = y - s
L = max lag in the fold
w(lag) = m ** (1 - (lag - 1) / (L - 1))
w(1) = m
w(L) = 1
L == 1 -> w = m
```

Calendar-year fixed weights are not used. The same training year can have
different weights across validation folds.

## Fold definitions

| fold | valid | train seasons |
|---|---|---|
| secondary_2022 | 2022 | `[2019, 2020, 2021]` |
| secondary_2023 | 2023 | `[2019, 2020, 2021, 2022]` |
| primary_2024 | 2024 | `[2019, 2020, 2021, 2022, 2023]` |

## Fit-level results

| valid | m | train seasons | train rows | valid rows | season weights | raw BSS | Brier | fit_s |
|---|---|---|---|---|---|---|---|---|
| 2022 | 1.0 | `2019,2020,2021` | 728588 | 247472 | `{"2019": 1.0, "2020": 1.0, "2021": 1.0}` | 2322.459708 | 0.24337688 | 71.8 |
| 2022 | 1.5 | `2019,2020,2021` | 728588 | 247472 | `{"2019": 1.0, "2020": 1.224744871391589, "2021": 1.5}` | 2323.260649 | 0.24337489 | 72.9 |
| 2022 | 2.0 | `2019,2020,2021` | 728588 | 247472 | `{"2019": 1.0, "2020": 1.4142135623730951, "2021": 2.0}` | 2296.495198 | 0.24344158 | 74.1 |
| 2022 | 3.0 | `2019,2020,2021` | 728588 | 247472 | `{"2019": 1.0, "2020": 1.7320508075688772, "2021": 3.0}` | 2305.345754 | 0.24341953 | 73.5 |
| 2022 | 4.0 | `2019,2020,2021` | 728588 | 247472 | `{"2019": 1.0, "2020": 2.0, "2021": 4.0}` | 2285.699889 | 0.24346848 | 73.0 |
| 2023 | 1.0 | `2019,2020,2021,2022` | 976060 | 245525 | `{"2019": 1.0, "2020": 1.0, "2021": 1.0, "2022": 1.0}` | -1261.539976 | 0.25315385 | 96.2 |
| 2023 | 1.5 | `2019,2020,2021,2022` | 976060 | 245525 | `{"2019": 1.0, "2020": 1.1447142425533319, "2021": 1.3103706971044484, "2022": 1.5}` | -1384.522204 | 0.25346130 | 96.3 |
| 2023 | 2.0 | `2019,2020,2021,2022` | 976060 | 245525 | `{"2019": 1.0, "2020": 1.2599210498948732, "2021": 1.5874010519681996, "2022": 2.0}` | -1361.800721 | 0.25340450 | 96.0 |
| 2023 | 3.0 | `2019,2020,2021,2022` | 976060 | 245525 | `{"2019": 1.0, "2020": 1.4422495703074085, "2021": 2.080083823051904, "2022": 3.0}` | -1363.226055 | 0.25340806 | 95.1 |
| 2023 | 4.0 | `2019,2020,2021,2022` | 976060 | 245525 | `{"2019": 1.0, "2020": 1.5874010519681996, "2021": 2.5198420997897464, "2022": 4.0}` | -1381.648049 | 0.25345412 | 95.1 |
| 2024 | 1.0 | `2019,2020,2021,2022,2023` | 1221585 | 253507 | `{"2019": 1.0, "2020": 1.0, "2021": 1.0, "2022": 1.0, "2023": 1.0}` | 734.819045 | 0.24797130 | 116.2 |
| 2024 | 1.5 | `2019,2020,2021,2022,2023` | 1221585 | 253507 | `{"2019": 1.0, "2020": 1.1066819197003217, "2021": 1.224744871391589, "2022": 1.3554030054147672, "2023": 1.5}` | 722.013950 | 0.24800329 | 116.2 |
| 2024 | 2.0 | `2019,2020,2021,2022,2023` | 1221585 | 253507 | `{"2019": 1.0, "2020": 1.189207115002721, "2021": 1.4142135623730951, "2022": 1.681792830507429, "2023": 2.0}` | 740.743912 | 0.24795650 | 116.3 |
| 2024 | 3.0 | `2019,2020,2021,2022,2023` | 1221585 | 253507 | `{"2019": 1.0, "2020": 1.3160740129524924, "2021": 1.7320508075688772, "2022": 2.2795070569547775, "2023": 3.0}` | 740.890236 | 0.24795613 | 115.5 |
| 2024 | 4.0 | `2019,2020,2021,2022,2023` | 1221585 | 253507 | `{"2019": 1.0, "2020": 1.4142135623730951, "2021": 2.0, "2022": 2.8284271247461903, "2023": 4.0}` | 726.710337 | 0.24799155 | 114.7 |

## Compact raw BSS

| m | 2022 | 2023 | 2024 | mean | std | min |
|---|---|---|---|---|---|---|
| 1.0 | 2322.459708 | -1261.539976 | 734.819045 | 598.579592 | 1795.879819 | -1261.539976 |
| 1.5 | 2323.260649 | -1384.522204 | 722.013950 | 553.584132 | 1859.620895 | -1384.522204 |
| 2.0 | 2296.495198 | -1361.800721 | 740.743912 | 558.479463 | 1835.945943 | -1361.800721 |
| 3.0 | 2305.345754 | -1363.226055 | 740.890236 | 561.003312 | 1840.889529 | -1363.226055 |
| 4.0 | 2285.699889 | -1381.648049 | 726.710337 | 543.587392 | 1840.519148 | -1381.648049 |

## Contrasts vs m=1.0 CONTROL

`delta_raw_bss = treatment_raw_bss - control_raw_bss`

| m | Δ2022 | Δ2023 | Δ2024 | mean Δ | std Δ | min Δ |
|---|---|---|---|---|---|---|
| 1.0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1.5 | 0.800941 | -122.982228 | -12.805094 | -44.995460 | 67.880284 | -122.982228 |
| 2.0 | -25.964510 | -100.260745 | 5.924867 | -40.100129 | 54.485850 | -100.260745 |
| 3.0 | -17.113954 | -101.686079 | 6.071192 | -37.576280 | 56.718052 | -101.686079 |
| 4.0 | -36.759819 | -120.108073 | -8.108708 | -54.992200 | 58.183153 | -120.108073 |

## Fold stability

Values below are descriptive statistics only. No adoption decision is made here.

- m=1.0: mean raw BSS `598.579592`, std `1795.879819`, min `-1261.539976`; mean Δ `0.000000`, std Δ `0.000000`, min Δ `0.000000`
- m=1.5: mean raw BSS `553.584132`, std `1859.620895`, min `-1384.522204`; mean Δ `-44.995460`, std Δ `67.880284`, min Δ `-122.982228`
- m=2.0: mean raw BSS `558.479463`, std `1835.945943`, min `-1361.800721`; mean Δ `-40.100129`, std Δ `54.485850`, min Δ `-100.260745`
- m=3.0: mean raw BSS `561.003312`, std `1840.889529`, min `-1363.226055`; mean Δ `-37.576280`, std Δ `56.718052`, min Δ `-101.686079`
- m=4.0: mean raw BSS `543.587392`, std `1840.519148`, min `-1381.648049`; mean Δ `-54.992200`, std Δ `58.183153`, min Δ `-120.108073`

## m=1.0 reproduction gate

- status: `PASS`
- raw BSS atol: `1e-09`
- brier atol: `1e-12`

| valid | recency raw BSS | control raw BSS | Δ | train rows | valid rows |
|---|---|---|---|---|---|
| 2022 | 2322.459708 | 2322.459708 | -0.000000 | 728588 | 247472 |
| 2023 | -1261.539976 | -1261.539976 | 0.000000 | 976060 | 245525 |
| 2024 | 734.819045 | 734.819045 | 0.000000 | 1221585 | 253507 |

## Limitations

- single seed 42
- CatBoost L42 only
- full-history window locked from RECENCY-AUDIT-001 B_expanding
- Public LB not used
- no submission
- `load_competition_data` opens test.csv, but this experiment never uses it for weights, preprocessing, or metrics

## Commands actually run

- `python -m scripts.run_weight_lag_001`

## experiments.csv

이번 실험은 `outputs/experiments.csv`에 append하지 않았다.
RECENCY-AUDIT-001 / VAL-AUDIT-001과 같은 진단 convention이다.

---

## Team decision

This section is the current final status for WEIGHT-LAG-001.
It does **not** rewrite the results-only body above. Those numbers remain the canonical run record.

**Decision: REJECT**

### Scope of this REJECT

REJECT applies only to this formulation and experimental range:

- L42
- CatBoost
- iterations=300, learning_rate=0.05, depth=6, Logloss
- seed=42
- full-history expanding validation (RECENCY-AUDIT-001 `B_expanding` windows)
- geometric validation-relative season-lag weights
- `m ∈ {1.5, 2.0, 3.0, 4.0}`
- `w(lag) = m ** (1 - (lag - 1) / (L - 1))`

This is **not** a rejection of recency weighting as a general idea.
It does not authorize a new `m` sweep, a different lag formula, a different backend, or a change to the full-history window.

### Retained training policy

Keep:

- full-history expanding windows
- uniform sample weight (`m = 1.0`)

### Evidence copied from canonical artifacts

Source: `outputs/analysis/weight_lag_001/contrasts.csv` and `results.csv`.
`m=1.0` CONTROL reproduced RECENCY-AUDIT-001 `B_expanding` on all three folds (raw BSS / Brier / row counts exact; gate `PASS`).

All `m > 1` treatments have **negative mean Δ raw BSS** versus `m=1.0`:

| m | mean Δ | Δ2022 | Δ2023 | Δ2024 |
|---|---:|---:|---:|---:|
| 1.5 | -44.995460 | +0.800941 | -122.982228 | -12.805094 |
| 2.0 | -40.100129 | -25.964510 | -100.260745 | +5.924867 |
| 3.0 | -37.576280 | -17.113954 | -101.686079 | +6.071192 |
| 4.0 | -54.992200 | -36.759819 | -120.108073 | -8.108708 |

- Every `m > 1` treatment **worsens 2023** by about −100 to −123 raw BSS.
- 2024 has only small gains at `m=2.0` (+5.924867) and `m=3.0` (+6.071192). Those do not offset the 2023 deterioration, and the three-fold mean remains negative.
- There is **no monotonic improvement** as `m` increases (mean Δ: −44.995460, −40.100129, −37.576280, −54.992200).

Therefore this geometric lag-weighting formulation is closed. No fine-grained follow-up `m` sweep (`WEIGHT-LAG-002`, `m=1.1/1.2/…`, LB-based weight selection) will be run from this result.

### What this does and does not imply

- It does **not** discard recency weighting in general; only this geometric lag schedule on locked CatBoost L42 is REJECTED.
- The 2023 fold getting worse under up-weighting of recent seasons **weakens** the hypothesis that the 2023 shift is simply old-data contamination or a simple recency problem.
- Next analytical priority is **2023 shift cause analysis** (feature / context / regime), not additional weight tuning.

`outputs/experiments.csv` was not appended (audit convention; see above). Canonical numeric artifacts under `outputs/analysis/weight_lag_001/` were not rewritten for this decision.
