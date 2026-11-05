# Gap Diagnosis: 2023 holdout vs future-season risk

- generated_at: `2026-08-07T04:45:05`
- setup: CatBoost EXP-003/004 params; train=`2019–2022`; valid=`2023`
- **no adopt/reject judgment — results only**

## Hypothesis under inspection

Predicting a future season (e.g. 2025/test) may share the same failure mode as secondary_2023 (train past seasons → valid next season).

## 1. EXP-004 settings on secondary_2023

Weights applied on train seasons present: `{"2019": 1.0, "2020": 1.2, "2021": 1.4, "2022": 1.6}` (from `SEASON_WEIGHTS`, 2023 key unused).

| run | n_features | sample_weight | raw BSS | official BSS | Brier |
|---|---:|---|---:|---:|---:|
| EXP-003 baseline (logged fold_metrics) | 44 | no | -1319.882960 | — | — |
| EXP-004 logged fold_metrics secondary_2023 | 44 | yes | -1366.481067 | — | — |
| EXP-004 style re-run (season + weights) | 44 | yes | -1366.481067 | 0.000000 | 0.253416 |

- re-run vs EXP-003 Δ raw BSS: `-46.598107`
- re-run vs EXP-004 logged Δ raw BSS: `+0.000000`

### Run details (EXP-004 style re-run)

- n_train / n_valid: `976,060` / `245,525`
- train target mean: `0.539537`
- valid target rate: `0.499957`
- pred mean±std: `0.520392` ± `0.065936`
- elapsed_s: `148.3`

## 3. Season kept vs dropped on the same 2019–2022 → 2023 split

| run | season | sample_weight | n_features | raw BSS | official BSS | Brier |
|---|---|---|---:|---:|---:|---:|
| season kept, unweighted | yes | no | 44 | -1319.882960 | 0.000000 | 0.253300 |
| season dropped (D), unweighted | no | no | 43 | -1044.663582 | 0.000000 | 0.252612 |
| season kept + weights (EXP-004 style) | yes | yes | 44 | -1366.481067 | 0.000000 | 0.253416 |
| season dropped + weights | no | yes | 43 | -1127.340273 | 0.000000 | 0.252818 |

### Paired deltas (raw BSS)

- drop − kept (unweighted): `+275.219378`
- drop − kept (weighted): `+239.140793`

- machine-readable: `outputs\eda\gap_diagnosis.json`

