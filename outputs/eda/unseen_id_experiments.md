# Unseen-ID / asof-relative / game_type experiments

- generated_at: `2026-08-07T07:06:17`
- base: CatBoost + EXP-004 season sample weights; season feature kept (except O)
- **no adopt/reject judgment — results only**

## References

| reference | fold | raw BSS |
|---|---|---:|
| EXP-004 | secondary_2023 | -1366.481067 |
| season dropped (D), unweighted | secondary_2023 | -1044.663582 |
| EXP-004 | primary_2024 | 578.184416 |

## Unseen ID rates (context for L)

| fold | column | unseen_row_rate | unseen_rows | unseen_nunique |
|---|---|---:|---:|---:|
| secondary_2023 | `pitcher_id` | 0.1380 | 33,878 | 63 |
| secondary_2023 | `batter_id` | 0.0940 | 23,091 | 71 |
| primary_2024 | `pitcher_id` | 0.1986 | 50,348 | 81 |
| primary_2024 | `batter_id` | 0.0930 | 23,588 | 88 |

## O diagnostic: game_type × season target rate

- types with early(2019–22) vs late(2023–24) **relative-to-season-mean sign flip**: `['F', 'R']`
- reversal_confirmed: `True`

| game_type | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|---:|---:|---:|
| `F` | 0.6892 | 0.5878 | 0.7038 | 0.7087 | 0.4729 | 0.4593 |
| `R` | 0.5495 | 0.5269 | 0.5128 | 0.5037 | 0.5031 | 0.4897 |

## Results: raw BSS by experiment × fold

| experiment | secondary_2023 raw BSS | Δ vs EXP-004(2023) | primary_2024 raw BSS | Δ vs EXP-004(2024) |
|---|---:|---:|---:|---:|
| EXP-004 reference | -1366.481067 | 0.000000 | 578.184416 | 0.000000 |
| season dropped D (ref, unweighted, 2023 only) | -1044.663582 | +321.817485 | — | — |
| L_drop_ids | -1306.558749 | +59.922317 | 739.969723 | +161.785307 |
| M_asof_relative | -1454.491675 | -88.010608 | 586.724068 | +8.539652 |
| N_L_plus_M | -1456.838737 | -90.357670 | 715.186726 | +137.002310 |
| O_drop_game_type | -384.121081 | +982.359986 | 464.493028 | -113.691388 |

## Detail rows

| experiment | fold | n_feat | asof_rel | raw BSS | official BSS | Brier | elapsed_s |
|---|---|---:|---|---:|---:|---:|---:|
| L_drop_ids | secondary_2023 | 42 | False | -1306.558749 | 0.000000 | 0.253266 | 473.3 |
| L_drop_ids | primary_2024 | 42 | False | 739.969723 | 739.969723 | 0.247958 | 191.1 |
| M_asof_relative | secondary_2023 | 44 | True | -1454.491675 | 0.000000 | 0.253636 | 194.8 |
| M_asof_relative | primary_2024 | 44 | True | 586.724068 | 586.724068 | 0.248341 | 224.5 |
| N_L_plus_M | secondary_2023 | 42 | True | -1456.838737 | 0.000000 | 0.253642 | 168.0 |
| N_L_plus_M | primary_2024 | 42 | True | 715.186726 | 715.186726 | 0.248020 | 197.9 |
| O_drop_game_type | secondary_2023 | 43 | False | -384.121081 | 0.000000 | 0.250960 | 182.1 |
| O_drop_game_type | primary_2024 | 43 | False | 464.493028 | 464.493028 | 0.248647 | 228.2 |

### M/N asof relative definition

- For each row with `season=S`, subtract mean(`rate_col`) over **train-fold rows with season < S**.
- Computed from train partition only; valid/test never enter the mean.
- If no prior season exists in train (e.g. S=2019), subtract `0.0`.
- Rate columns: `['asof_pitcher_success_rate', 'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate', 'asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate', 'asof_pitcher_prev5_game_success_rate', 'asof_pitcher_prev1_game_middle_rate', 'asof_pitcher_prev3_game_middle_rate', 'asof_pitcher_prev5_game_middle_rate', 'asof_batter_success_rate', 'asof_batter_middle_rate', 'asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate']`

- machine-readable: `outputs\eda\unseen_id_experiments.json`

