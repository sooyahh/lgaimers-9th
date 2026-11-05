# Data Integrity Report

- source_train: data\raw\train.csv
- source_test: data\raw\test.csv (format sample; not full evaluation set)
- generated_with: pandas integrity scan

## 1. Shape and dtypes

### train.csv
- rows: 1,475,092
- columns: 49

| column | dtype |
|---|---|
| row_id | str |
| season | int64 |
| game_month | int64 |
| game_dayofweek | int64 |
| inning | int64 |
| top_bottom | str |
| game_type | str |
| balls_before | int64 |
| strikes_before | int64 |
| outs_before | int64 |
| run_top_before | int64 |
| run_bot_before | int64 |
| run_total_before | int64 |
| score_diff_home | int64 |
| score_diff_pitcher_team | int64 |
| runner_on_1b | int64 |
| runner_on_2b | int64 |
| runner_on_3b | int64 |
| num_runners_on | int64 |
| base_state | str |
| home_win_expectancy | float64 |
| away_win_expectancy | float64 |
| li | float64 |
| pitcher_id | int64 |
| batter_id | int64 |
| pitcher_hand | int64 |
| batter_hand | int64 |
| pitcher_team_id | int64 |
| batter_team_id | int64 |
| asof_pitcher_n | int64 |
| asof_pitcher_success_rate | float64 |
| asof_pitcher_reverse_rate | float64 |
| asof_pitcher_middle_rate | float64 |
| asof_pitcher_ball_rate | float64 |
| asof_pitcher_strike_rate | float64 |
| asof_pitcher_prev1_game_success_rate | float64 |
| asof_pitcher_prev3_game_success_rate | float64 |
| asof_pitcher_prev5_game_success_rate | float64 |
| asof_pitcher_prev1_game_middle_rate | float64 |
| asof_pitcher_prev3_game_middle_rate | float64 |
| asof_pitcher_prev5_game_middle_rate | float64 |
| asof_batter_n | int64 |
| asof_batter_success_rate | float64 |
| asof_batter_middle_rate | float64 |
| asof_pitcher_pitchmix_n | int64 |
| asof_pitcher_fastball_rate | float64 |
| asof_pitcher_breaking_rate | float64 |
| asof_pitcher_offspeed_rate | float64 |
| control_success | int64 |

### test.csv (deployed sample)
- rows: 5
- columns: 48
- columns vs train (train-only): ['control_success']
- columns vs train (test-only): []

## 2. Duplicate rows

- exact full-row duplicates (keep="first" count of extras): 0
- rows involved in any exact full-row duplicate group: 0

### Key-column uniqueness / duplicates

| key | nunique | n_rows | n_duplicate_rows (keep=False) | is_unique |
|---|---:|---:|---:|---|
| row_id | 1475092 | 1475092 | 0 | True |
| pitcher_id | 792 | 1475092 | 1475092 | False |
| batter_id | 830 | 1475092 | 1475091 | False |
| pitcher_team_id | 13 | 1475092 | 1475092 | False |
| batter_team_id | 13 | 1475092 | 1475092 | False |
| season | 6 | 1475092 | 1475092 | False |

### Composite key duplicates

| key | nunique | n_duplicate_rows | is_unique |
|---|---:|---:|---|
| ['season', 'pitcher_id'] | 2260 | 1475091 | False |
| ['season', 'batter_id'] | 2393 | 1475085 | False |
| ['season', 'pitcher_id', 'batter_id'] | 150624 | 1467760 | False |
| ['season', 'pitcher_id', 'batter_id', 'inning', 'top_bottom', 'balls_before', 'strikes_before', 'outs_before'] | 1346716 | 228674 | False |

- explicit game_id-like columns present: none

## 3. Join-key candidate cardinality

| column | nunique (dropna) | n_null | notes |
|---|---:|---:|---|
| row_id | 1475092 | 0 | submission/sample key |
| pitcher_id | 792 | 0 | entity/team id candidate |
| batter_id | 830 | 0 | entity/team id candidate |
| pitcher_team_id | 13 | 0 | entity/team id candidate |
| batter_team_id | 13 | 0 | entity/team id candidate |

- **No `game_id` / `trackman_game_id` column exists in train.csv.** Join to Trackman cannot use a direct game key from main data.

## 4. Missing-value rates by column (descending)

- columns with any NA: 16 / 49

| column | missing_rate | n_missing |
|---|---:|---:|
| asof_pitcher_prev5_game_success_rate | 0.019785 | 29185 |
| asof_pitcher_prev3_game_middle_rate | 0.019785 | 29185 |
| asof_pitcher_prev5_game_middle_rate | 0.019785 | 29185 |
| asof_pitcher_prev3_game_success_rate | 0.019785 | 29185 |
| asof_pitcher_prev1_game_success_rate | 0.019785 | 29185 |
| asof_pitcher_prev1_game_middle_rate | 0.019785 | 29185 |
| asof_batter_success_rate | 0.000563 | 830 |
| asof_batter_middle_rate | 0.000563 | 830 |
| asof_pitcher_middle_rate | 0.000537 | 792 |
| asof_pitcher_strike_rate | 0.000537 | 792 |
| asof_pitcher_ball_rate | 0.000537 | 792 |
| asof_pitcher_breaking_rate | 0.000537 | 792 |
| asof_pitcher_fastball_rate | 0.000537 | 792 |
| asof_pitcher_offspeed_rate | 0.000537 | 792 |
| asof_pitcher_reverse_rate | 0.000537 | 792 |
| asof_pitcher_success_rate | 0.000537 | 792 |

### Zero-missing columns count
- 33 columns have 0 missing values

## 5. Season row counts and date/period coverage

### Available time-related fields
- present: ['season', 'game_month', 'game_dayofweek']
- **No full date column (game_date / event_time) in train.csv.** Period can only be summarized via season + game_month (+ weekday).

### All seasons

| season | n_rows | game_month min–max | game_dayofweek values |
|---:|---:|---|---|
| 2019 | 237,413 | 3–10 | [0, 1, 2, 3, 4, 5, 6] |
| 2020 | 244,087 | 5–10 | [0, 1, 2, 3, 4, 5, 6] |
| 2021 | 247,088 | 4–10 | [0, 1, 2, 3, 4, 5, 6] |
| 2022 | 247,472 | 4–10 | [0, 1, 2, 3, 4, 5, 6] |
| 2023 | 245,525 | 4–10 | [0, 1, 2, 3, 4, 5, 6] |
| 2024 | 253,507 | 3–10 | [0, 1, 2, 3, 4, 5, 6] |

### Requested years 2022 / 2023 / 2024

| season | n_rows | share_of_train | month range | n_distinct_months | weekday range |
|---:|---:|---:|---|---:|---|
| 2022 | 247,472 | 0.1678 | 4–10 (months=[4, 5, 6, 7, 8, 9, 10]) | 7 | [0, 1, 2, 3, 4, 5, 6] |
| 2023 | 245,525 | 0.1664 | 4–10 (months=[4, 5, 6, 7, 8, 9, 10]) | 7 | [0, 1, 2, 3, 4, 5, 6] |
| 2024 | 253,507 | 0.1719 | 3–10 (months=[3, 4, 5, 6, 7, 8, 9, 10]) | 8 | [0, 1, 2, 3, 4, 5, 6] |

### Interpretation notes

- Date *range* is not a calendar min/max date; only season-year and within-season month/weekday are available.
- test.csv seasons (sample): [2025]

