# Data paths for analysis (read-only)

## Raw competition data (do not modify)

| path | role | notes from repo |
|---|---|---|
| `data/raw/train.csv` | labeled train | 1,475,092 × 49 (`data_integrity.md`) |
| `data/raw/test.csv` | test feature sample | 5 rows in local copy; no `control_success` |
| `data/raw/sample_submission.csv` | submission schema | keys `row_id` |
| `data/raw/trackman_history.csv` | Trackman history | **not joinable** to train on pitcher IDs (`target_regime_check.md`) |

Configured in `configs/baseline.yaml` under `data.*_path`.

## Column inventory

See `data_column_inventory.csv` (sourced from `outputs/eda/data_integrity.md`).

## Analysis-relevant context columns (for OOF joins)

Used when enriching existing predictions in `oof/`:

- ids: `row_id`, `pitcher_id`, `batter_id`, `pitcher_team_id`, `batter_team_id`
- time/context: `season`, `game_month`, `game_dayofweek`, `game_type`, `inning`, `top_bottom`
- count/state: `balls_before`, `strikes_before`, `outs_before`, `base_state`, `num_runners_on`, `score_diff_pitcher_team`
- asof: `asof_pitcher_n`, `asof_pitcher_success_rate`, `asof_batter_success_rate`
- target: `control_success` (train only)

Excluded from modeling (still present in train for leakage review if needed): `home_win_expectancy`, `away_win_expectancy`, `li` — **not** included in enriched OOF exports by default.
)
