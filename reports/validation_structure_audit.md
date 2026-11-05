# Validation Structure Audit

Generated from `data/raw/train.csv` only. No order keys were invented.
Validation strategy is **not** finalized by this report.

## Summary

- train shape: `(1475092, 49)`
- seasons present: `[2019, 2020, 2021, 2022, 2023, 2024]`
- season counts: `{2019: 237413, 2020: 244087, 2021: 247088, 2022: 247472, 2023: 245525, 2024: 253507}`
- explicit game_id-like columns: `none`
- calendar/game-context fields (not game ids): `['game_month', 'game_dayofweek', 'game_type']`
- id-like columns: `['row_id', 'pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id']`
- official event datetime column: **none** (`time_col` remains null)

## row_id as ordering signal

- `row_id` is unique: `True`
- file order equals lexicographic `row_id` sort: `True`
- all `row_id` have numeric suffix `_*digits`: `True`
- numeric suffixes unique: `True`
- numeric suffixes monotonic in file order: `True`
- `season` monotonic in file order: `True`

**Interpretation:** uniqueness of `row_id` does **not** by itself prove pitch chronology. Unless official docs state that `row_id` encodes time order, do not treat it as a temporal key.

## Available time-related fields

| Field | Role | Notes |
|---|---|---|
| `season` | coarse time axis | usable for season holdout / expanding-window |
| `game_month` | within-season month | not a full date; ties across games possible |
| `game_dayofweek` | weekday | weak; many collisions |
| event datetime | — | **absent** |
| pitch sequence within game | — | **no `pitch_no` / game_id in train** |

## Available group keys

| Key | Present | Notes |
|---|---|---|
| `pitcher_id` | yes | entity for pitcher-level diagnostics |
| `batter_id` | yes | entity for matchup diagnostics |
| `pitcher_team_id` / `batter_team_id` | yes | team groups |
| game id | **no** | cannot block game leakage via explicit game groups |

## Uniqueness / duplicate audit

| Candidate key | Result |
|---|---|
| `['row_id']` | unique=True, n_unique_keys=1475092, dup_rows=0, dup_groups=0, max_group_size=1, has_null=False |
| `['season']` | unique=False, n_unique_keys=6, dup_rows=1475092, dup_groups=6, max_group_size=253507, has_null=False |
| `['pitcher_id']` | unique=False, n_unique_keys=792, dup_rows=1475092, dup_groups=792, max_group_size=15450, has_null=False |
| `['batter_id']` | unique=False, n_unique_keys=830, dup_rows=1475091, dup_groups=829, max_group_size=13928, has_null=False |
| `['season', 'pitcher_id']` | unique=False, n_unique_keys=2260, dup_rows=1475091, dup_groups=2259, max_group_size=3525, has_null=False |
| `['season', 'batter_id']` | unique=False, n_unique_keys=2393, dup_rows=1475085, dup_groups=2386, max_group_size=2758, has_null=False |
| `['season', 'pitcher_id', 'batter_id']` | unique=False, n_unique_keys=150624, dup_rows=1467760, dup_groups=143292, max_group_size=96, has_null=False |
| `['season', 'pitcher_id', 'batter_id', 'inning', 'top_bottom']` | unique=False, n_unique_keys=345618, dup_rows=1438369, dup_groups=308895, max_group_size=28, has_null=False |
| `['season', 'pitcher_id', 'batter_id', 'inning', 'top_bottom', 'balls_before', 'strikes_before', 'outs_before']` | unique=False, n_unique_keys=1346716, dup_rows=228674, dup_groups=100298, max_group_size=11, has_null=False |
| `['season', 'game_month', 'game_dayofweek', 'pitcher_id', 'batter_id', 'inning', 'top_bottom', 'balls_before', 'strikes_before', 'outs_before', 'base_state', 'num_runners_on']` | unique=False, n_unique_keys=1373162, dup_rows=177896, dup_groups=75966, max_group_size=11, has_null=False |
| `['season', 'game_month', 'game_dayofweek', 'pitcher_team_id', 'batter_team_id', 'pitcher_id', 'batter_id', 'inning', 'top_bottom', 'balls_before', 'strikes_before', 'outs_before', 'base_state', 'li', 'home_win_expectancy', 'away_win_expectancy', 'score_diff_home', 'score_diff_pitcher_team', 'run_total_before']` | unique=False, n_unique_keys=1373207, dup_rows=177809, dup_groups=75924, max_group_size=11, has_null=False |
| `['season', 'pitcher_id', 'batter_id', 'inning', 'top_bottom', 'balls_before', 'strikes_before', 'outs_before', 'base_state', 'game_month', 'game_dayofweek', 'li']` | unique=False, n_unique_keys=1373200, dup_rows=177822, dup_groups=75930, max_group_size=11, has_null=False |

## Pitch-order recoverability within a game

No explicit game identifier exists in `train.csv`. Same pitcher–batter–inning–count context can still collide across games/days. **Within-game pitch order cannot be reconstructed from provided columns alone.**

Rich context key uniqueness: `unique=False, n_unique_keys=1373200, dup_rows=177822, dup_groups=75930, max_group_size=11, has_null=False`
Even a rich situational key still has `177822` duplicate rows / `75930` colliding groups (max group size `11`). Do **not** invent a synthetic order to break ties.

## Temporal validation feasibility

- **Season-level temporal split: possible.** `season` is present, non-null for audit purposes, and covers 2019–2024.
- Expanding-window / latest-season holdout can be built on `season` without requiring pitch order.
- Within-season chronological CV is **not supported** by available keys (no date/time/pitch_no/game_id).
- `order_cols` in config remains `[]` until an officially unique chronological key is confirmed.

## Game leakage blocking

- **Not possible with an explicit game group key** (no game id).
- Season split reduces cross-season leakage but does not prevent same-game rows from landing in both train and valid within a season.
- Approximate proxies (team+month+dow+inning+...) are **not unique** and must not be treated as game ids.

## Still unknown / not confirmed

- Whether `row_id` numeric order matches true pitch chronology (official docs do not state this).
- Mapping between main-data anonymized IDs and Trackman IDs (direct intersection was 0; out of scope here).
- True within-game sequence for 2025 evaluation rows.
- Final choice among season expanding / holdout / other schemes (explicitly **not finalized**).
- Whether any undocumented tie-breaker exists outside the CSV columns.

## Recommendation (non-final)

1. Keep `data.order_cols: []` and `data.time_col: null` until a unique chronological key is proven.
2. Prefer season-based temporal splits for local CV candidates.
3. Do not fabricate pitch order from `row_id` or colliding situational keys.
4. Do not claim game-leakage-free validation without a game identifier.

