# Leakage Check Report

- source: `data/raw/train.csv`
- focus: `asof_pitcher_prev*_game_*` temporal sanity, CV fold separation, pitcher cross-year overlap
- note: row-level competition samples are **not** published here; local-only detail (if regenerated) lives in gitignored `outputs/eda/_leakage_meta.json`.

## 1. asof column verification (`asof_pitcher_prev*_game_*`)

### Ordering caveat

- train.csv has **no `game_id` / event timestamp**. Chronology within a pitcher-season is approximated by sorting on `asof_pitcher_n` (official cumulative pitch count before the current pitch), then `row_id`.
- `game_month` / `game_dayofweek` are secondary context and used for an approximate “first 5 games” proxy (new proxy-game when month or weekday changes for that pitcher).
- This is a **sanity check**, not a proof of official feature construction.

### Illustrative single pitcher-season (de-identified)

- One real pitcher-season was inspected locally as `SAMPLE_PITCHER` (identity and `row_id` withheld from this tracked report).
- expected if past-only: cold-start NA → values after enough prior games → infrequent updates (game-level) → no mid-season flip back to NA.

### Observations on this sample

- `asof_pitcher_n` is strictly non-decreasing along the sort (`decreases=0` along the selected pitcher-season).
- Early ranks are all-NA on every `prev*` column; values become available near the first non-null rank and then remain populated.
- After first non-null, null re-entry count for this pitcher-season: `0` on all six `prev*` columns.
- `prev1_success` discrete value changes across the season: `33` (game-step like; not pitch-by-pitch churn).

### Global re-entry check (all pitcher × season)

- pitcher-season groups: `2260`
- global `asof_pitcher_n` decreases under the same sort: `0`

| column | nulls after first non-null (row count) | pitcher-seasons with any re-entry |
|---|---:|---:|
| `asof_pitcher_prev1_game_success_rate` | 0 | 0 |
| `asof_pitcher_prev3_game_success_rate` | 0 | 0 |
| `asof_pitcher_prev5_game_success_rate` | 0 | 0 |
| `asof_pitcher_prev1_game_middle_rate` | 0 | 0 |
| `asof_pitcher_prev3_game_middle_rate` | 0 | 0 |
| `asof_pitcher_prev5_game_middle_rate` | 0 | 0 |

- Interpretation: once a `prev*` value appears in a pitcher-season, it does **not** return to NA later under this ordering — consistent with cold-start then past-game aggregation.

### Early-season (first ~5 games) vs later: missing / outlier rates

Game proxy: within each `(pitcher_id, season)`, sort by `asof_pitcher_n`; start a new proxy-game when `game_month` or `game_dayofweek` changes. “First 5 games” = `game_rank <= 5`.

| segment | n_rows | miss prev1_success | miss prev3_success | miss prev5_success | miss prev1_middle | miss prev3_middle | miss prev5_middle |
|---|---:|---:|---:|---:|---:|---:|---:|
| first_5_game_proxies | 369,600 | 0.078964 | 0.078964 | 0.078964 | 0.078964 | 0.078964 | 0.078964 |
| after_5_game_proxies | 1,105,492 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| months_le_4 | 232,394 | 0.061568 | 0.061568 | 0.061568 | 0.061568 | 0.061568 | 0.061568 |
| months_ge_5 | 1,242,698 | 0.011972 | 0.011972 | 0.011972 | 0.011972 | 0.011972 | 0.011972 |

| segment | outlier rate (value not in [0,1]) prev1_s / prev3_s / prev5_s / prev1_m / prev3_m / prev5_m |
|---|---|
| first_5_game_proxies | 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 |
| after_5_game_proxies | 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 |

**Verdict (asof):** Early proxy-games have markedly higher missingness (~7.9% vs ~0% after 5 games; months≤4 also higher than later months). Non-null values stay in `[0,1]` (outlier rate 0). Pattern matches past-only cold-start behavior; no evidence in this check of future leakage via mid-season NA re-entry or out-of-range rates.

## 2. CV group structure (`tree_exp_common.py`)

### Fold construction as used by tree experiments

From `scripts/tree_exp_common.py` (fold specs + holdout call):

```python
primary = cv_cfg["primary"]
secondary = list(cv_cfg.get("secondary_folds") or [])
fold_specs = [
    {
        "fold_id": "secondary_2022",
        "train_seasons": list(secondary[0]["train_seasons"]),
        "valid_seasons": list(secondary[0]["valid_seasons"]),
        "role": "secondary",
    },
    {
        "fold_id": "secondary_2023",
        "train_seasons": list(secondary[1]["train_seasons"]),
        "valid_seasons": list(secondary[1]["valid_seasons"]),
        "role": "secondary",
    },
    {
        "fold_id": "primary_2024",
        "train_seasons": list(primary["train_seasons"]),
        "valid_seasons": list(primary["valid_seasons"]),
        "role": "primary",
    },
]

for spec in fold_specs:
    fold = build_season_holdout(
        train_df,
        season_col,
        spec["train_seasons"],
        spec["valid_seasons"],
        name=spec["fold_id"],
    )
```

Underlying splitter (`src/evaluate.py::build_season_holdout`):

```python
def build_season_holdout(...):
    """Explicit season holdout (e.g. 2019-2023 train / 2024 valid)."""
    ...
    if set(train_seasons_t) & set(valid_seasons_t):
        raise ValueError("train_seasons and valid_seasons must be disjoint.")
    if max(train_seasons_t) >= min(valid_seasons_t):
        raise ValueError(
            "train seasons must be strictly earlier than valid seasons "
            f"(train_max={max(train_seasons_t)}, valid_min={min(valid_seasons_t)})."
        )
    return _make_fold(...)
```

### Configured seasons (`configs/baseline.yaml`)

| fold_id | train_seasons | valid_seasons |
|---|---|---|
| secondary_2022 | 2019–2021 | 2022 |
| secondary_2023 | 2019–2022 | 2023 |
| primary_2024 | 2019–2023 | 2024 |

### Explicit answers

- **Random shuffle in fold split?** **No.** Indices come from boolean masks on `season` only (`build_season_holdout` / `_make_fold`). There is no `shuffle`, `sample`, or `train_test_split` in the fold construction path.
- **Season 완전 분리?** **Yes for train vs valid within each fold:** train/valid season sets are required to be disjoint, and train seasons must be strictly earlier than valid seasons (expanding temporal holdout). Valid seasons do not leak into that fold’s training rows.
- Note: the same early seasons appear in multiple folds’ training sets (expanding CV); that is intentional and is **not** random mixing of a valid season into train.
- `runtime.seed` is used for model RNG (`set_seed` / LightGBM `random_state` / CatBoost `random_seed`), **not** for splitting.

## 3. Pitcher cross-year overlap (2022 / 2023 / 2024)

- distinct `pitcher_id` appearing in any of 2022–2024: `591`

| n_years present (among 2022–2024) | n_pitchers | share |
|---:|---:|---:|
| 1 | 227 | 0.3841 |
| 2 | 156 | 0.2640 |
| 3 | 208 | 0.3519 |

### Year-set detail

| year set | n_pitchers |
|---|---:|
| `[2022, 2023, 2024]` | 208 |
| `[2024]` | 108 |
| `[2022]` | 81 |
| `[2022, 2023]` | 81 |
| `[2023, 2024]` | 55 |
| `[2023]` | 38 |
| `[2022, 2024]` | 20 |

**Implication for CV:** many pitchers (208 / 591 ≈ 35%) appear in **all three** fold years. Season holdout prevents *row* leakage from the valid season into train, but **entity (pitcher) identity still overlaps** across years — models can memorize pitcher-specific patterns that transfer into later holdouts.

