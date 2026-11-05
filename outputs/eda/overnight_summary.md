# Overnight summary (A / B / C after EXP-006)

- generated_at: `2026-08-07T16:06:59`
- common: CatBoost EXP-005 holdout (L, game_type kept); train 2019–2023 → valid 2024; raw BSS
- **results only — no adopt/reject; no submission**

## Seed noise reference (Task A)

- seeds: `[42, 1, 7, 123, 2024]`
- mean raw BSS: `728.611546`
- **seed std (sample): `7.842338`**
- range (max−min): `17.114096`

| seed | raw BSS |
|---:|---:|
| 42 | 739.969723 |
| 1 | 733.738086 |
| 7 | 723.379950 |
| 123 | 722.855627 |
| 2024 | 723.114344 |

## Task B — weight schedules vs B-1

- B-1 raw BSS: `739.969723`
- compare Δ to seed_std `7.842338`

| schedule | raw BSS | Δ vs B-1 | vs seed noise |
|---|---:|---:|---|
| B-1_current | 739.969723 | +0.000000 | |Δ| ≤ seed_std (7.842338) |
| B-2_mild | 725.401003 | -14.568720 | |Δ| > seed_std (7.842338) |
| B-3_aggressive | 735.940909 | -4.028814 | |Δ| ≤ seed_std (7.842338) |
| B-4_extreme | 719.870339 | -20.099384 | |Δ| > seed_std (7.842338) |

## Task C — reinstate excluded features vs C-0

- C-0 (42-feat baseline) raw BSS: `739.969723`
- safety: all of ['li', 'home_win_expectancy', 'away_win_expectancy'] are pre-pitch per data_description.md
- compare Δ to seed_std `7.842338`

| variant | n_feat | raw BSS | Δ vs C-0 | vs seed noise |
|---|---:|---:|---:|---|
| C-0_baseline_42 | 42 | 739.969723 | +0.000000 | |Δ| ≤ seed_std (7.842338) |
| C-1_add_li | 43 | 743.857943 | +3.888220 | |Δ| ≤ seed_std (7.842338) |
| C-2_add_home_win_expectancy | 43 | 743.458972 | +3.489249 | |Δ| ≤ seed_std (7.842338) |
| C-3_add_away_win_expectancy | 43 | 743.794357 | +3.824634 | |Δ| ≤ seed_std (7.842338) |
| C-4_add_all_three | 45 | 745.759315 | +5.789592 | |Δ| ≤ seed_std (7.842338) |

## Artifact paths

- `outputs/eda/seed_variance.md`
- `outputs/eda/weight_schedule_grid.md`
- `outputs/eda/excluded_feature_recheck.md`
- `outputs/eda/overnight_summary.md`
