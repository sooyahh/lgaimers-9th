# Excluded feature recheck (Task C)

- generated_at: `2026-08-07T15:55:11`
- question: are `li` / `home_win_expectancy` / `away_win_expectancy` pre-pitch or post-target?

## Safety check (`data_description.md`)

| feature | pre-pitch? | quote |
|---|---|---|
| `li` | yes | 투구 직전 상황 중요도 지표입니다. |
| `home_win_expectancy` | yes | 투구 직전 경기 상황에서 홈 팀의 기대 승률입니다. |
| `away_win_expectancy` | yes | 투구 직전 경기 상황에서 원정 팀의 기대 승률입니다. |

- conclusion: all three are labeled **투구 직전** (pre-pitch situation metrics).
- therefore: **no skip** — proceed with 4 add-back experiments.
- baseline features remain EXP-005 L set (42); variants add the named columns.
- seed / weights: seed=42, B-1 current schedule.
- **results only — no adopt/reject**

## Results

| variant | added | n_feat | raw BSS | Δ vs C-0 | official BSS | Brier | elapsed_s |
|---|---|---:|---:|---:|---:|---:|---:|
| C-0_baseline_42 | `—` | 42 | 739.969723 | +0.000000 | 739.969723 | 0.247958 | 148.4 |
| C-1_add_li | `li` | 43 | 743.857943 | +3.888220 | 743.857943 | 0.247949 | 139.2 |
| C-2_add_home_win_expectancy | `home_win_expectancy` | 43 | 743.458972 | +3.489249 | 743.458972 | 0.247950 | 141.7 |
| C-3_add_away_win_expectancy | `away_win_expectancy` | 43 | 743.794357 | +3.824634 | 743.794357 | 0.247949 | 141.7 |
| C-4_add_all_three | `li,home_win_expectancy,away_win_expectancy` | 45 | 745.759315 | +5.789592 | 745.759315 | 0.247944 | 136.2 |
