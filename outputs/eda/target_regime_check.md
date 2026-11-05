# Target Regime Check (2022→2023 hypothesis inputs)

- hypothesis under inspection: target construction may have changed between 2022 and 2023
- **no adopt/reject judgment in this document — results only**
- train source: `data\raw\train.csv`

## 1. Year-wise correlation: `control_success` vs asof pitcher rates

Pairwise Pearson / Spearman within each season (rows with NA in either side dropped).

| season | feature | n | pearson | spearman |
|---:|---|---:|---:|---:|
| 2019 | `asof_pitcher_middle_rate` | 237,058 | -0.038774 | -0.041590 |
| 2019 | `asof_pitcher_ball_rate` | 237,058 | -0.009539 | -0.016404 |
| 2019 | `asof_pitcher_strike_rate` | 237,058 | -0.002039 | -0.001528 |
| 2020 | `asof_pitcher_middle_rate` | 243,977 | -0.008178 | -0.014265 |
| 2020 | `asof_pitcher_ball_rate` | 243,977 | -0.022883 | -0.025488 |
| 2020 | `asof_pitcher_strike_rate` | 243,977 | 0.010665 | 0.011901 |
| 2021 | `asof_pitcher_middle_rate` | 246,993 | -0.028389 | -0.024982 |
| 2021 | `asof_pitcher_ball_rate` | 246,993 | 0.001084 | -0.007451 |
| 2021 | `asof_pitcher_strike_rate` | 246,993 | -0.011870 | -0.003616 |
| 2022 | `asof_pitcher_middle_rate` | 247,384 | -0.037832 | -0.042588 |
| 2022 | `asof_pitcher_ball_rate` | 247,384 | 0.014252 | 0.008091 |
| 2022 | `asof_pitcher_strike_rate` | 247,384 | -0.023918 | -0.021202 |
| 2023 | `asof_pitcher_middle_rate` | 245,462 | -0.006822 | -0.010303 |
| 2023 | `asof_pitcher_ball_rate` | 245,462 | -0.047391 | -0.055655 |
| 2023 | `asof_pitcher_strike_rate` | 245,462 | 0.035371 | 0.042298 |
| 2024 | `asof_pitcher_middle_rate` | 253,426 | -0.012940 | -0.010991 |
| 2024 | `asof_pitcher_ball_rate` | 253,426 | -0.041287 | -0.046793 |
| 2024 | `asof_pitcher_strike_rate` | 253,426 | 0.030032 | 0.034091 |

### Wide view (Pearson)

| feature | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|---:|---:|---:|
| `asof_pitcher_middle_rate` | -0.038774 | -0.008178 | -0.028389 | -0.037832 | -0.006822 | -0.012940 |
| `asof_pitcher_ball_rate` | -0.009539 | -0.022883 | 0.001084 | 0.014252 | -0.047391 | -0.041287 |
| `asof_pitcher_strike_rate` | -0.002039 | 0.010665 | -0.011870 | -0.023918 | 0.035371 | 0.030032 |

### Wide view (Spearman)

| feature | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|---:|---:|---:|
| `asof_pitcher_middle_rate` | -0.041590 | -0.014265 | -0.024982 | -0.042588 | -0.010303 | -0.010991 |
| `asof_pitcher_ball_rate` | -0.016404 | -0.025488 | -0.007451 | 0.008091 | -0.055655 | -0.046793 |
| `asof_pitcher_strike_rate` | -0.001528 | 0.011901 | -0.003616 | -0.021202 | 0.042298 | 0.034091 |

### 2022 vs 2023 Pearson delta (2023 − 2022)

| feature | pearson_2022 | pearson_2023 | delta |
|---|---:|---:|---:|
| `asof_pitcher_middle_rate` | -0.037832 | -0.006822 | 0.031010 |
| `asof_pitcher_ball_rate` | 0.014252 | -0.047391 | -0.061644 |
| `asof_pitcher_strike_rate` | -0.023918 | 0.035371 | 0.059290 |

## 2. `control_success=1` rate by `pitch_type_group` × season

**Not computable at pitch level with current keys.** `control_success` lives only in `train.csv`; `pitch_type_group` lives only in `trackman_history.csv`. There is no shared join key:

- `|pitcher_id ∩ pitcher_trackman_id| = 0` (distinct ID namespaces)
- no `game_id` / `trackman_game_id` on train
- training pipeline does not join Trackman into train (see prior Trackman-join check)

Therefore a year × `pitch_type_group` cross-tab of `P(control_success=1)` **cannot be produced** without an external mapping that is not in the released files.

### Contextual year-level margins (not a joint cross-tab)

Shown separately so distributions can still be compared side-by-side; these are **not** success rates within pitch type.

| season | train P(y=1) | tm share fastball | breaking | offspeed | other | tm n |
|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 0.564670 | 0.576816 | 0.258340 | 0.144798 | 0.020046 | 255,957 |
| 2020 | 0.532712 | 0.577675 | 0.255630 | 0.148449 | 0.018246 | 279,126 |
| 2021 | 0.532762 | 0.574557 | 0.258066 | 0.148605 | 0.018772 | 301,032 |
| 2022 | 0.528920 | 0.477326 | 0.296876 | 0.216161 | 0.009638 | 307,637 |
| 2023 | 0.499957 | 0.463624 | 0.310425 | 0.220971 | 0.004979 | 315,100 |
| 2024 | 0.486105 | 0.467782 | 0.324759 | 0.201807 | 0.005652 | 334,226 |

## 3. `P(control_success=1)` by `balls_before` × `strikes_before` (2022 vs 2023)

| balls_before | strikes_before | n_2022 | rate_2022 | n_2023 | rate_2023 | Δ rate (2023−2022) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 64,084 | 0.531599 | 63,138 | 0.502851 | -0.028748 |
| 0 | 1 | 31,161 | 0.540227 | 30,327 | 0.512052 | -0.028175 |
| 0 | 2 | 15,595 | 0.521834 | 14,526 | 0.501790 | -0.020044 |
| 1 | 0 | 25,393 | 0.528020 | 25,591 | 0.496581 | -0.031439 |
| 1 | 1 | 25,225 | 0.536174 | 25,293 | 0.505080 | -0.031094 |
| 1 | 2 | 23,740 | 0.529149 | 23,456 | 0.504306 | -0.024843 |
| 2 | 0 | 8,700 | 0.526437 | 8,774 | 0.492136 | -0.034301 |
| 2 | 1 | 13,114 | 0.527375 | 13,479 | 0.496921 | -0.030454 |
| 2 | 2 | 20,240 | 0.526779 | 20,380 | 0.497301 | -0.029477 |
| 3 | 0 | 2,809 | 0.506230 | 2,858 | 0.487754 | -0.018476 |
| 3 | 1 | 5,605 | 0.509545 | 5,807 | 0.480282 | -0.029263 |
| 3 | 2 | 11,806 | 0.501694 | 11,896 | 0.465619 | -0.036075 |

### Margin by `balls_before` only

| balls_before | rate_2022 | n_2022 | rate_2023 | n_2023 | Δ |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.532651 | 110,840 | 0.505292 | 107,991 | -0.027359 |
| 1 | 0.531147 | 74,358 | 0.501910 | 74,340 | -0.029236 |
| 2 | 0.526894 | 42,054 | 0.496118 | 42,633 | -0.030776 |
| 3 | 0.504500 | 20,220 | 0.472837 | 20,561 | -0.031664 |

### Margin by `strikes_before` only

| strikes_before | rate_2022 | n_2022 | rate_2023 | n_2023 | Δ |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.529549 | 100,986 | 0.499885 | 100,361 | -0.029663 |
| 1 | 0.534332 | 75,105 | 0.504512 | 74,906 | -0.029820 |
| 2 | 0.522338 | 71,381 | 0.495203 | 70,258 | -0.027134 |

### Overall rates

| season | n | P(control_success=1) |
|---:|---:|---:|
| 2019 | 237,413 | 0.564670 |
| 2020 | 244,087 | 0.532712 |
| 2021 | 247,088 | 0.532762 |
| 2022 | 247,472 | 0.528920 |
| 2023 | 245,525 | 0.499957 |
| 2024 | 253,507 | 0.486105 |

