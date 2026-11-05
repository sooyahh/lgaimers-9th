# FEATURE-STABILITY-AUDIT

- purpose: measure 2019–2024 stability of priority features vs `control_success`
- data: `data/raw/train.csv` only (no test-row aggregates, no retraining)
- metric of association: Pearson and Spearman vs target; not a model-selection metric
- classification: conservative heuristic, not a hypothesis test
- train rows: 1,475,092

## 1. Executive summary

Target rate falls from **0.5647** (2019) to **0.4861** (2024), with a clear 2022→2023 drop. Several asof rate features that look usable in 2019–2022 change magnitude or **sign** in 2023–2024. Count/context features are more compositionally stable than pitcher-history rates. Prev-game asof features are dominated by cold-start NA at the start of a pitcher-season, not by a strong stable signal.

Numeric-feature class counts:

- **STABLE**: 2 — `asof_pitcher_prev3_game_success_rate`, `asof_pitcher_prev5_game_success_rate`
- **WEAK_STABLE**: 13 — `asof_pitcher_middle_rate`, `asof_pitcher_prev5_game_middle_rate`, `asof_pitcher_breaking_rate`, `asof_pitcher_offspeed_rate`, `balls_before`, `strikes_before`, `outs_before`, `inning`, `num_runners_on`, `score_diff_pitcher_team`, `li`, `home_win_expectancy`, `away_win_expectancy`
- **UNSTABLE**: 6 — `asof_pitcher_success_rate`, `asof_pitcher_reverse_rate`, `asof_pitcher_prev1_game_success_rate`, `asof_batter_n`, `asof_batter_success_rate`, `asof_batter_middle_rate`
- **SIGN_FLIP**: 5 — `asof_pitcher_n`, `asof_pitcher_ball_rate`, `asof_pitcher_strike_rate`, `asof_pitcher_pitchmix_n`, `asof_pitcher_fastball_rate`
- **SPARSE/COLD_START**: 2 — `asof_pitcher_prev1_game_middle_rate`, `asof_pitcher_prev3_game_middle_rate`
- **UNKNOWN**: 0

All requested priority columns are present in `train.csv`.

## 2. Season-wise target change

Overall train target mean: **0.523766**

| season | n | target_mean | delta_vs_prev |
|---|---|---|---|
| 2019 | 237,413 | 0.564670 | — |
| 2020 | 244,087 | 0.532712 | -0.031958 |
| 2021 | 247,088 | 0.532762 | +0.000050 |
| 2022 | 247,472 | 0.528920 | -0.003841 |
| 2023 | 245,525 | 0.499957 | -0.028963 |
| 2024 | 253,507 | 0.486105 | -0.013852 |

The 2022→2023 drop is the largest adjacent change and is the main 2025-generalization risk already flagged by VAL-AUDIT-001 / gap diagnosis.

## 3. Feature stability table

Labels are conservative. `mean_abs_pearson` is the mean of |seasonal Pearson|. Categorical rows use mean rank-Spearman of category target rates instead.

| feature | family | kind | class | mean|r| or rank ρ | pearson min | pearson max | max missing | note |
|---|---|---|---|---|---|---|---|---|
| `asof_batter_middle_rate` | batter_history | numeric | UNSTABLE | 0.0206 | -0.0396 | 0.0061 | 0.0017 | direction mostly consistent but magnitude moves a lot (pearson range=0.0457, mean|r|=0.0206); 2023 |Pearson| is materially weaker than 2022 |
| `asof_batter_n` | batter_history | numeric | UNSTABLE | 0.0246 | -0.0540 | 0.0071 | 0.0000 | direction mostly consistent but magnitude moves a lot (pearson range=0.0611, mean|r|=0.0246); quantile-bin target-rate slope also changes sign across seasons; 2023 |Pearson| is materially weaker than 2022 |
| `asof_batter_success_rate` | batter_history | numeric | UNSTABLE | 0.0433 | 0.0003 | 0.0893 | 0.0017 | direction mostly consistent but magnitude moves a lot (pearson range=0.0889, mean|r|=0.0433); 2023 |Pearson| is materially weaker than 2022 |
| `away_win_expectancy` | game_context | numeric | WEAK_STABLE | 0.0018 | -0.0039 | 0.0018 | 0.0000 | association is small in every season; direction is not strongly identified |
| `balls_before` | game_context | categorical | STABLE | 0.8400 | missing | missing | 0.0000 | category target-rate ranks are consistent (mean Spearman vs first season=0.840) |
| `balls_before` | game_context | numeric | WEAK_STABLE | 0.0119 | -0.0156 | -0.0037 | 0.0000 | association is small in every season; direction is not strongly identified |
| `base_state` | game_context | categorical | WEAK_STABLE | 0.5714 | missing | missing | 0.0000 | ranks somewhat consistent (mean Spearman=0.571); max |2023-2022| rate shift=0.0444 |
| `batter_hand` | game_context | categorical | UNKNOWN | missing | missing | missing | 0.0000 | too few overlapping categories to judge rank stability |
| `home_win_expectancy` | game_context | numeric | WEAK_STABLE | 0.0018 | -0.0018 | 0.0039 | 0.0000 | association is small in every season; direction is not strongly identified |
| `inning` | game_context | categorical | WEAK_STABLE | 0.4699 | missing | missing | 0.0000 | ranks somewhat consistent (mean Spearman=0.470); max |2023-2022| rate shift=0.0421 |
| `inning` | game_context | numeric | WEAK_STABLE | 0.0127 | -0.0166 | -0.0085 | 0.0000 | direction is consistent among non-near-zero seasons but effect is small |
| `li` | game_context | numeric | WEAK_STABLE | 0.0044 | 0.0030 | 0.0062 | 0.0000 | association is small in every season; direction is not strongly identified |
| `num_runners_on` | game_context | categorical | UNSTABLE | 0.3200 | missing | missing | 0.0000 | category target-rate ranking is not stable (mean Spearman=0.320) |
| `num_runners_on` | game_context | numeric | WEAK_STABLE | 0.0028 | -0.0035 | 0.0064 | 0.0000 | association is small in every season; direction is not strongly identified |
| `outs_before` | game_context | categorical | UNSTABLE | -0.3000 | missing | missing | 0.0000 | category target-rate ranking is not stable (mean Spearman=-0.300) |
| `outs_before` | game_context | numeric | WEAK_STABLE | 0.0043 | -0.0009 | 0.0061 | 0.0000 | association is small in every season; direction is not strongly identified |
| `pitcher_hand` | game_context | categorical | UNKNOWN | missing | missing | missing | 0.0000 | too few overlapping categories to judge rank stability |
| `score_diff_pitcher_team` | game_context | numeric | WEAK_STABLE | 0.0059 | 0.0021 | 0.0112 | 0.0000 | association is small in every season; direction is not strongly identified |
| `strikes_before` | game_context | categorical | UNSTABLE | 0.3000 | missing | missing | 0.0000 | category target-rate ranking is not stable (mean Spearman=0.300) |
| `strikes_before` | game_context | numeric | WEAK_STABLE | 0.0063 | -0.0187 | 0.0040 | 0.0000 | association is small in every season; direction is not strongly identified |
| `asof_pitcher_breaking_rate` | pitch_mix | numeric | WEAK_STABLE | 0.0116 | -0.0172 | -0.0055 | 0.0015 | association is small in every season; direction is not strongly identified |
| `asof_pitcher_fastball_rate` | pitch_mix | numeric | SIGN_FLIP | 0.0104 | -0.0148 | 0.0187 | 0.0015 | Pearson sign differs across seasons (pos=1, neg=2, near0=3; range=0.0335) |
| `asof_pitcher_offspeed_rate` | pitch_mix | numeric | WEAK_STABLE | 0.0157 | -0.0044 | 0.0283 | 0.0015 | direction is consistent among non-near-zero seasons but effect is small; quantile-bin target-rate slope also changes sign across seasons |
| `asof_pitcher_pitchmix_n` | pitch_mix | numeric | SIGN_FLIP | 0.0146 | -0.0208 | 0.0269 | 0.0000 | Pearson sign differs across seasons (pos=3, neg=1, near0=2; range=0.0476) |
| `asof_pitcher_ball_rate` | pitcher_history | numeric | SIGN_FLIP | 0.0227 | -0.0474 | 0.0143 | 0.0015 | Pearson sign differs across seasons (pos=1, neg=3, near0=2; range=0.0616) |
| `asof_pitcher_middle_rate` | pitcher_history | numeric | WEAK_STABLE | 0.0222 | -0.0388 | -0.0068 | 0.0015 | direction is consistent among non-near-zero seasons but effect is small; 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_n` | pitcher_history | numeric | SIGN_FLIP | 0.0146 | -0.0208 | 0.0269 | 0.0000 | Pearson sign differs across seasons (pos=3, neg=1, near0=2; range=0.0476) |
| `asof_pitcher_prev1_game_middle_rate` | pitcher_history | numeric | SPARSE/COLD_START | 0.0125 | -0.0190 | -0.0058 | 0.0517 | missing/zero-n is material and the observed target association is weak |
| `asof_pitcher_prev1_game_success_rate` | pitcher_history | numeric | UNSTABLE | 0.0539 | 0.0365 | 0.0770 | 0.0517 | direction mostly consistent but magnitude moves a lot (pearson range=0.0405, mean|r|=0.0539); 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_prev3_game_middle_rate` | pitcher_history | numeric | SPARSE/COLD_START | 0.0180 | -0.0342 | -0.0076 | 0.0517 | missing/zero-n is material and the observed target association is weak |
| `asof_pitcher_prev3_game_success_rate` | pitcher_history | numeric | STABLE | 0.0676 | 0.0508 | 0.0885 | 0.0517 | direction is consistent across seasons with a non-trivial mean |r|; cold-start/missing mass is non-trivial even if the observed-row association is usable; 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_prev5_game_middle_rate` | pitcher_history | numeric | WEAK_STABLE | 0.0200 | -0.0391 | -0.0085 | 0.0517 | direction is consistent among non-near-zero seasons but effect is small; cold-start/missing mass is non-trivial even if the observed-row association is usable |
| `asof_pitcher_prev5_game_success_rate` | pitcher_history | numeric | STABLE | 0.0707 | 0.0527 | 0.0908 | 0.0517 | direction is consistent across seasons with a non-trivial mean |r|; cold-start/missing mass is non-trivial even if the observed-row association is usable; 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_reverse_rate` | pitcher_history | numeric | UNSTABLE | 0.0683 | -0.0946 | -0.0499 | 0.0015 | direction mostly consistent but magnitude moves a lot (pearson range=0.0447, mean|r|=0.0683); 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_strike_rate` | pitcher_history | numeric | SIGN_FLIP | 0.0190 | -0.0239 | 0.0354 | 0.0015 | Pearson sign differs across seasons (pos=3, neg=2, near0=1; range=0.0593) |
| `asof_pitcher_success_rate` | pitcher_history | numeric | UNSTABLE | 0.0738 | 0.0527 | 0.0986 | 0.0015 | direction mostly consistent but magnitude moves a lot (pearson range=0.0459, mean|r|=0.0738); 2023 |Pearson| is materially weaker than 2022 |

Seasonal Pearson for numeric priority features (season=0 is pooled 2019–2024, not used for classification):

| feature | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | pooled |
|---|---|---|---|---|---|---|---|
| `asof_pitcher_n` | -0.0008 | 0.0124 | -0.0011 | -0.0208 | 0.0269 | 0.0254 | -0.0116 |
| `asof_pitcher_success_rate` | 0.0944 | 0.0612 | 0.0784 | 0.0986 | 0.0527 | 0.0576 | 0.0843 |
| `asof_pitcher_reverse_rate` | -0.0781 | -0.0590 | -0.0739 | -0.0946 | -0.0499 | -0.0545 | -0.0795 |
| `asof_pitcher_middle_rate` | -0.0388 | -0.0082 | -0.0284 | -0.0378 | -0.0068 | -0.0129 | -0.0361 |
| `asof_pitcher_ball_rate` | -0.0095 | -0.0229 | 0.0011 | 0.0143 | -0.0474 | -0.0413 | -0.0177 |
| `asof_pitcher_strike_rate` | -0.0020 | 0.0107 | -0.0119 | -0.0239 | 0.0354 | 0.0300 | 0.0039 |
| `asof_pitcher_prev1_game_success_rate` | 0.0648 | 0.0401 | 0.0646 | 0.0770 | 0.0405 | 0.0365 | 0.0617 |
| `asof_pitcher_prev3_game_success_rate` | 0.0817 | 0.0516 | 0.0821 | 0.0885 | 0.0508 | 0.0508 | 0.0778 |
| `asof_pitcher_prev5_game_success_rate` | 0.0831 | 0.0527 | 0.0851 | 0.0908 | 0.0551 | 0.0574 | 0.0816 |
| `asof_pitcher_prev1_game_middle_rate` | -0.0128 | -0.0058 | -0.0190 | -0.0139 | -0.0116 | -0.0120 | -0.0195 |
| `asof_pitcher_prev3_game_middle_rate` | -0.0342 | -0.0090 | -0.0285 | -0.0130 | -0.0076 | -0.0155 | -0.0283 |
| `asof_pitcher_prev5_game_middle_rate` | -0.0391 | -0.0093 | -0.0319 | -0.0150 | -0.0085 | -0.0163 | -0.0320 |
| `asof_batter_n` | -0.0227 | -0.0153 | -0.0421 | -0.0540 | 0.0066 | 0.0071 | -0.0363 |
| `asof_batter_success_rate` | 0.0690 | 0.0281 | 0.0663 | 0.0893 | 0.0003 | 0.0067 | 0.0589 |
| `asof_batter_middle_rate` | -0.0339 | -0.0026 | -0.0335 | -0.0396 | 0.0061 | -0.0078 | -0.0338 |
| `asof_pitcher_pitchmix_n` | -0.0008 | 0.0124 | -0.0011 | -0.0208 | 0.0269 | 0.0254 | -0.0116 |
| `asof_pitcher_fastball_rate` | -0.0051 | -0.0148 | 0.0043 | 0.0187 | -0.0082 | -0.0112 | -0.0002 |
| `asof_pitcher_breaking_rate` | -0.0115 | -0.0055 | -0.0098 | -0.0130 | -0.0123 | -0.0172 | -0.0136 |
| `asof_pitcher_offspeed_rate` | 0.0163 | 0.0197 | 0.0058 | -0.0044 | 0.0194 | 0.0283 | 0.0138 |
| `balls_before` | -0.0037 | -0.0123 | -0.0125 | -0.0125 | -0.0156 | -0.0147 | -0.0121 |
| `strikes_before` | -0.0187 | -0.0015 | -0.0054 | -0.0052 | -0.0032 | 0.0040 | -0.0051 |
| `outs_before` | -0.0009 | 0.0032 | 0.0048 | 0.0049 | 0.0061 | 0.0056 | 0.0039 |
| `inning` | -0.0125 | -0.0166 | -0.0130 | -0.0129 | -0.0085 | -0.0129 | -0.0130 |
| `num_runners_on` | -0.0035 | -0.0018 | 0.0020 | 0.0064 | -0.0027 | -0.0002 | -0.0004 |
| `score_diff_pitcher_team` | 0.0112 | 0.0067 | 0.0084 | 0.0021 | 0.0023 | 0.0046 | 0.0058 |
| `li` | 0.0034 | 0.0035 | 0.0062 | 0.0045 | 0.0030 | 0.0060 | 0.0043 |
| `home_win_expectancy` | 0.0012 | 0.0039 | 0.0020 | -0.0014 | 0.0003 | -0.0018 | 0.0005 |
| `away_win_expectancy` | -0.0013 | -0.0039 | -0.0020 | 0.0014 | -0.0003 | 0.0018 | -0.0005 |

## 4. Most stable features (TOP)

| feature | class | mean|r| | pearson min | pearson max | reason |
|---|---|---|---|---|---|
| `asof_pitcher_prev5_game_success_rate` | STABLE | 0.0707 | 0.0527 | 0.0908 | direction is consistent across seasons with a non-trivial mean |r|; cold-start/missing mass is non-trivial even if the observed-row association is usable; 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_prev3_game_success_rate` | STABLE | 0.0676 | 0.0508 | 0.0885 | direction is consistent across seasons with a non-trivial mean |r|; cold-start/missing mass is non-trivial even if the observed-row association is usable; 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_middle_rate` | WEAK_STABLE | 0.0222 | -0.0388 | -0.0068 | direction is consistent among non-near-zero seasons but effect is small; 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_prev5_game_middle_rate` | WEAK_STABLE | 0.0200 | -0.0391 | -0.0085 | direction is consistent among non-near-zero seasons but effect is small; cold-start/missing mass is non-trivial even if the observed-row association is usable |
| `asof_pitcher_offspeed_rate` | WEAK_STABLE | 0.0157 | -0.0044 | 0.0283 | direction is consistent among non-near-zero seasons but effect is small; quantile-bin target-rate slope also changes sign across seasons |
| `inning` | WEAK_STABLE | 0.0127 | -0.0166 | -0.0085 | direction is consistent among non-near-zero seasons but effect is small |
| `balls_before` | WEAK_STABLE | 0.0119 | -0.0156 | -0.0037 | association is small in every season; direction is not strongly identified |
| `asof_pitcher_breaking_rate` | WEAK_STABLE | 0.0116 | -0.0172 | -0.0055 | association is small in every season; direction is not strongly identified |

Context features with stable category ranking (from categorical analysis):

| feature | class | rank Spearman | max 22→23 |rate| shift | reason |
|---|---|---|---|---|
| `balls_before` | STABLE | 0.840 | 0.0317 | category target-rate ranks are consistent (mean Spearman vs first season=0.840) |
| `base_state` | WEAK_STABLE | 0.571 | 0.0444 | ranks somewhat consistent (mean Spearman=0.571); max |2023-2022| rate shift=0.0444 |
| `inning` | WEAK_STABLE | 0.470 | 0.0421 | ranks somewhat consistent (mean Spearman=0.470); max |2023-2022| rate shift=0.0421 |

## 5. Most unstable features (TOP)

| feature | class | pearson range | min | max | reason |
|---|---|---|---|---|---|
| `asof_batter_success_rate` | UNSTABLE | 0.0889 | 0.0003 | 0.0893 | direction mostly consistent but magnitude moves a lot (pearson range=0.0889, mean|r|=0.0433); 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_ball_rate` | SIGN_FLIP | 0.0616 | -0.0474 | 0.0143 | Pearson sign differs across seasons (pos=1, neg=3, near0=2; range=0.0616) |
| `asof_batter_n` | UNSTABLE | 0.0611 | -0.0540 | 0.0071 | direction mostly consistent but magnitude moves a lot (pearson range=0.0611, mean|r|=0.0246); quantile-bin target-rate slope also changes sign across seasons; 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_strike_rate` | SIGN_FLIP | 0.0593 | -0.0239 | 0.0354 | Pearson sign differs across seasons (pos=3, neg=2, near0=1; range=0.0593) |
| `asof_pitcher_n` | SIGN_FLIP | 0.0476 | -0.0208 | 0.0269 | Pearson sign differs across seasons (pos=3, neg=1, near0=2; range=0.0476) |
| `asof_pitcher_pitchmix_n` | SIGN_FLIP | 0.0476 | -0.0208 | 0.0269 | Pearson sign differs across seasons (pos=3, neg=1, near0=2; range=0.0476) |
| `asof_pitcher_success_rate` | UNSTABLE | 0.0459 | 0.0527 | 0.0986 | direction mostly consistent but magnitude moves a lot (pearson range=0.0459, mean|r|=0.0738); 2023 |Pearson| is materially weaker than 2022 |
| `asof_batter_middle_rate` | UNSTABLE | 0.0457 | -0.0396 | 0.0061 | direction mostly consistent but magnitude moves a lot (pearson range=0.0457, mean|r|=0.0206); 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_reverse_rate` | UNSTABLE | 0.0447 | -0.0946 | -0.0499 | direction mostly consistent but magnitude moves a lot (pearson range=0.0447, mean|r|=0.0683); 2023 |Pearson| is materially weaker than 2022 |
| `asof_pitcher_prev1_game_success_rate` | UNSTABLE | 0.0405 | 0.0365 | 0.0770 | direction mostly consistent but magnitude moves a lot (pearson range=0.0405, mean|r|=0.0539); 2023 |Pearson| is materially weaker than 2022 |

## 6. SIGN_FLIP or sharply weakening features

- `asof_pitcher_n`: 2019:-0.0008, 2020:0.0124, 2021:-0.0011, 2022:-0.0208, 2023:0.0269, 2024:0.0254
- `asof_pitcher_ball_rate`: 2019:-0.0095, 2020:-0.0229, 2021:0.0011, 2022:0.0143, 2023:-0.0474, 2024:-0.0413
- `asof_pitcher_strike_rate`: 2019:-0.0020, 2020:0.0107, 2021:-0.0119, 2022:-0.0239, 2023:0.0354, 2024:0.0300
- `asof_pitcher_pitchmix_n`: 2019:-0.0008, 2020:0.0124, 2021:-0.0011, 2022:-0.0208, 2023:0.0269, 2024:0.0254
- `asof_pitcher_fastball_rate`: 2019:-0.0051, 2020:-0.0148, 2021:0.0043, 2022:0.0187, 2023:-0.0082, 2024:-0.0112

This matches the earlier `target_regime_check.md` observation that `asof_pitcher_ball_rate` / `asof_pitcher_strike_rate` reverse around 2022→2023.

Features whose |Pearson| is much weaker in 2023 than 2022 are noted in the class table.

## 7. Cold-start related characteristics

Share of rows with `*_n == 0` by season:

| feature | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| `asof_pitcher_n` | 0.0015 | 0.0005 | 0.0004 | 0.0004 | 0.0003 | 0.0003 |
| `asof_batter_n` | 0.0017 | 0.0004 | 0.0004 | 0.0004 | 0.0003 | 0.0003 |
| `asof_pitcher_pitchmix_n` | 0.0015 | 0.0005 | 0.0004 | 0.0004 | 0.0003 | 0.0003 |

Prev-game asof missing rates (identical across prev1/3/5 in prior integrity report):

| feature | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| `asof_pitcher_prev1_game_success_rate` | 0.0517 | 0.0158 | 0.0151 | 0.0119 | 0.0111 | 0.0144 |
| `asof_pitcher_prev3_game_success_rate` | 0.0517 | 0.0158 | 0.0151 | 0.0119 | 0.0111 | 0.0144 |
| `asof_pitcher_prev5_game_success_rate` | 0.0517 | 0.0158 | 0.0151 | 0.0119 | 0.0111 | 0.0144 |
| `asof_pitcher_prev1_game_middle_rate` | 0.0517 | 0.0158 | 0.0151 | 0.0119 | 0.0111 | 0.0144 |
| `asof_pitcher_prev3_game_middle_rate` | 0.0517 | 0.0158 | 0.0151 | 0.0119 | 0.0111 | 0.0144 |
| `asof_pitcher_prev5_game_middle_rate` | 0.0517 | 0.0158 | 0.0151 | 0.0119 | 0.0111 | 0.0144 |

Rate features (`asof_*_rate`) are missing exactly when the corresponding n is 0 (first observation of that pitcher/batter). That is a structural cold-start, not random MCAR. Prev-game features have a larger missing block (~2% overall in `data_integrity.md`) because they require prior games in the proxy order.

## 8. 2025 generalization risks

1. Target mean is still trending down through 2024; a 2025 model trained with a 2024-like base rate can be miscalibrated if 2025 continues or reverts.
2. SIGN_FLIP asof rates (ball/strike, and possibly others) will not transfer if 2025 resembles 2023–2024 rather than 2019–2022.
3. Absolute `season` is already known from D/G experiments to be a brittle 2024 crutch; this audit does not re-test that, but the target shift is the same mechanism.
4. Cold-start pitchers/batters in a new season cannot rely on prev-game asof or long asof_n. 2025 will re-introduce this mass.
5. `li` / win expectancy remain excluded from models (`feature_safety_audit.md`); their statistical association is reported here only as context, not as an FE recommendation.

## 9. Observations that can feed FE candidates

These are observations, not implementations:

- Count (`balls_before` × `strikes_before`) has a persistent level effect (3-0 / 3-2 lower success) and a roughly parallel 2022→2023 drop (`target_regime_check.md`). Count itself looks more stable than pitcher-rate history.
- Pitcher asof success/middle rates are weak but more sign-consistent than ball/strike rates; shrinking or recency-weighting them may be safer than adding more raw asof rates.
- Pitch mix rates exist and are official-asof; they may track the 2022+ league mix shift better than physical Trackman if TM cannot be player-matched.
- Cold-start indicators (`asof_pitcher_n==0`, prev-game NA flags) are already implicit in missingness; explicit flags/shrinkage could be a low-leakage FE if error analysis shows those slices fail.
- Do **not** treat SIGN_FLIP features as 2025-safe without a fold-wise check.

## 10. Items that cannot be concluded yet

- Whether instability is causal (target definition / calling / mix) vs correlational.
- Whether a 2025 test distribution looks like 2023, 2024, or a new regime (full test labels not available; local test.csv is a 5-row format sample).
- Whether WE/LI are leak-free; association strength is not a safety proof.
- Player-level Trackman physics vs these asof rates (no ID join).
- Model error concentration (see ERR-AUDIT-001); this audit is feature–target only.

## Appendix

- numeric CSV: `outputs/analysis/feature_stability_by_season.csv`
- categorical CSV: `outputs/analysis/feature_stability_categorical_by_season.csv`
- classification CSV: `outputs/analysis/feature_stability_classification.csv`
- plot: `outputs/analysis/feature_stability/target_rate_by_season.png`
- plot: `outputs/analysis/feature_stability/pearson_by_season_asof_rates.png`
- plot: `outputs/analysis/feature_stability/missing_rate_by_season.png`
- plot: `outputs/analysis/feature_stability/stability_class_overview.png`

Heuristic notes: `NEAR_ZERO=0.01` Pearson; SIGN_FLIP if ≥1 season >+0.01 and ≥1 season <−0.01; UNSTABLE if Pearson range ≥0.04 with non-trivial mean |r|. These cutoffs organize the table; they are not significance tests.

```json
{
  "n_train": 1475092,
  "missing_requested_columns": [],
  "used_columns": [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "balls_before",
    "strikes_before",
    "outs_before",
    "inning",
    "base_state",
    "num_runners_on",
    "score_diff_pitcher_team",
    "li",
    "home_win_expectancy",
    "away_win_expectancy",
    "pitcher_hand",
    "batter_hand"
  ],
  "class_counts": {
    "SIGN_FLIP": 5,
    "SPARSE/COLD_START": 2,
    "STABLE": 2,
    "UNSTABLE": 6,
    "WEAK_STABLE": 13
  }
}
```
