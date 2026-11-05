# Feature Safety Audit

Source of truth: `open/data_description.md` only.  
No column meanings were inferred beyond the official wording.  
Scope: **48 input columns** in `train.csv` / `test.csv` (excludes target `control_success`).

Generated for post–EXP-000 / VAL-AUDIT-001 planning. No model training performed.

## Classification legend

| Label | Meaning |
|---|---|
| `safe` | Official text states the field is available at/before pitch time, or is an officially allowed as-of past feature. |
| `leakage review required` | Official text says “투구 직전” (or similar) but does **not** specify construction details enough to rule out leakage without further checks. |
| `exclude` | Not a predictive input for modeling (identifier / submission key). |
| `unknown` | Official description is insufficient to classify; **no meaning was guessed**. |

## Summary counts

| Class | Count |
|---|---:|
| safe | 44 |
| leakage review required | 3 |
| exclude | 1 |
| unknown | 0 |

## Classification table

| Column | Class | Official basis |
|---|---|---|
| `row_id` | exclude | “샘플 고유 식별자… 제출 파일과 매칭”; not a predictive feature. |
| `season` | safe | “시즌 연도”; game/context field provided as input. |
| `game_month` | safe | “경기 월”. |
| `game_dayofweek` | safe | “경기 요일”. |
| `inning` | safe | “투구 직전 이닝”. |
| `top_bottom` | safe | “초/말 구분”. |
| `game_type` | safe | “경기 유형 코드”. |
| `balls_before` | safe | “투구 직전 볼 카운트”. |
| `strikes_before` | safe | “투구 직전 스트라이크 카운트”. |
| `outs_before` | safe | “투구 직전 아웃 카운트”. |
| `run_top_before` | safe | “투구 직전 초 공격 팀의 점수”. |
| `run_bot_before` | safe | “투구 직전 말 공격 팀의 점수”. |
| `run_total_before` | safe | “투구 직전 양 팀 합산 점수”. |
| `score_diff_home` | safe | “투구 직전 홈 팀 기준 점수 차”. |
| `score_diff_pitcher_team` | safe | “투구 직전 투수 소속 팀 기준 점수 차”. |
| `runner_on_1b` | safe | “투구 직전 1루 주자 여부”. |
| `runner_on_2b` | safe | “투구 직전 2루 주자 여부”. |
| `runner_on_3b` | safe | “투구 직전 3루 주자 여부”. |
| `num_runners_on` | safe | “투구 직전 출루 주자 수”. |
| `base_state` | safe | “투구 직전 주자 상황”. |
| `home_win_expectancy` | leakage review required | Text says “투구 직전 경기 상황에서 홈 팀의 기대 승률” but does **not** document the calculation source/formula. |
| `away_win_expectancy` | leakage review required | Same as above for away team; construction unspecified. |
| `li` | leakage review required | Text says “투구 직전 상황 중요도 지표” without formula/source. |
| `pitcher_id` | safe | “투수 익명 ID”. |
| `batter_id` | safe | “타자 익명 ID”. |
| `pitcher_hand` | safe | “투수의 좌우 유형 코드”. |
| `batter_hand` | safe | “타자의 좌우 유형 코드”. |
| `pitcher_team_id` | safe | “투수 소속 팀 ID”. |
| `batter_team_id` | safe | “타자 소속 팀 ID”. |
| `asof_pitcher_n` | safe | Official as-of past feature; “투구 직전까지… 누적 투구 수”. Doc also states asof_* are usable. |
| `asof_pitcher_success_rate` | safe | Official as-of past feature; usable per data description §1 / §5. |
| `asof_pitcher_reverse_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_middle_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_ball_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_strike_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_prev1_game_success_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_prev3_game_success_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_prev5_game_success_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_prev1_game_middle_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_prev3_game_middle_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_prev5_game_middle_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_batter_n` | safe | Official as-of past feature; usable per data description. |
| `asof_batter_success_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_batter_middle_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_pitchmix_n` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_fastball_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_breaking_rate` | safe | Official as-of past feature; usable per data description. |
| `asof_pitcher_offspeed_rate` | safe | Official as-of past feature; usable per data description. |

## Explicitly out of scope (not among the 48 inputs)

| Item | Note |
|---|---|
| `control_success` | Target; present only in train. |
| Current-pitch location / call / pitch type / Trackman | Forbidden by data description §6. |
| Participant-built test-row aggregates | Forbidden by §5. |
| `trackman_history.csv` columns | Separate table; no 1:1 join key; not classified here. |

## Review follow-ups (not executed)

1. Confirm construction of `home_win_expectancy`, `away_win_expectancy`, and `li` with organizers or baseline materials if available.
2. Cold-start null handling for `asof_*` rates is allowed by the description but is a modeling choice, not a safety reclassification.
3. Do not invent meanings for undocumented code values (e.g., concrete `game_type` enums) beyond “경기 유형 코드”.
