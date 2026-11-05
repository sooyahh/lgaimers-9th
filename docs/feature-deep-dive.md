---
id: "project-lg-aimers-feature-deep-dive"
type: "project"
status: "draft"
aliases: ["LG_Aimers_Feature_Deep_Dive"]
topics: ["competition", "features"]
sources: ["[[source-lg-aimers-hackathon-introduction]]"]
created: "2026-08-18"
updated: "2026-08-19"
---
# LG Aimers 피처 심층 분석

## 목차
1. 이 대회가 예측하는 것 (라벨 재정의)
2. 그룹 1: 경기 기본 정보
3. 그룹 2: 카운트 및 스코어 상황
4. 그룹 3: 주자 및 상황 중요도 (세이버메트릭스)
5. 그룹 4: 선수 및 팀 정보
6. 그룹 5: `asof_*` 히스토리 피처 (핵심, 19개)
7. 그룹 6: `trackman_history.csv` 참고 컬럼 (구종 추적 용어)
8. 피처 ↔ EDA/FE 우선순위 매핑
9. 주의사항 (leakage / cold-start / distribution shift)
10. 다음 실행 단계

---

## 0. 근거 등급 표기

혼동 가능성이 큰 주장에는 아래 태그를 붙였습니다. 태그가 없는 문장은 대체로 서술·정리 목적입니다.

| 태그 | 의미 |
|---|---|
| `OFFICIAL` | 공식 데이터 설명서(`open/data_description.md`) 또는 대회 규칙에서 직접 확인 |
| `DATA_VERIFIED` | 실제 CSV 검사로 확인 (train 1,475,092행 / trackman 1,793,078행 기준) |
| `DOMAIN_INTERPRETATION` | 야구·세이버메트릭스 일반 지식에 근거한 해석. 이번 데이터가 직접 증명하지 않음 |
| `COMPETITION_HYPOTHESIS` | 아직 성능 실험으로 검증되지 않은 대회 가설 |

공식 설명서는 저장소 밖(`../open/data_description.md`)에 있습니다.

---

## 1. 이 대회가 예측하는 것 (라벨 재정의)

### 1-1. 공식 정의 `OFFICIAL`

공식 설명서(§2)가 `control_success`에 대해 규정하는 내용은 아래가 전부입니다.

| 항목 | 공식 문구 |
|---|---|
| 값 | "`1`은 제구 성공, `0`은 제구 실패를 의미합니다." |
| 산출 근거 | "`control_success`는 **운영 기준에 따라 산출된** 제구 성공 여부입니다." |
| 입력 제한 | "Target 산출에 사용되는 현재 투구의 사후 정보는 입력 피처로 제공되지 않습니다." |

**세부 판정 로직(코스 경계, "벗어남"의 거리 기준, 방향 판정 각도)은 공식 문서에 공개되어 있지 않습니다.** `train.csv`의 `control_success`는 `{0, 1}` 결측 0으로 확인됩니다 `DATA_VERIFIED`.

### 1-2. 실패 조건 3가지 — 출처 미확인 `COMPETITION_HYPOTHESIS`

> ⚠️ 아래 표는 **공식 데이터 설명서에서 확인되지 않았습니다.** 별도 출처(대회 페이지·공지·Q&A 등)로 확인되기 전까지 검증되지 않은 가설로 취급합니다. 공식 규격으로 인용하지 마십시오.

| 조건 (가설) | 판정 (가설) |
|---|---|
| 스트라이크존 가운데 부근으로 들어간 공 | 제구 실패 (0) |
| 스트라이크존에서 크게 벗어난 공 | 제구 실패 (0) |
| 의도한 방향과 반대로 들어간 공 | 제구 실패 (0) |
| 위 3가지에 해당하지 않는 유효 투구 | 제구 성공 (1) |

이 가설이 나온 유일한 근거는 공식 `asof_*` 피처명이 "가운데 또는 위험 코스 비율"(`middle_rate`)과 "의도 반대성 투구 비율"(`reverse_rate`)을 노출한다는 점입니다. 즉 **피처 정의에서 역추정한 것이며, target 판정 규칙으로 공식 확인된 바는 없습니다.** 특히 "포수 요구 방향"이라는 표현은 공식 문서 어디에도 등장하지 않으므로 사용하지 않습니다.

### 1-3. 해석 `DOMAIN_INTERPRETATION`

위 가설이 맞다면 `control_success`는 "스트라이크냐 볼이냐"가 아니라 "투수가 의도한 코스대로, 위험하지 않게 던졌는가"에 가깝습니다. 다만 이는 §1-2 가설에 의존하는 해석이며, 공식적으로는 "운영 기준"의 내용이 비공개라는 사실만 확정적입니다.

판정 threshold가 비공개이므로, EDA에서 `asof_pitcher_middle_rate` / `asof_pitcher_reverse_rate`와 `control_success`의 상관관계를 직접 확인해 라벨의 실제 분포를 역추정하는 작업이 필요합니다 (섹션 8 참고).

---

## 2. 그룹 1: 경기 기본 정보

| 컬럼               | 의미           | 실전 포인트                                                                                                                                                 |
| ---------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `row_id`         | 샘플 ID        | 모델 입력에서 반드시 제외 (baseline script도 drop함)                                                                                                                |
| `season`         | 시즌 연도        | train=2019~2024, test=2025 → **연도가 겹치지 않는 완전한 미래 예측 문제**. season을 그대로 범주형으로 넣으면 2025는 unseen category가 되어 위험. season별 트렌드(리그 전체 제구 성공률 변화)가 있는지 EDA 필수 |
| `game_month`     | 경기 월         | 계절 효과(초반 vs 후반 시즌 컨디션) 가능성                                                                                                                             |
| `game_dayofweek` | 요일 (월=0~일=6) | 큰 신호 기대는 낮음, 참고용                                                                                                                                       |
| `inning`         | 이닝           | 후반 이닝일수록 투수 피로 누적 → 제구 난이도 상승 가설                                                                                                                       |
| `top_bottom`     | 초(T)/말(B)    | 원정/홈 공격 여부와 연동. `score_diff_home`, `score_diff_pitcher_team`과 관계 확인 필요                                                                                 |
| `game_type`      | 경기 유형 코드     | 실측값은 `{'F', 'R'}` 2종 `DATA_VERIFIED`. 각 코드의 **의미는 공식 미기재**이며, 정규시즌/포스트시즌 대응은 확인되지 않은 추정                                                                 |

---

## 3. 그룹 2: 카운트 및 스코어 상황

| 컬럼                                  | 의미                   | 실전 포인트                                                                                                     |
| ----------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------- |
| `balls_before`                      | 투구 직전 볼카운트 (0~3) `DATA_VERIFIED`     | 볼카운트가 불리할수록(3볼) 투수가 스트라이크존 가운데로 몰아 던질 유인 증가 → middle_rate·control_success와 상관 예상 `DOMAIN_INTERPRETATION`. 피처 중요도 순위는 미검증 `COMPETITION_HYPOTHESIS` |
| `strikes_before`                    | 투구 직전 스트라이크카운트 (0~2) `DATA_VERIFIED` | 2스트라이크에서는 반대로 유인구(존 밖)를 던질 유인 → reverse/ball 쪽 상관 가능 `DOMAIN_INTERPRETATION`                                                       |
| `outs_before`                       | 아웃카운트 (0~2) `DATA_VERIFIED`          | 상황 압박도 보조지표                                                                                                |
| `run_top_before` / `run_bot_before` | 초/말 공격팀 각각 점수        | **원자료.** 아래 3개 컬럼이 이 두 값의 정확한 파생이므로, 정보량 기준으로는 이쪽이 원본                                                     |
| `run_total_before`                  | 양팀 합산 점수             | `run_top + run_bot`와 **100% 일치** (불일치 0행) `DATA_VERIFIED`                                                   |
| `score_diff_home`                   | 홈팀 기준 점수차            | `run_bot − run_top`와 **100% 일치** (불일치 0행) `DATA_VERIFIED`                                                   |
| `score_diff_pitcher_team`           | **투수 소속팀 기준** 점수차    | `top_bottom=='T'`면 `score_diff_home`와, `'B'`면 `−score_diff_home`와 **각각 100% 일치** `DATA_VERIFIED`. 투수 관점 부호로 정렬돼 있어 모델링에 더 직접적일 가능성은 미검증 `COMPETITION_HYPOTHESIS` |

→ **정보 중복 주의** `DATA_VERIFIED`: `run_total_before`, `score_diff_home`, `score_diff_pitcher_team` 세 컬럼은 모두 `(top_bottom, run_top_before, run_bot_before)`의 **정확한 결정함수**입니다. 새로운 정보를 추가하지 않으므로, 이 블록에서 실제 자유도는 3개(초 점수·말 점수·초말 구분)뿐입니다.

→ `balls_before` × `strikes_before` 교차(카운트 조합, 예: 3-0, 0-2)를 파생 피처로 만드는 안 `COMPETITION_HYPOTHESIS` — 볼카운트가 투수의 투구 전략을 좌우한다는 야구 통념에 근거하지만 `DOMAIN_INTERPRETATION`, 이 데이터에서의 성능 이득은 검증되지 않았습니다.

---

## 4. 그룹 3: 주자 및 상황 중요도 (세이버메트릭스)

| 컬럼                                            | 의미                                                                                                        | 배경지식                                                 |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `runner_on_1b/2b/3b`                          | 각 루 주자 유무 (0/1)                                                                                           | 원자료                                                  |
| `num_runners_on`                              | 출루 주자 수                                                                                                   | 세 indicator의 합과 **100% 일치** (불일치 0행) `DATA_VERIFIED` |
| `base_state`                                  | 주자 상황 코드 (`___`, `1__`, `_2_`, `__3`, `12_`, `1_3`, `_23`, `123`) `OFFICIAL`                              | 8가지 범주형. 세 indicator로 **100% 재구성 가능**(불일치 0행)이므로 정보 중복 `DATA_VERIFIED` |
| `home_win_expectancy` / `away_win_expectancy` | 투구 직전 홈/원정 팀의 **기대 승률**, 0~100 범위 `OFFICIAL`. 산출 공식·출처는 공식 미기재                                            | 두 값의 합은 **항상 정확히 100은 아님** — 정확히 100인 행 96.03%, 불일치 58,607행(3.97%). 합의 전체 범위는 99.9~100.1, 합의 고유값은 5종 `DATA_VERIFIED`. 사실상 상보 관계지만 항등식으로 가정하지 말 것 |
| `li`                                          | 투구 직전 **상황 중요도 지표**. "값이 클수록 경기 흐름에 미치는 영향이 큰 상황" `OFFICIAL`                                             | 실측 분포: min 0 / 중앙값 0.80 / 평균 0.9818 / p99 4.37 / max 10.83, `li==1.0`인 행은 0.34% `DATA_VERIFIED`. 세이버메트릭스 Leverage Index의 "평균=1.0" 관례는 공식 미기재 `DOMAIN_INTERPRETATION` |

→ `li`가 높은 상황(위기 상황)에서 `control_success` 비율이 낮아지는지는 검증해볼 만한 가설입니다 `COMPETITION_HYPOTHESIS`. 심리적 압박이 제구를 떨어뜨린다는 연결은 야구 통념이며 이 데이터가 증명한 바 없습니다 `DOMAIN_INTERPRETATION`.

---

## 5. 그룹 4: 선수 및 팀 정보

| 컬럼                                   | 의미         | 실전 포인트                                                                                                               |
| ------------------------------------ | ---------- | -------------------------------------------------------------------------------------------------------------------- |
| `pitcher_id` / `batter_id`           | 익명화된 선수 ID `OFFICIAL` | 실측 고유값 **792 / 830** `DATA_VERIFIED` — 수백 수준이며 "매우 높은 카디널리티"는 아님. 원-핫·target encoding·`asof_*` 대체 중 무엇이 유리한지는 **미검증** `COMPETITION_HYPOTHESIS`. target encoding을 쓸 경우 train 내부 k-fold로 leakage를 차단해야 함 |
| `pitcher_hand` / `batter_hand`       | 좌우 유형 **코드** `OFFICIAL` | 실측 정수 `{1, 2}` `DATA_VERIFIED`. **어느 코드가 좌/우인지는 공식 미기재**이므로 좌투·우타 등으로 단정하지 말 것. 다만 `pitcher_hand == batter_hand`(동일 손 여부) 파생은 매핑을 몰라도 성립 |
| `pitcher_team_id` / `batter_team_id` | 소속팀 ID `OFFICIAL`    | 실측 고유값 **13종**(12~25, 24 결번) `DATA_VERIFIED` → 원-핫 적용 가능. 팀별 특성이 신호가 되는지는 미검증 `COMPETITION_HYPOTHESIS` |

---

## 6. 그룹 5: `asof_*` 히스토리 피처 (핵심, 19개)

**공통 원칙** `OFFICIAL`: `asof_*`는 "해당 행의 투구 직전까지 확인 가능한 과거 기록으로 사전 계산된 피처"이며 "현재 투구 이후에 확정되는 정보는 사용하지 않았습니다". 규칙상 test에서도 그대로 사용 가능합니다 (설명서 5번 항목).

실측 `asof_*` 컬럼 수는 **19개**입니다 `DATA_VERIFIED` (공식 설명서 표도 19행).

**결측 현황** `DATA_VERIFIED` — train 전체 49컬럼 중 결측이 있는 것은 `asof_*` 뿐입니다.

| 블록 | 결측 행 | 비율 |
|---|---:|---:|
| pitcher rate 5개 + mix rate 3개 | 792 | 0.0537% |
| batter rate 2개 | 830 | 0.0563% |
| **prev1/3/5 계열 6개** | **29,185** | **1.9785%** |
| `asof_pitcher_n`, `asof_pitcher_pitchmix_n`, `asof_batter_n` | 0 | 0% |

공식은 "표본 수가 0인 경우 **일부** rate 컬럼은 결측값**일 수 있습니다**"로만 규정하지만, 실데이터에서는 `n == 0 ⟺ rate 결측`이 **예외 0건으로 정확히** 성립합니다 `DATA_VERIFIED`.

### 6-1. 투수 전체 이력

| 컬럼 | 의미 | 라벨과의 관계 (가설) |
|---|---|---|
| `asof_pitcher_n` | 투구 직전까지 해당 투수의 누적 투구 수 `OFFICIAL` | 실측 0~15,449. **`asof_pitcher_pitchmix_n`과 전 행 완전 동일** (§6-4 참고) `DATA_VERIFIED`. 표본이 작으면 rate류가 노이즈 많음 (cold-start) |
| `asof_pitcher_success_rate` | 투구 직전까지 해당 투수의 제구 성공률 `OFFICIAL` | 투수의 "기본 실력" 대리 변수 `DOMAIN_INTERPRETATION`. 단일 피처 중요도 순위는 **미검증** `COMPETITION_HYPOTHESIS` |
| `asof_pitcher_reverse_rate` | 투구 직전까지 해당 투수의 **의도 반대성 투구 비율** `OFFICIAL` | 공식 문구는 "의도 반대성"이며 **포수 요구 방향은 공식에 없음**. §1-2 실패조건과의 대응은 가설 `COMPETITION_HYPOTHESIS` |
| `asof_pitcher_middle_rate` | 투구 직전까지 해당 투수의 **가운데 또는 위험 코스 비율** `OFFICIAL` | §1-2 실패조건과의 대응은 가설 `COMPETITION_HYPOTHESIS`. 값이 높으면 제구가 불안정하다는 해석은 야구 통념 `DOMAIN_INTERPRETATION` |
| `asof_pitcher_ball_rate` | 투구 직전까지 해당 투수의 볼성 결과 비율 `OFFICIAL` | 카운트 판정 축. `control_success`와의 관계는 상관계수 확인 필요 |
| `asof_pitcher_strike_rate` | 투구 직전까지 해당 투수의 스트라이크성 결과 비율 `OFFICIAL` | `ball_rate + strike_rate`는 **1이 아님** — 평균 0.814, 정확히 1인 행 0.26%, 항상 ≤1 `DATA_VERIFIED`. 두 축 밖의 제3범주가 존재 |

### 6-2. 투수 최근 폼 (단기 트렌드)

| 컬럼 | 의미 |
|---|---|
| `asof_pitcher_prev1/3/5_game_success_rate` | 직전 1/3/5경기 제구 성공률 `OFFICIAL` |
| `asof_pitcher_prev1/3/5_game_middle_rate` | 직전 1/3/5경기 가운데 또는 위험 코스 비율 `OFFICIAL` |

**결측 주의** `DATA_VERIFIED`: 이 6개 컬럼은 **29,185행(1.9785%)** 결측이며, `asof_pitcher_n == 0`(792행)과 **무관한 별도 블록**입니다. 즉 cold-start 규모는 `n == 0`보다 이쪽이 훨씬 큽니다.

→ 전체 누적(`asof_pitcher_success_rate`)과 최근 폼(`prev1/3/5`)의 **차이(diff)**를 파생 피처로 만들어 "최근에 제구가 무너지고 있는 투수"를 잡는 안 `COMPETITION_HYPOTHESIS` — 컨디션 저하 신호라는 해석은 야구 통념 `DOMAIN_INTERPRETATION`.

### 6-3. 타자 이력

| 컬럼 | 의미 |
|---|---|
| `asof_batter_n` | 투구 직전까지 해당 타자가 상대한 누적 투구 수 `OFFICIAL` |
| `asof_batter_success_rate` | 그 타자가 상대한 투구의 제구 성공률 `OFFICIAL` |
| `asof_batter_middle_rate` | 그 타자가 상대한 투구의 가운데 또는 위험 코스 비율 `OFFICIAL` |

→ 이 값이 높으면 "투수들이 이 타자에게 유독 실투가 잦다"고 읽는 해석 `DOMAIN_INTERPRETATION` — 방심·압박 등 원인 추정은 이 데이터로 확인되지 않습니다.

### 6-4. 투수 구종 사용 이력

| 컬럼 | 의미 |
|---|---|
| `asof_pitcher_pitchmix_n` | 투구 직전까지 해당 투수의 구종 사용 이력 표본 수 `OFFICIAL` |
| `asof_pitcher_fastball_rate` / `breaking_rate` / `offspeed_rate` | 각 구종군(fastball/breaking/offspeed) 사용 비율 `OFFICIAL` |

**중복 주의** `DATA_VERIFIED`: 공식 정의는 다르지만, `asof_pitcher_pitchmix_n`은 `asof_pitcher_n`과 **1,475,092행 전부에서 값이 완전히 동일**합니다(불일치 0행). 실질적으로 중복 컬럼이므로 둘을 독립 정보로 다루지 마십시오.

**합=1** `DATA_VERIFIED`: 세 rate의 합은 관측 행에서 min 0.999999 / max 1.000001로, 부동소수점 오차 범위 내에서 1입니다. 다만 공식 설명서가 합=1을 규정하지는 않습니다. 또한 Trackman의 `other` 구종군(1.24%)에 대응하는 컬럼이 train에 없으므로, **이 3개 rate는 `other`를 제외한 3군 기준**입니다.

→ 구종별 **제구 성공률**은 제공되지 않으며, **`trackman_history.csv`로도 만들 수 없습니다** `DATA_VERIFIED`. Trackman 30개 컬럼에는 `control_success`도, 코스·판정·결과를 담은 컬럼도 존재하지 않습니다(`control`/`success`/`course`/`location`/`plate`/`call`/`result` 패턴 검색 결과 0건). Trackman으로 파생 가능한 것은 구속·회전수·무브먼트·릴리스 좌표 등 **물리 프로필 요약뿐**입니다 (§7 참고).

---

## 7. 그룹 6: `trackman_history.csv` 참고 컬럼 (구종 추적 용어)

공식은 이 파일을 "메인 학습/평가 데이터와 1:1로 결합하는 정답 테이블이 아니라, 참가자가 과거 이력 기반 피처를 만들 때 참고할 수 있는 로그 데이터"로 규정합니다 `OFFICIAL`. 2019~2024만 포함되며 2025 Trackman은 제공되지 않습니다 `OFFICIAL`.

### 7-1. 컬럼

| 컬럼 | 의미 |
|---|---|
| `tagged_pitch_type` / `auto_pitch_type` | 수동/자동 분류 구종명 `OFFICIAL`. **표기 오염 있음** — `tagged` 17종에 `ChangeUp`/`Changeup`, `SInker`/`Sinker` 혼재, `auto` 11종에 `Undefined` 포함 `DATA_VERIFIED`. 사용 시 정규화 필요 |
| `pitch_type_group` | 구종군 단순화 `OFFICIAL`. 실측 4군 — fastball 51.93% / breaking 28.60% / offspeed 18.23% / **other 1.24%** `DATA_VERIFIED`. train의 `asof_*_rate`는 **`other`가 없는 3군**이므로 어휘는 같아도 체계가 완전히 일치하지는 않음 |
| `rel_speed` | 릴리스 시점 구속 `OFFICIAL`. **단위 공식 미기재** (§7-2) |
| `spin_rate` | 투구 회전수 `OFFICIAL`. **단위 공식 미기재**. 실측 중앙값 2,223 `DATA_VERIFIED` (rpm과 정합). 회전수가 무브먼트/체감속도에 영향을 준다는 설명은 야구 통념 `DOMAIN_INTERPRETATION` |
| `induced_vert_break` (IVB) | 유도 수직 무브먼트 `OFFICIAL`. 중력 제외·마그누스 효과라는 설명과 "안 떨어지는 공(라이징 패스트볼)" 해석은 `DOMAIN_INTERPRETATION` |
| `horz_break` | 수평 무브먼트 `OFFICIAL` |
| `extension` | 릴리스 확장 거리 `OFFICIAL`. 실측에 **음수(min −0.387)** 존재 `DATA_VERIFIED` → 이상치 처리 필요. 체감 구속 상승 해석은 `DOMAIN_INTERPRETATION` |
| `rel_height` / `rel_side` | 릴리스 높이 / 좌우 위치 `OFFICIAL`. 팔 각도·사이드암 추정은 `DOMAIN_INTERPRETATION` |
| `zone_speed` | 홈플레이트 근처 구속 `OFFICIAL`. **`zone_speed < rel_speed`가 99.9999%** 성립(중앙값 감속 11.39, 비율 0.9165) `DATA_VERIFIED` |

### 7-2. 단위 — 공식 미기재 vs 실측 추론

공식 데이터 설명서에는 위 물리량 **8개 전부에 대해 단위가 명시되어 있지 않습니다.**

`rel_speed` 실측 분포 `DATA_VERIFIED`: 중앙값 137.41, fastball 군 중앙값 143.01, p1 111.61, p99 152.43, max 160.66. 120~165 구간에 93.58%, 70~105 구간에 0.16%가 분포합니다. 이는 **km/h와 정합하고 mph와는 불정합**하며, `extension`(중앙값 1.76)·`rel_height`(중앙값 1.76)가 미터계와 정합하는 점이 이를 뒷받침합니다. 다만 이는 **분포 기반 추론이지 공식 확인이 아닙니다.**

### 7-3. 메인 데이터와의 연결 — 규정과 데이터를 분리할 것

- **규정** `OFFICIAL`: "참가자는 이 파일을 이용해 과거 투구 특성, 구종 특성, **투수 단위 요약값** 등 추가 피처를 만들 수 있습니다." 즉 투수 단위 요약 생성 자체는 허용됩니다.
- **데이터** `DATA_VERIFIED`: 그러나 현재 데이터에서는 **투수/타자 단위로 메인 테이블에 붙일 조인 키가 없습니다.**

| 키 | 메인 고유값 | Trackman 고유값 | 직접 교집합 |
|---|---:|---:|---:|
| `pitcher_id` ↔ `pitcher_trackman_id` | 792 | 906 | **0** |
| `batter_id` ↔ `batter_trackman_id` | 830 | 913 | **0** |

ID 체계가 서로 다릅니다(메인 5자리 20700~24633, Trackman 7~8자리). `top_bottom`도 메인 `{B, T}` vs Trackman `{Bottom, Top}`, hand도 메인 `{1, 2}` vs Trackman `{Left, Right}`로 인코딩이 다릅니다 `DATA_VERIFIED`. 공식 문서에 두 ID 체계를 잇는 매핑은 제공되지 않습니다.

→ 따라서 **"투수별 평균 spin_rate" 같은 개인 단위 피처는 현재 데이터로 붙일 수 없습니다.** 실행 가능한 것은 리그 전체·시즌·구종군 등 **ID가 필요 없는 단위의 집계**입니다. 또한 현재 투구의 Trackman 측정값과 2025 Trackman은 사용 금지입니다 `OFFICIAL`.

---

## 8. 피처 ↔ EDA/FE 우선순위 매핑

| 우선순위 | 피처군                                                        | 이유                   |
| ---- | ---------------------------------------------------------- | -------------------- |
| 필수   | `asof_pitcher_success_rate`, `middle_rate`, `reverse_rate` | 라벨 정의와 구조적으로 가장 가까움  |
| 필수   | `balls_before`, `strikes_before` (교차 포함)                   | 투수 전략을 직접 결정하는 변수    |
| 권장   | `li`, `home_win_expectancy`                                | 압박 상황 신호, 상호작용 피처 후보 |
| 권장   | `prev1/3/5_game_*` vs 전체 누적 diff                           | 최근 폼 저하 탐지           |
| 권장   | `pitcher_hand == batter_hand`                              | 매치업 파생               |
| 선택   | trackman 기반 **ID 불필요 단위** 요약 통계                            | 투수 단위는 조인 키 부재로 불가 (§7-3) |
| 선택   | `game_type`, `game_dayofweek`                              | 낮은 신호 기대, 최후순위       |

이 표의 우선순위는 전부 **미검증 가설**입니다 `COMPETITION_HYPOTHESIS`.

---

## 9. 주의사항

- **Data leakage** `OFFICIAL`: `asof_*`는 사용 가능하지만, 직접 만드는 파생 피처는 반드시 "해당 행 시점 이전" 데이터만 사용. train 전체를 미리 groupby하면 안 됨 (미래 정보 유입).
- **Test leakage 규칙** `OFFICIAL`: 공식 §5가 금지하는 것 — `test.csv` 내부 행을 이용한 선수별/팀별/월별 누적 통계, 빈도·분포 통계, target encoding, 행 순서 기반 rolling·expanding, 평가 데이터 전체를 보고 만든 사후 보정값. 공식 §6은 추가로 현재 투구의 위치·판정·결과·구종·**Trackman 측정값**, 그리고 **2025년 Trackman**을 금지.
- **Cold-start** `DATA_VERIFIED`: 두 축을 구분할 것.
  - `asof_pitcher_n == 0` → 해당 pitcher rate 5개가 결측. 단 **792행(0.0537%)**으로 규모가 매우 작음. batter는 830행, pitchmix는 792행이며 모두 `n == 0 ⟺ 결측`이 예외 0건으로 성립.
  - **prev1/3/5 계열 6개는 29,185행(1.9785%) 결측**이며 `n == 0`과 무관. cold-start 대응의 실제 주 대상은 이쪽.
  - 트리 계열은 결측을 자체 처리하지만, 별도 "표본수 낮음" 플래그 추가는 검토 가치가 있음 `COMPETITION_HYPOTHESIS`.
- **CV-LB gap / 분포 이동**: train은 2019~2024, test는 2025 `DATA_VERIFIED`(test 샘플 `season` 실측 = 2025) → **season 기준 temporal validation**(저장소 기준: season hold-out primary + **expanding-window** secondary, `reports/validation_policy.md`)으로 검증. 랜덤 K-fold는 시간 순서를 무시하므로 낙관적인 CV 점수를 낼 위험이 큼 `DOMAIN_INTERPRETATION`.
- **Metric**: 공식 배포 자료인 `data_description.md`에는 **평가 지표의 정의가 포함되어 있지 않습니다.** 현재 저장소의 metric 구현은 **Brier Skill Score**입니다(`configs/baseline.yaml`의 `task.metric` — clip at 0, scale 100000 / `src/metrics.py`). 저장소 구현이 곧 공식 규칙은 아니므로 **공식 metric 출처는 별도 확인이 필요합니다.** 확률 보정(calibration)이 중요하다는 점은 Brier 계열 지표의 성질 `DOMAIN_INTERPRETATION`.
- **Reproducibility**: `pitcher_id`/`batter_id` 기반 target encoding을 쓸 경우 seed 고정 + fold별 재현성 확인 필수.
- **정보 중복** `DATA_VERIFIED`: `num_runners_on`·`base_state`는 runner indicator의, `run_total_before`·`score_diff_home`·`score_diff_pitcher_team`은 `(top_bottom, run_top, run_bot)`의, `asof_pitcher_pitchmix_n`은 `asof_pitcher_n`의 정확한 중복/파생입니다.

---

## 10. 다음 실행 단계 (제안)

1. `asof_pitcher_middle_rate`, `asof_pitcher_reverse_rate`와 `control_success`의 실제 상관관계를 EDA로 확인 → 라벨 정의 가설 검증
2. `balls_before` × `strikes_before` 조합별 `control_success` 평균 비교 (카운트 압박 효과 시각화)
3. season별(2019~2024) `control_success` 비율 추세 확인 → 2025(test)로의 외삽 리스크 점검
4. cold-start 표본(`asof_pitcher_n` 작은 그룹, prev1/3/5 결측 그룹) 분리 확인

이 문서는 Feature Dictionary이며 모델 비교 계획서가 아닙니다. 모델·validation 관련 결정은 `reports/validation_policy.md`와 `outputs/experiments.csv`를 따릅니다.
