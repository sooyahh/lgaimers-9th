# LG Aimers Phase 2 — STATUS

> **이 문서는 현재 상태만 담는다.** 이력은 `git log`가 갖는다. 갱신할 때는 **덮어쓴다** (append 금지).
> 발견의 상세와 근거는 `reports/`로 분리한다. 여기에는 결론 한 줄과 링크만 둔다.

- 최종 갱신: **2026-08-22 (v9 — RECENCY-AUDIT-001 / WEIGHT-LAG-001 결과 및 PR #13 merge 반영)**
- 갱신 트리거: canonical 실험 완료 / LB 점수 수신 / baseline·판정규칙 변경 / 브랜치 병합
- 작성 근거: 기존 artifact(`reports/weight_lag_001.md`, `reports/recency_audit_001.md`, `outputs/analysis/weight_lag_001/*`)에서 복사. **이 문서 작성 과정에서 새로 학습한 모델은 없다.**

---

## 1. 제출 현황

| 제출 | 일시 | feature | 학습 창 | 로컬 2024 raw BSS | **Public LB** |
|---|---|---|---|---:|---:|
| `EXP-004_submit.zip` | 2026-08-07 04:24 | 44 (ID 포함) | 2019–2023 | 578.18 | **406.9505006093** |
| `LB-PROBE-L42-2019-2023-001` | 2026-08-19 | 42 (L, ID 제거) | 2019–2023 | 739.97 | **676.2128868651** |
| `EXP-005_submit.zip` | 2026-08-07 07:38 | 42 (L, ID 제거) | 2019–**2024** | 739.97 | **877.6038710153** ← **현재 대표** |

> 대표 제출은 DACON이 **최고 점수로 자동 유지**하므로 EXP-005(877.60)가 대표다. 수동 확인 불필요.

| 항목 | 값 |
|---|---|
| 현재 순위 | **656 / 1090** (0점자 7명 제외) |
| 수료 기준 549.51 | **통과** |
| Phase 3 진출 | **100등 이내** — 컷 1102.18, **+224.57 필요** |
| 1위 | 1196.94 |
| 제출 사용 | **3회** (하루 5회 한도, 잔여 2~3주) |
| 서버 추론 소요 | **2초** (10분 제한 여유 충분) |

### Phase 0 결과 — 두 축 분리 완료

$$470.65 = \underbrace{+269.26}_{\text{feature (ID 제거), 57.2\%}} + \underbrace{+201.39}_{\text{2024 추가, 42.8\%}}$$

| 항목 | 값 |
|---|---:|
| 로컬 ID-drop 효과 (2024 holdout) | +161.79 |
| LB ID-drop 효과 (2025) | **+269.26** |
| **transfer ratio** | **1.664** |

**통제 검증 완료:** EXP-005와 LB-PROBE의 번들 메타를 대조한 결과 `features_used`(42) 완전 동일,
차이는 `train_seasons` 차집합 `[2024]`와 `sample_weight_by_season`의 `2024:2.4` 항목뿐.
→ **별도 통제 probe 추가 제출 불필요.** (잔여 리스크는 `reports/lb_anchor_analysis_001.md` §2.5)

**둘 다 크다.** 그리고 **일반화 지향 변화(ID 제거)는 LB에서 증폭된다.**
2025의 unseen 선수 비율이 2024(투수 19.86%)보다 높다는 가설과 일치한다.

> ⚠️ `1.664`는 **일반화 지향 변화 1건의 전이율**이다. capacity·추가 feature·앙상블 같은
> 일반 정확도 개선은 1:1에 가까울 가능성이 높다. 단일 계수로 모든 로컬 개선을 환산하지 말 것.

**`EXP-004_submit.zip`의 학습 창은 2019–2023으로 확인됨** (검증 완료):
`model/feature_meta.json` 키 `numeric_medians_primary_train_2019_2023`, `model_filename: catboost_primary.joblib`,
번들 모델이 `outputs/models/EXP-004/catboost_primary.joblib`와 동일 파일(29,384,442 B / 08-06 18:06),
`summary.json`의 `sample_weight_by_season`에 **2024 키 없음**. → 두 앵커의 학습 창 차이는 실재한다.

> 과거 스냅샷(v3): 이 세 Public LB가 `outputs/experiments.csv`에 없었고 `public_lb: not_submitted`였다.
> **현재(PR #9):** 아래 세 값이 로그에 기록됨 — EXP-004 `406.9505006093`, LB-PROBE `676.2128868651`, EXP-005 `877.6038710153`. 상세는 §4.

상세: [`reports/lb_anchor_analysis_001.md`](../reports/lb_anchor_analysis_001.md), [`reports/leaderboard_position_001.md`](../reports/leaderboard_position_001.md)

---

## 2. 현재 baseline / 판정 규칙

| 항목 | 값 |
|---|---|
| 제출 라인 | **L42** (ID 제거) + `game_type` 유지 + weight B-1(2024:2.4) + full-train 2019–2024 + seed 42 |
| 실험 비교 baseline | **EXP-004** (44 feat, ID 포함, train 2019–2023) |
| Primary fold | train 2019–2023 → valid 2024 |
| Secondary | expanding 2019-21→2022 / 2019-22→2023 / 2019-23→2024 |
| Metric | raw BSS (official clip 전), `src/metrics.py` |
| Seed | 42 |

> ⚠️ **실험 baseline과 현재 대표 제출 라인이 다르다** (44 feat vs L42, 로컬 2024 기준 **162점 차이**).
> line을 섞지 말 것:
> - **EXP-004** — 44 feat (ID 포함), train **2019–2023**, Public LB **406.9505006093** (실험 비교 baseline)
> - **LB-PROBE-L42-2019-2023-001** — L42 (ID 제거), train **2019–2023**, Public LB **676.2128868651** (feature-set 축 분리)
> - **EXP-005** — L42, train **2019–2024** (+ 2024 weight 2.4), Public LB **877.6038710153** (현재 대표 제출)
> 상세: [`reports/experiment_line_mismatch_001.md`](../reports/experiment_line_mismatch_001.md)

**판정 규율 (유지)**
- 사전 선언된 규칙은 **결과를 본 뒤 소급 변경하지 않는다.**
- validation policy를 바꾸려면 새 policy를 날짜·commit과 함께 freeze하고 **그 이후 실험부터** 적용한다.
- 과거 REJECT를 뒤집고 싶으면 재판정이 아니라 **새 experiment_id로 재실행**한다.

> TM-003A 사전 규칙에 `specifically, no 2023-fold degradation` / `2023 fold or the worst fold clearly degrades`가
> **명시적으로 존재함이 원문에서 확인됐다** (`exp/tm-003a-expected-ivb:reports/next_experiment_decision.md` §7.11).
> 결과를 본 뒤 2023 조건을 제거해 과거 판정을 다시 매기는 것은 허용되지 않는다.

---

## 3. 브랜치 상태

- remote: `https://github.com/sooyahh/lgaimers.git` (PR은 `LGaimers-9th` org 경유)
- `main` = `866c0a5` (`Merge pull request #14 from LGaimers-9th/claude/lgaimers-gitignore-local-fl4plt`)
- 현재 체크아웃: `docs/status-after-weight-lag` (이 문서 갱신 중)

**이미 main에 병합**

| 항목 | 상태 |
|---|---|
| TM-003A | PR **#10** merge. canonical CV 완료, Team **ADOPT**. 구현·report·`experiments.csv` 행이 main에 있음 |
| TM-003A pre-run decision | 본문은 `#10`의 `reports/next_experiment_decision.md` §7에 포함. 별도 PR **#4** (`docs/tm-003a-pre-run-decision`)는 **superseded** |
| LB-PROBE / Public LB 로그 | PR **#9**. EXP-004 / EXP-005 / LB-PROBE `public_lb` 기록 |
| feature-deep-dive | PR **#8** merge. `docs/feature-deep-dive.md`가 main에 있음 |
| RECENCY-AUDIT-001 | PR **#12** merge. `reports/recency_audit_001.md` — results-only(ADOPT/REJECT 판정 없음), full-history expanding window + uniform weight가 training-window policy candidate로 정리됨 |
| WEIGHT-LAG-001 | PR **#13** merge. `reports/weight_lag_001.md` Team decision — **REJECT**(scope: L42 / CatBoost seed 42 / full-history / geometric season-lag weight, `m∈{1.5,2,3,4}`). uniform weight(m=1.0) 유지 |

**VAL-SHIFT-2023-001** — runner + tests는 `audit/val-shift-2023-001`에 **commit됨**. smoke artifact는 untracked. **canonical run은 아직 미실행** (Phase C 보류와 동일).

**미병합 브랜치**

| 브랜치 | 내용 | 상태 |
|---|---|---|
| `audit/val-shift-2023-001` | VAL-SHIFT runner + tests | runner commit됨 / canonical 미실행 / main에 없음 |
| `exp/fe-001-count-code` | FE-001 구현 + REJECT 기록 | main에 없음 |
| `docs/status-after-weight-lag` | 이 STATUS 갱신 (RECENCY-AUDIT-001 / WEIGHT-LAG-001 반영) | 현재 branch, 미병합 |

---

## 4. 다음 우선순위

**Phase 0 — ✅ 완료 (2026-08-19)**

| 작업 | 상태 |
|---|---|
| L42 + train 2019–2023 제출 (`LB-PROBE-L42-2019-2023-001`) | ✅ LB **676.21**, 두 축 분리 완료 |
| LB 점수 3건을 `outputs/experiments.csv`에 기록 | ✅ 완료 |
| `LB-PROBE` 중복 행 정리 | ✅ 완료 |

**로그 정리 내역 (2026-08-19)**

- `LB-PROBE-L42-2019-2023-001` 중복 행 2건 중 `17:46:17` 제거, `17:48:30` 유지
  (두 행은 `date`를 제외하고 완전 동일. `summary.json`의 `generated_at: 17:48:25`와 정합)
- `public_lb_score` / `cv_lb_gap` 기록 완료 (`cv_lb_gap = public_lb_score − cv_score`)

| experiment_id | public_lb_score | cv_lb_gap |
|---|---:|---:|
| EXP-004 | 406.9505006093 | −83.4167248699 |
| LB-PROBE-L42-2019-2023-001 | **676.2128868651** | **−63.7568361919** |
| EXP-005 | 877.6038710153 | +137.6341479583 |

- 백업: `outputs/experiments.csv.bak_20260819T1748_pre_lbprobe_fix` (수정 전 원본, 미추적 파일)
- 검증: 나머지 23행 **바이트 단위 동일** 확인, 행 수 25 → 24

**제출 zip 파일명** — 결함 아님. DACON 제출 파일명 **30자 제한** 때문에
`LB-PROBE-L42-2019-2023-001_submit.zip` → `L42_2019_2023_probe.zip.zip`으로 사용자가 변경.
sha256(`543af2fd…`)과 크기(247,602 B)가 `summary.json`과 **일치 확인**. 경위는 notes에 기록함.

**Phase 0 실행 순서** (분업: Cursor 구현·실행 → Claude Code / Codex 독립 감사)

1. EXP-004 / EXP-005의 학습창·feature·params를 machine-readable하게 잠금
2. LB `406.95`, `877.60` 를 `outputs/experiments.csv`에 복구
3. 새 EXP 생성 — `L42 + train 2019–2023`, **EXP-004와 가능한 한 동일 조건**
4. submission validation → 제출
5. EXP-004 / NEW / EXP-005 세 모델 비교

**사전 해석 규칙 (제출 전 freeze, 결과 보고 수정 금지)** — `S` = 새 제출의 LB

`(877.60 − 406.95) = (S − 406.95) + (877.60 − S)` 로 가산 분해된다.
로컬 ID-drop 효과 +161.79가 1:1 전이되면 `S ≈ 569`.

| `S` | 읽는 법 |
|---|---|
| ≈ 406.95 | ID 제거의 2025 효과 거의 없음 → 상승분 대부분이 2024 추가 |
| ≈ 569 | 전이율 ≈ 1 → 2024 추가 ≈ +309 |
| > 700 | ID 제거 효과 증폭 (2025 unseen ID 비율이 더 높을 가능성) |
| ≈ 877.60 | ID 제거가 핵심 → 2024 추가 효과는 작음 |

**Phase 0 완료.** `RECENCY-AUDIT-001` / `WEIGHT-LAG-001`도 완료됐다 (아래 참고). 둘 다 results-only 진단이며,
`outputs/experiments.csv`에는 기록되지 않는다 (진단 실험 convention) — 대표 제출 라인은 바뀌지 않는다.

**Phase A — 지금부터 (목표 400~500등)**

> Phase 0 결과에 따라 **일반화 지향 변화를 일반 정확도 개선보다 앞에 둔다.**
> 100등까지 LB +224.57 → ratio 1.66이면 로컬 **+135**, ratio 1.0이면 로컬 **+225**.

| 작업 | 근거 |
|---|---|
| **RECENCY-AUDIT-001** — ✅ 완료 (2026-08-19, PR #12). 결정: 오래된 history를 버리지 않음 (full-history가 policy candidate) | 상세: 아래 "RECENCY 상태" |
| **WEIGHT-LAG-001** — ✅ 완료 (2026-08-19, PR #13). geometric season-lag weight `m∈{1.5,2,3,4}` **REJECT**, uniform weight(m=1.0) 유지 | 상세: 아래 "WEIGHT-LAG-001 — 실험 판정" |
| **2023 shift 원인 분석** (`VAL-SHIFT-2023-001` 상태 감사) | WEIGHT-LAG-001 결과로 이 축의 계획 우선순위가 fine `m` sweep보다 높아짐 (아래 "다음 투자 우선순위" 참고). **NOT YET CANONICAL** — runner는 `audit/val-shift-2023-001`에 commit됨, canonical run 미실행 |
| **C-4** (`li` + `home/away_win_expectancy`) 반영 | 측정 완료 **+5.79**, 미반영 |
| **capacity sweep** (`lr 0.01~0.03 × iters 1000~8000` + early stopping) | Optuna가 `iters ≤ 500`만 탐색 |
| seed 5개 ensemble | 이득 = `E[Var_seed(p)]`, 2024 기준 +11.2 |

### WEIGHT-LAG-001 — 실험 판정

**REJECT**

범위 (이 판정이 적용되는 정확한 조건, 그 이상으로 확대 해석하지 않음):

- L42 / CatBoost (`iterations=300, learning_rate=0.05, depth=6, Logloss`) / seed=42
- full-history expanding validation (RECENCY-AUDIT-001 `B_expanding` windows)
- geometric validation-relative season-lag weighting, `w(lag) = m ** (1 - (lag - 1) / (L - 1))`, `m ∈ {1.5, 2.0, 3.0, 4.0}`

Keep: full-history + uniform weight (`m = 1.0`).

근거 (canonical artifact, `outputs/analysis/weight_lag_001/{results,contrasts}.csv`):

- `m=1.0` CONTROL은 RECENCY-AUDIT-001 `B_expanding`과 3-fold 전부 exact reproduction (gate `PASS`).
- 모든 `m>1` treatment의 3-fold mean Δ raw BSS가 음수 (`-44.99`, `-40.10`, `-37.58`, `-54.99`).
- 2023 fold는 모든 `m>1`에서 크게 악화 (`Δ2023`: `-122.98` ~ `-100.26`).
- 2024는 `m=2.0`(`+5.92`), `m=3.0`(`+6.07`)에서 소폭 개선하지만 2023 악화를 상쇄하지 못하고, `m`에 대한 단조 개선도 없다.

**이 REJECT는 recency weighting 전체에 대한 판정이 아니다.** 위 scope(이 geometric formulation, 이 `m` 범위)에 한정된다.
상세: [`reports/weight_lag_001.md`](../reports/weight_lag_001.md)

### 다음 투자 우선순위 — 계획 판단

이 절은 실험 판정이 아니라 **자원 배분 판단**이다.

현재 증거에서는 동일 geometric lag-weight formulation을 더 세밀한 `m` grid로 추가 탐색하는 기대값이 낮다
(모든 treatment mean Δ 음수, `m` 증가에 따른 단조 개선 없음). 따라서 fine sweep보다 **2023 validation shift 원인 분석**을
우선한다 — recent-season up-weighting이 오히려 2023을 악화시켰다는 사실은, 2023 shift에 대한 단순 recency 설명을
**약화**시킨다 (기각을 확정하는 것은 아니다).

다음 우선순위: `VAL-SHIFT-2023-001` 상태 감사 → canonical 필요성 판단 → 2023 feature/context/regime shift 분석.
`VAL-SHIFT-2023-001`은 아직 canonical run이 실행되지 않았다 (**NOT YET CANONICAL / 상태 감사 필요**).

### RECENCY 상태 (RECENCY-AUDIT-001)

현재 확정된 정책 후보: **full-history expanding training window 유지 + uniform weighting(m=1.0) 유지.**
results-only 진단이며 ADOPT/REJECT 판정은 없다 — `B−C`(old-history 포함 vs 미포함)가 2022/2023/2024 모두 양수라
오래된 시즌을 제거할 근거가 없다는 것뿐이다. RECENCY-AUDIT-001 결과를 "모든 recency 아이디어의 부정"으로 확대
해석하지 않는다 — WEIGHT-LAG-001(가중치 축)과는 별개 실험이다. 이 산출물은 정책 **확정**이 아니라
**CatBoost L42 canonical의 1차 후보**다 (TM-003A·다른 backend·앙상블에서 최적 창이 달라질 수 있음).

> ⚠️ RECENCY-AUDIT-001은 **uniform weight** 조건에서만 학습 창을 비교했다. 결과를
> "현행 L42 대표 제출 라인(weight B-1, train 2019–2024)에서 2019를 빼면 이만큼 바뀐다"로 읽으면 안 된다 —
> "uniform weighting 조건에서의 학습 창 비교"일 뿐이다.

상세: [`reports/recency_audit_001.md`](../reports/recency_audit_001.md)

**Phase A 내 순서** (전이율 기대치 순)

1. **RECENCY-AUDIT-001 → WEIGHT-LAG-001** — ✅ 완료. 결정: full-history + uniform weight(m=1.0) 유지 (위 두 subsection 참고).
2. **2023 shift 원인 분석** (`VAL-SHIFT-2023-001` 상태 감사 → canonical 필요성 판단 → 2023 feature/context/regime shift 분석) —
   WEIGHT-LAG-001 이후 계획 판단으로 fine `m` sweep보다 우선순위가 높아짐. **NOT YET CANONICAL**, 아직 실행되지 않음.
3. **C-4** (`li` + win_expectancy) — 측정 완료 +5.79, 학습 1회. 일반 정확도(ratio ≈ 1 예상).
4. **capacity sweep** — 일반 정확도(ratio ≈ 1 예상).
5. **seed ensemble** — 분산 감소. 최종 feature set 확정 후 1회만.

> ⚠️ **2024 weight를 LB로 직접 튜닝하지 않는다.** Public = Private이므로 `2.4/4.0/6.0 제출` 식 탐색은
> Public LB를 validation set으로 쓰는 것이며 대회 원칙과 충돌한다. 반드시 로컬 lag-relative 규칙으로 고른다.

**Phase B — 남은 기간 (목표 +150) — 기대 전이율이 가장 높은 구간**

> Trackman 물리량과 선수 표현은 **선수 ID에 의존하지 않는 일반화 지향** 신호다.
> Phase 0에서 관측된 증폭(ratio 1.66)이 이 계열에도 적용될 가능성이 있다.

| 작업 | 근거 |
|---|---|
| **Trackman hand map 식별 → feature family** | 353MB 중 파생 feature 1개만 사용 중 |
| 선수 OOF target encoding + shrinkage | 2024 fold 투수 unseen 19.86% |
| 파생 feature 확장 | 42개는 해당 구간 표준 이하 |

**Phase C — 보류** (현 순위에서 +0점)

`SEED-ROBUSTNESS-001`, TM-002A 재평가, TM-003A L42 transfer test 단독 실행.
→ policy freeze 문서만 Phase 3 코드 검증 대비로 유지.
(`VAL-SHIFT-2023-001`은 WEIGHT-LAG-001 이후 계획 판단으로 Phase A 우선순위로 상향 — 위 "Phase A 내 순서" 2번 참고)

상세: [`reports/unused_signal_inventory_001.md`](../reports/unused_signal_inventory_001.md)

---

## 5. 재시도 금지 (측정 완료, 효과 없음 또는 음수)

| 시도 | 결과 | 출처 |
|---|---|---|
| season weight 스케줄 조정 (2019–2023 구간) | 현행 B-1 최고, 나머지 −4 ~ −20 | `outputs/eda/weight_schedule_grid.md` |
| Optuna 튜닝 (`iters 200–500`) | seed 평균 기준 +2.5 | `outputs/eda/seed_ensemble.md` |
| asof-relative 변환 (M) | 단독 +8.5, L42와 결합(N) 시 **−24.8** | `outputs/eda/unseen_id_experiments.md` |
| `game_type` 제거 | 2024 **−113.7** (2023은 +982) | 동일 |
| CatBoost × LightGBM 블렌드 | 50/50 = 666.3 < CatBoost 단독 740.0 | `outputs/eda/model_transfer_audit_001.md` |
| feature 1개 제거 (FE-001A / FE-002A) | 음수 또는 노이즈 내 | `reports/fe_001a_*`, `fe_002a_*` |
| `count_code` 범주형 (FE-001) | mean −3.24, 2024 −16.9 | `reports/fe_001_count_code.md` (미병합 브랜치) |
| prev1 shrinkage k=500 (FE-002) | prev1 슬라이스 −25.6 | `reports/fe_002_prev1_shrinkage.md` |
| `tm_expected_rel_speed` (TM-002A) | mean −1.7 | `reports/tm_002a_expected_velocity.md` |
| geometric season-lag weighting, `m∈{1.5,2,3,4}` (WEIGHT-LAG-001, full-history 유지) | 3-fold mean Δ 전부 음수 (−37.6~−55.0), 2023 −100~−123 | `reports/weight_lag_001.md` |

> ⚠️ weight 스케줄 결론은 **무효가 아니다.** EXP-004는 실제로 weighting으로 2024 validation에서 EXP-003 대비 +15.9를 얻었다.
> 정확한 한계는 이것이다 — **2024 데이터를 2025 예측에서 얼마나 가중할지는 검증된 적이 없다**
> (B-1 실험은 2024를 학습에서 뺀 상태에서 2019–2023 구간만 조정했다). `WEIGHT-LAG-001`(geometric lag-weight,
> `m∈{1.5,2,3,4}`)은 이 특정 formulation에 한해 REJECT됐다 — recency weighting 일반에 대한 결론은 아니다
> (§4 "WEIGHT-LAG-001 — 실험 판정" 참고).

---

## 6. 미해결

1. **일반 정확도 개선의 전이율** — 일반화 지향 변화는 1.664로 측정됐으나, capacity·feature 추가·앙상블의 전이율은 미측정. Phase A의 첫 제출로 추정 가능.
2. **상위 100명의 차별화 요인** — 1등→100등 구간 밀도가 0.957 점/등으로 그 아래(0.37~0.43)의 2.4배. 무언가 다른 것을 갖고 있음. Trackman이 유력 후보이나 미확인.
3. **ID-drop × 2024-추가 상호작용** — 네 번째 칸(44 feat + 2019–2024) 미제출. 두 효과가 모두 커서 상호작용도 0이 아닐 수 있으나 시급하지 않음.
4. 팀 구성 상세 (인원, 역할 분담, 팀 누적 제출 횟수) — 현재 모든 커밋 author가 `sooyah` 단독.
5. Phase 2 정확한 마감일.
