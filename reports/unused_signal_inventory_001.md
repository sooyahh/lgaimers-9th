# UNUSED-SIGNAL-INVENTORY-001 — 미탐색·미반영 자원 목록

- 작성: 2026-08-19 (v2 — §1 표현 정정, LB 튜닝 금지 명시)
- 성격: 기존 artifact 및 코드 조사. 새로 학습한 모델 없음.
- 목적: 656/1090등에서 필요한 **+224.57**을 어디서 찾을지 후보를 정리한다.
- 결론 성격: `results_only` — 어느 항목도 ADOPT/REJECT 아님

---

## 요약

| # | 항목 | 상태 | 측정된 이득 | 비용 |
|---|---|---|---|---|
| 1 | **recency (최근 시즌 학습 포함 / weight)** | 최대 축, 로컬 미측정 | 수백 점 추정 | 중 |
| 2 | **C-4** (`li` + `home/away_win_expectancy`) | **측정 완료, 미반영** | **+5.79** | 학습 1회 |
| 3 | **모델 capacity** | 탐색 범위 밖 | 미지 | 반나절 |
| 4 | **Trackman 353MB** | 파생 feature 1개만 사용 | 미지 (단일 feature가 +19) | 1~2일 |
| 5 | 선수 정체성의 일반화 표현 | 부재 | 미지 | 반나절 |
| 6 | seed ensemble | 미반영 | +11.24 (2024) | 30분 |

---

## 1. recency — 가장 큰 축이며 로컬에서 측정된 적이 없다

`reports/lb_anchor_analysis_001.md` 참조.

- `EXP-004`(train 2019–2023) 로컬 578.18 → LB **406.95** (−171)
- `EXP-005`(train 2019–**2024**) 로컬 739.97 → LB **877.60** (+138)

두 앵커의 부호가 반대이며, 로컬이 측정하지 못하는 축은 **2024 학습 포함 여부** 하나다.

### 1.1 기존 weight 결론의 지위 (표현 정정)

`outputs/eda/weight_schedule_grid.md`의 "B-1이 최적" 결론은 **무효가 아니다.**
EXP-004는 실제로 season weighting으로 2024 validation에서 EXP-003 대비 **+15.9**를 얻었고, 그 policy가 EXP-005/006에 재사용됐다.

정확한 문제 제기는 이것이다:

> **2024 데이터를 2025 예측에서 얼마만큼 가중해야 하는지는 검증된 적이 없다.**
> (B-1 실험은 2024를 학습에서 뺀 상태에서 2019–2023 구간의 가중치만 조정했다.)

### 1.2 ⚠️ LB로 weight를 튜닝하면 안 된다

Public = Private이므로 `2024 weight 2.4 제출 / 4.0 제출 / 6.0 제출` 식의 탐색은
**Public LB를 validation set으로 쓰는 것**이며 대회 원칙과 충돌한다. **금지.**

### 1.3 로컬에서 결정하는 방법

- **RECENCY-AUDIT-001** — 3 valid 시즌 × {gap / immediate / volume-matched} 9 fit, weight 균일.
  → 최근 시즌 추가 효과를 데이터 양 효과와 분리해 측정.
- **WEIGHT-LAG-001** — lag-relative 스케줄 `w(lag) = m^(1 − (lag−1)/(L−1))` 로 재파라미터화.
  현행 B-1은 `m = 2.0`에 해당한다. `m ∈ {1.0, 1.5, 2.0, 3.0, 4.0}`을 세 fold에서 동일 규칙으로 테스트 (15 fit).
  → **2025 LB를 보지 않고 final의 2024 weight를 결정할 수 있다.**

설계 상세: `reports/lb_anchor_analysis_001.md` §5.

---

## 2. C-4 — 이미 검증된 +5.79가 제출본에 없다

`outputs/eda/excluded_feature_recheck.md` (2026-08-07):

공식 데이터 설명서 인용으로 셋 다 **"투구 직전"** 지표임이 확인됐다(leakage 아님).

| variant | added | n_feat | raw BSS | Δ vs C-0 |
|---|---|---:|---:|---:|
| C-0 baseline (= **현재 제출본**) | — | 42 | 739.97 | — |
| C-1 | `li` | 43 | 743.86 | +3.89 |
| C-2 | `home_win_expectancy` | 43 | 743.46 | +3.49 |
| C-3 | `away_win_expectancy` | 43 | 743.79 | +3.82 |
| **C-4** | **세 개 전부** | **45** | **745.76** | **+5.79** |

측정도 안전 확인도 끝났으나 `configs/baseline.yaml`의 `exclude`에 남아 있고 제출본은 42 feat 그대로다.

---

## 3. 모델 capacity — Optuna 탐색 범위 밖

`scripts/run_exp006_tuning_ensemble.py`:

```python
"iterations":    trial.suggest_int("iterations", 200, 500)
"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
```

**iterations 상한 500, lr 하한 0.02.** 약신호 tabular의 표준 영역인 `lr 0.01~0.02 × 2000~8000 iterations + early stopping`이 탐색 밖이다.

현재 제출 모델은 `300 trees × lr 0.05`로 유효 학습 예산 `≈15`다. 신호가 상수 대비 0.88%인 문제에서는 통상 30~50이 필요하다. **underfit 가설이며, 검증 비용이 가장 싸다.**

Optuna 최적값(361 iters, lr 0.0282, depth 7)이 baseline 대비 +4.6에 그친 것도 "탐색 범위 안에서는 더 얻을 게 없다"는 뜻이지 capacity가 충분하다는 뜻이 아니다.

---

## 4. Trackman 353MB — 파생 feature 1개만 사용 중

- 파일: `data/raw/trackman_history.csv`, **1,793,078행**, 2019–2024
- 모델에 들어간 것: `tm_expected_ivb_by_count` **1개** (TM-003A, 2024 +19.3, 미병합 브랜치)

**막힌 지점** (`reports/trackman_audit_001b.md`, `trackman_audit_002.md`):

| 항목 | 결과 |
|---|---|
| 투수/타자 ID 겹침 | **0 / 0** |
| hand 코딩 | main `{1,2}` vs TM `{Left, Right}` — **공식 맵 없음** |
| L1/L2/L3 커버리지 | hand 맵 없이 조인 불가 |
| L0 count-only 커버리지 | 1.000 (2022/2023/2024), 그룹 n 충분 |

저장소는 이를 **"공식 맵 없음 → HOLD"**로 닫았다.

**재검토 근거:** hand는 **2-way 매핑**이므로 데이터로 식별 가능하다.
- 좌완/우완 marginal 분포 매칭
- 팀 × 시즌 단위 집계 대조
- 구종 구성(pitch mix) 분포 대조

세 방법이 일치하면 매핑이 확정된다. **"공식 문서에 없다"와 "데이터로 알 수 없다"는 다르다.**
hand가 풀리면 `hand × count × pitch_type` 프로파일이 열리며, 이는 단일 feature(+19)와 규모가 다른 family다.

**전략적 위치:** 리더보드 1등→100등 구간 밀도가 그 아래의 2.4배라는 것은 상위 100명이 다른 무언가를 갖고 있다는 신호다(`reports/leaderboard_position_001.md` §3). Trackman이 현재 유일하게 식별된 후보다.

**제약 (유지):** `trackman.season < row.season`, 현재 투구 TM 사용 금지, 2025 TM 금지, test 집계 금지.

---

## 5. 선수 정체성의 일반화 표현 — 부재

L42는 `pitcher_id` / `batter_id`를 제거해 +161.79를 얻었으나, **선수 정체성 정보를 통째로 버렸다.**
`asof_*` 19개 컬럼이 일부 대체하지만, **unseen 선수에게도 작동하는 표현**은 없다.

- 2024 fold unseen 비율: `pitcher_id` **19.86%**, `batter_id` 9.30%
- 표준 해법: OOF target encoding + shrinkage `w = n/(n+k)`, unseen → prior fallback
- 구현 자산: FE-002에서 row-local shrinkage 코드를 이미 작성함 (`scripts/fe002_prev1_shrinkage.py`)

**주의:** fold 내부에서만 계산해야 하며 test 집계는 규칙 위반이다.

---

## 6. seed ensemble — 이득의 크기가 항등식으로 정해진다

`outputs/eda/seed_variance.md`, `seed_ensemble.md` (L42, 2024 holdout, EXP-005 params):

| seed | 42 | 1 | 7 | 123 | 2024 |
|---|---:|---:|---:|---:|---:|
| raw BSS | 739.97 | 733.74 | 723.38 | 722.86 | 723.11 |

mean **728.61**, std(ddof=1) **7.84**, range **17.11**

| | raw BSS | 기댓값(728.61) 대비 |
|---|---:|---:|
| single model 기댓값 | 728.61 | — |
| n3 ensemble (42,1,7) | 742.08 | +13.47 |
| **n5 ensemble** | **739.85** | **+11.24** |

**항등식:** 고정된 validation set에서

$$\frac1m\sum_s (p_s-y)^2 = (\bar p - y)^2 + \frac1m\sum_s (p_s-\bar p)^2
\;\Longrightarrow\;
\overline{BS}_{\text{single}} - BS_{\text{ens}} = \mathbb{E}_{\text{row}}\big[\mathrm{Var}_{\text{seed}}(p)\big]$$

실측 대입: `E[Var_seed(p)] = 2.81 × 10⁻⁵`, seed 간 예측 RMS 편차 **0.0053**.

**정확한 표현**
- **방향**: 개별 모델 Brier 평균 대비 이득 ≥ 0 — 항등식으로 보장.
- **크기**: 2025 test 행에서의 `E[Var_seed(p)]`이며 미지. 2024 fold 추정치는 +11.24.
- m의 역할: 이득 ≈ `(1−1/m)·E[Var_seed]`. m=5면 최대치의 80%, m=10이면 90%.
- **고정 seed 42(739.97) 대비 이득은 보장되지 않는다.** 다만 seed 운은 학습 데이터·평가 데이터가 바뀌면 전이될 이유가 없으므로, 2025에서 seed 42 모델의 기대 성능은 seed 분포 평균으로 보는 것이 타당하다. 이는 합리적 가정이며 수학적 보장은 아니다.
- **리스크**: 통계적 downside는 없으나 `script.py` 변경에 따른 운영 리스크는 있다. 추론 시간은 문제가 아니다(기존 제출 2건 모두 서버 소요 **2초**).

### 6.1 철회된 주장

이전 분석에서 "seed 42가 5개 중 최고값이므로 기존 판정이 낙관 편향(selection bias)"이라고 기술했다. **이는 철회한다.**
`runtime.seed = 42`는 결과를 보기 전부터 `configs/baseline.yaml`에 고정돼 있었고, 판정은 동일 seed의 paired delta로 이뤄졌으므로 seed의 절대 수준은 상쇄된다. selection bias가 아니다.

남는 유효한 지적은 **분해능 부족**뿐이다: 단일 seed paired delta의 노이즈 폭(CatBoost seed spread 11~16)이 TM-003A(+11.4 mean) · TM-002A(+17.9) 와 같은 크기다. 이는 `MODEL-TRANSFER-AUDIT-001`이 이미 기록한 내용이다.
