# LG Aimers Phase 2 — GPT / Data Analysis Handoff

- 작성일: 2026-08-07
- 범위: EXP-000 ~ EXP-006 (합성 smoke `temporal_*` 제외)
- 원칙: 저장소에 기록된 파일·로그만 반영. 수치·결정은 추측하지 않음.
- 주의: 본 문서 작성 시점에 **새 실험 실행 없음**, **기존 결과 수정 없음**.

---

## 1. 공식 metric과 local 구현

### 공식 (DACON / Competition Spec §2 — `src/metrics.py` docstring)

```
Brier Score = mean((p_i - y_i)^2)
r = mean(y)   # 평가 집합(또는 local fold) 양성 비율
Score = max(0, 100000 * (1 - BrierScore / (r * (1 - r))))
```

- Target: `control_success` (binary 0/1)
- Scale: `100000`
- Official score = `max(0, raw)` (음수 raw는 공식 점수 0)

### Local 구현

| 항목 | 위치 |
|---|---|
| 공식 BSS / raw BSS | `src/metrics.py` (`brier_skill_score`, `compute_brier_metrics`) |
| 평가 번들 | `src/evaluate.py` (`evaluate_predictions`, `score_metric`) |
| Config | `configs/baseline.yaml` → `task.metric.name: brier_skill_score`, `direction: maximize`, `clip_at_zero: true`, `scale: 100000` |

반환 필드:

- `brier_score`
- `raw_brier_skill_score` — clip 전 (진단·모델 비교용)
- `official_brier_skill_score` — `max(0, raw)` (공식 선택 점수)

`r*(1-r)==0`이면 `ValueError` (점수 발명 금지).

실험 로그·리포트는 official이 floor-clip되므로 **비교는 주로 raw BSS / Brier**를 사용한다.

---

## 2. Primary / Secondary validation

상태: **잠정 확정** (`reports/validation_policy.md`, VAL-AUDIT-001 / EXP-000 이후).  
Config: `configs/baseline.yaml` `cv.*`.

### Primary

- Train: `2019–2023`
- Valid: `2024`

### Secondary (expanding-window)

1. `2019–2021` → `2022`
2. `2019–2022` → `2023`
3. `2019–2023` → `2024` (primary와 동일 split; secondary 세트의 마지막 fold)

### 로깅 요구 (`validation_policy.md`)

Fold별: Brier, raw BSS, official BSS, valid target rate, pred mean/std/min/max.  
Secondary 집계: fold raw BSS mean±std, concatenated OOF Brier/raw/official.  
항상 primary 2024 official/raw/Brier 보고.

### Non-goals (잠정 lock이 해결하지 않는 것)

- Within-season chronology (`order_cols: []`)
- Game-leakage-free split (train에 `game_id` 없음)
- Within-season random split를 default로 쓰지 않음

---

## 3. EXP-000 ~ EXP-006 요약표

공통 데이터: `phase2_open_data_raw_copy` (`data/raw/train.csv` 등).  
공통 seed (해당 시): `42`.  
`public_lb_score`: `outputs/experiments.csv` 전 EXP 행에서 **비어 있음**.

채택 열 의미: 저장소에 명시된 `decision` / report 문구. **정식 adopt/reject는 EXP-000~006 모두 미수행** (`results_only_no_adopt_reject` 또는 동등 문구).

| experiment_id | 핵심 가설 / purpose | 변경점 | 모델 | 주요 feature | validation | fold raw BSS (sec22 / sec23 / pri24) | CV mean±std (3-fold raw) | Public LB | 채택/보류/폐기 |
|---|---|---|---|---|---|---|---|---|---|
| **EXP-000** | 시즌 holdout에서 train target mean만으로 기준선 | Feature 없음, constant `p=0.53158151` | target_mean | none | primary만 기록 (2019–23→2024). VAL-AUDIT-001에서 expanding 3-fold도 동일 로직 측정 | VAL-AUDIT: −81.177 / −626.607 / −827.887 | VAL-AUDIT mean −511.891 ± 315.451 | 없음 (미제출) | **기준선**. 결과 보고만. Public LB 없음 |
| **EXP-001A** | provisionally safe 44 features + LR | WE/LI/`row_id` 제외, asof 포함, OneHot+sparse LR | logistic_regression (`C=1`, `liblinear`, `max_iter=1000`) | 44 cols: season·상황·ID·asof_* (exclude WE/LI/row_id) | primary + secondary 3 folds | 2092.173 / −1050.469 / −464.580 | 192.375 ± 1364.488 | 없음 | **보류** (`results_only`). primary official=0 |
| **EXP-002** | 동일 feature + LightGBM native cats | encoding=`native_lightgbm` | lightgbm (`n_estimators=300`, `lr=0.05`, `num_leaves=31`) | 동일 44 | 동일 | 1580.500 / −2208.138 / 8.247 | −206.463 ± 1554.139 | 없음 | **보류**. CatBoost 대비 약함 |
| **EXP-003** | 동일 feature + CatBoost native cats | encoding=`native_catboost` | catboost (`iter=300`, `lr=0.05`, `depth=6`, Logloss) | 동일 44 | 동일 | 2251.518 / −1319.883 / 562.278 | 497.971 ± 1458.727 | 없음 | **보류**. 이후 tree 기준선으로 사용 |
| **EXP-004** | 최근 시즌에 더 큰 sample weight | season weights `2019:1.0 … 2023:2.0` (regime A 승격) | catboost (동일 params + weights) | 동일 44 | 동일 | 2259.398 / −1366.481 / 578.184 | 490.367 ± 1481.561 | 없음 | **보류**. primary는 EXP-003 대비 +15.9 raw; weight 정책은 이후 EXP에 재사용. `EXP-004_submit.zip` 존재 |
| **EXP-005** | L(drop pitcher/batter id) + full-train 제출 | S: primary에서 L±game_type 비교 → `R_L_keep_game_type` 선택; R: 2019–2024 full train, weights에 `2024:2.4` | catboost (EXP-004 params) | **42** (drop `pitcher_id`,`batter_id`; **game_type 유지**) | 선택: S primary_2024. Final: full_train 홀드아웃 없음 | S1(L keep GT): pri24 **739.970**; S2(L drop GT): **703.145**. Secondary 전체 CV 미실행 | CSV `cv_score`=739.970 (S1; std 없음) | 없음 | **보류**. submit zip 준비됨. `decision: results_only` |
| **EXP-006** | Optuna CB/LGB + ensemble, L feature, full-train | 35 trials each; ensemble grid; **tuned_catboost** 선택 후 full train | selected: catboost tuned (`depth=7`, `lr≈0.0282`, `l2≈1.215`, `iter=361`, `bagging_temp≈0.966`); LGB/ensemble은 holdout만 | **42** (L + game_type kept) | tune holdout: 2019–23→2024. Final: full 2019–2024 | secondary folds **미실행**. Holdout raw: CB **744.676**, LGB 613.279, ens 0.7/0.3=727.026 | CSV `cv_score`=`holdout`=744.676 (std 없음) | 없음 | **보류**. local primary 최고. `EXP-006_submit.zip` 준비. `next_action`: optional submit |

### 점수 출처

- EXP-000: `outputs/predictions/EXP-000/report.md`, VAL-AUDIT: `reports/season_validation_audit.md`
- EXP-001A~004: `outputs/predictions/EXP-00x/summary.json` + `report.md`, `outputs/experiments.csv`
- EXP-005: `outputs/eda/full_train_experiments.md` / `.json`, `outputs/predictions/EXP-005/summary.json`
- EXP-006: `outputs/eda/tuning_ensemble.md` / `.json`, `outputs/predictions/EXP-006/summary.json`

### 참고: 정식 EXP ID가 아닌 진단 실험 (결과만)

| ID | 요약 | 주요 수치 |
|---|---|---|
| regime A | sample weight → EXP-004 | pri24 raw 578.184 |
| regime B | train 2022–23 only | pri24 raw 540.829 (EXP-003 대비 −21.4) |
| D / E / F / G | season feature 변형 | D drop season −61.6; G categorical −357.9 |
| L_drop_ids | drop pitcher/batter id | sec23 −1306.6; pri24 **739.970** |
| O_drop_game_type | drop game_type | sec23 **−384.1**; pri24 **464.5** (2024 하락) |
| M / N | asof relative-to-prior-season | pri24 소폭~중간 개선, sec23 악화 |

출처: `outputs/eda/regime_shift_experiments.md`, `season_extrapolation_experiments.md`, `unseen_id_experiments.md`, `gap_diagnosis.md`.

---

## 4. 현재 best experiment

**Local primary (2019–2023 → 2024) raw BSS 기준 최고 기록값: EXP-006 `tuned_catboost` = 744.675900**  
(`outputs/eda/tuning_ensemble.md`, `outputs/experiments.csv` EXP-006).

비교:

| 실험 | Primary 2024 raw BSS | Secondary 3-fold 완료 |
|---|---:|:---:|
| EXP-003 | 562.278 | Yes |
| EXP-004 | 578.184 | Yes |
| EXP-005 S1 (L keep game_type) | 739.970 | No (S만) |
| **EXP-006 tuned CB** | **744.676** | **No** |

Public LB로 검증된 best는 **없음** (`public_lb_score` 전부 공란).

---

## 5. 현재 채택된 feature / model / validation 결정

정식 “채택” 문서 결정은 없으나, **운영상 잠정 고정·재사용 중인 설정**은 다음과 같다.

### Validation — 잠정 확정

- Primary / secondary: §2 (`reports/validation_policy.md`, `configs/baseline.yaml`)

### Feature — 잠정

| 결정 | 상태 | 근거 |
|---|---|---|
| Exclude `row_id`, `home_win_expectancy`, `away_win_expectancy`, `li` | baseline.yaml / 전 EXP | `reports/feature_safety_audit.md` (WE/LI = leakage review required) |
| 공식 asof_* 사용 | 사용 중 | feature safety: safe |
| Drop `pitcher_id`, `batter_id` (L) | EXP-005/006 submit 경로 | unseen_id L: pri24 +161.8 vs EXP-004 |
| Keep `game_type` | EXP-005/006 선택 | S1 > S2 on primary_2024 |
| Keep numeric `season` | 유지 | D/G 등에서 season 제거/categorical 악화 |

### Model — 잠정 운영

- Tree 계열 **CatBoost**가 LR/LightGBM(untuned)보다 primary에서 우위 (EXP-003+).
- Submit 후보 최신: **EXP-006 full-train tuned CatBoost** (`outputs/submissions/EXP-006_submit.zip`).
- Season sample weights: holdout `{2019:1.0, 2020:1.2, 2021:1.4, 2022:1.6, 2023:2.0}`; full train 시 `2024:2.4` 추가 (EXP-005/006).

### 명시적으로 폐기하지 않은 것

- EXP-001A/002: 폐기 선언 없음 → **보류**.
- Ensemble (EXP-006): holdout에서 single CB보다 낮아 submit 미선택; 폐기 선언 없음.

---

## 6. Leakage 관련 확인·미해결

### 확인됨 (`outputs/eda/leakage_check.md`, audits)

| 항목 | 결과 |
|---|---|
| Fold: season 완전 분리, shuffle 없음 | `build_season_holdout` — train/valid disjoint, train 시즌이 valid보다 엄격히 이전 |
| asof `prev*_game_*` | pitcher-season 내 mid-season NA re-entry 0; cold-start 패턴과 일치 (sanity, 공식 구성 증명 아님) |
| 금지 입력 | 현재 pitch Trackman / 참가자 test aggregate 미사용 (feature audit) |

### 미해결 / 한계

| 항목 | 내용 |
|---|---|
| WE / LI | 공식 문구만으로는 산출식 미확인 → **계속 exclude** |
| `game_id` 없음 | 시즌 내 game leakage 차단 불가 (`validation_structure_audit.md`) |
| `order_cols: []` | within-season 시간축 CV 불가 |
| Pitcher entity overlap | 2022–24 중 3년 모두 등장 pitcher 208/591 ≈ 35% — row leakage는 아니나 ID 암기 가능 |
| Trackman | `pitcher_id ∩ pitcher_trackman_id = 0`, join key 없음 → pitch-type × target 교차표 **불가** (`target_regime_check.md`) |
| asof 정렬 | train에 game timestamp 없어 `asof_pitcher_n`+`row_id` proxy |

---

## 7. CV–LB gap

| 항목 | 상태 |
|---|---|
| `outputs/experiments.csv`의 `public_lb_score`, `cv_lb_gap` | EXP-000~006 **전부 공란** |
| Public / Private LB 점수 | 저장소에 **기록 없음** |
| Local “gap” 진단 | `gap_diagnosis.md`: **future season risk ≈ secondary_2023 실패 모드** 가설 점검. EXP-004 style on 2019–22→2023: raw BSS **−1366.481** (official 0). Drop `season`은 2023에서 완화(+275 raw vs kept)하나, primary_2024에서 season drop은 별도 D 실험에서 붕괴 |

→ **CV–Public LB gap은 수치로 확인 불가.** Local로는 primary(2024)와 secondary_2023 간 괴리가 큼 (예: EXP-004 pri24 +578 vs sec23 −1366).

---

## 8. 주요 reports / outputs / configs 경로

### Config

- `configs/baseline.yaml` — 공식 데이터·metric·CV·feature exclude
- `configs/synthetic.yaml` — smoke only

### Reports

- `reports/validation_policy.md`
- `reports/validation_structure_audit.md`
- `reports/feature_safety_audit.md`
- `reports/season_validation_audit.md` (VAL-AUDIT-001)
- `reports/eda_report.md`

### Experiment registry

- `outputs/experiments.csv`

### Predictions / models / submissions

| EXP | predictions | models | submissions |
|---|---|---|---|
| EXP-000 | `outputs/predictions/EXP-000/` | `outputs/models/EXP-000/` | `submission_EXP-000_target_mean.csv` |
| EXP-001A | `outputs/predictions/EXP-001A/` | `outputs/models/EXP-001A/` | — |
| EXP-002 | `outputs/predictions/EXP-002/` | `outputs/models/EXP-002/` | — |
| EXP-003 | `outputs/predictions/EXP-003/` | `outputs/models/EXP-003/` | — |
| EXP-004 | `outputs/predictions/EXP-004/` | `outputs/models/EXP-004/` | `EXP-004_submit.zip` (+ reports) |
| EXP-005 | `outputs/predictions/EXP-005/summary.json` | `outputs/models/EXP-005/R_L_*` | `EXP-005_submit.zip` |
| EXP-006 | `outputs/predictions/EXP-006/summary.json` | (zip 내 model) | `EXP-006_submit.zip` |
| VAL-AUDIT-001 | `outputs/predictions/VAL-AUDIT-001/` | — | — |

### EDA / 진단

- `outputs/eda/leakage_check.md`, `_leakage_meta.json`
- `outputs/eda/data_integrity.md`, `fold_comparison.md`, `target_regime_check.md`
- `outputs/eda/regime_shift_experiments.md|json`
- `outputs/eda/season_extrapolation_experiments.md|json`
- `outputs/eda/season_categorical_experiment_G.md|json`
- `outputs/eda/gap_diagnosis.md|json`
- `outputs/eda/unseen_id_experiments.md|json`
- `outputs/eda/full_train_experiments.md|json`
- `outputs/eda/tuning_ensemble.md|json`

### Scripts (재현)

- `scripts/run_exp000_target_mean.py` … `run_exp004_catboost_weighted.py`
- `scripts/run_full_train_exp005.py`
- `scripts/run_exp006_tuning_ensemble.py`
- `scripts/tree_exp_common.py`, `run_val_audit_001.py`, regime/unseen/gap 스크립트들

### Metric code

- `src/metrics.py`, `src/evaluate.py`

---

## 9. 다음 실험 후보 (저장소에 근거가 있는 항목만)

새 EXP ID는 아직 정의되지 않음. 문서/`next_action`에 남은 것:

1. **Optional: Public 제출** — `EXP-006_submit.zip` (또는 EXP-005). LB 점수 기록 후 `experiments.csv`의 `public_lb_score` / `cv_lb_gap` 채우기.
2. **secondary_2023 실패 모드** — `gap_diagnosis.md` / `unseen_id` O: `game_type` drop은 2023을 크게 개선(+982 raw vs EXP-004)하지만 2024는 악화. 2025 test 일반화와 tradeoff 검증 필요.
3. **정식 adopt/reject** — 전 EXP가 `results_only`. feature L, weights, CatBoost, validation lock을 문서상 채택할지 결정.
4. **WE/LI 재검토** — 주최측 산출식 확인 후에만 재도입 (`feature_safety_audit.md` follow-up).
5. **Trackman** — join key 없으면 pitch-level FE 불가; mapping 확보 전 보류.
6. **EXP-006 한계** — Optuna/ensemble이 **단일 2024 holdout**에서만 튜닝됨. Secondary 3-fold로 tuned params 재평가 여부.
7. `configs/baseline.yaml`의 `project.next_action` 문자열은 EXP-001A 시점으로 **오래됨** — 문서/설정 동기화.

---

## 10. 아직 확인되지 않은 사항

| 항목 | 비고 |
|---|---|
| Public / Private LB 점수 | 저장소 미기록 |
| EXP-005/006 secondary fold scores | 미실행 |
| EXP-006 ensemble의 full-train 성능 | submit은 single CB만 |
| WE/LI 실제 leakage 여부 | 산출식 미확인 |
| `row_id`가 공식 시간 순서인지 | uniqueness만 확인; time key로 미채택 |
| Trackman↔train 매핑 | ID namespace 불일치 |
| 2025 test 분포 vs 2023/2024 | test는 format sample 언급 (`data_integrity.md`); full test label 없음 |
| `configs/baseline.yaml` next_action / notes vs 최신 EXP | drift |
| Competition Spec 원문 파일 경로 | metric은 `src/metrics.py` docstring에 인용; 별도 Spec 파일은 본 handoff에서 경로 미확인 |

---

## Quick start for next agent

1. Read this file + `outputs/experiments.csv`.
2. Metric: `src/metrics.py`.
3. Validation: `reports/validation_policy.md`.
4. Best local holdout: EXP-006 CB 744.676; submit zip ready; **no LB yet**.
5. Do **not** invent adopt/reject — none were recorded.
6. Do **not** assume Trackman join works.
7. Prefer raw BSS for ranking when official BSS is clipped at 0.
)
