# LG Aimers Phase 2

> **공개용 사본 안내**
> 이 저장소는 팀 저장소(`LGaimers-9th/lgaimers`, 비공개)의 대회 종료 시점 `main`(2026-08-22, `7d37d2d`)을 포트폴리오용으로 옮긴 것입니다. `main`의 코드·문서·분석 결과는 모두 sooyahh가 작성했으며, 팀원의 실험 브랜치는 넣지 않았습니다.
>
> - **참가 형태: 개인 참가.** 대회 중 잠시 팀을 결성해 이 저장소를 팀 협업용으로 구성했으나, 팀을 해체하고 개인 참가로 돌아갔습니다. 아래 본문의 '팀 단위 협업' 규칙은 그때 만든 것으로, 이후에도 혼자 같은 규칙(브랜치 → PR → 병합, 실험 레지스트리)을 지키며 작업했습니다.
>
> - 대회 데이터 재배포 금지에 따라 원본 데이터(`data/raw/`), 행 단위 예측, 모델 파일은 넣지 않았습니다. `outputs/`와 `reports/`의 결과는 모두 집계 통계입니다.
> - 커밋 이력은 옮기지 않았고, 문서 안의 로컬 절대 경로는 저장소 기준 상대 경로로 바꿨습니다.
> - 아래 본문은 대회 당시 안내서를 그대로 둔 것입니다. 실행하려면 DACON에서 받은 데이터를 `data/raw/`에 직접 넣어야 합니다.

DACON LG Aimers 9기 Phase 2 — 투구 단위 이진 확률 예측 (`control_success`).

공식 오픈 데이터·metric·스키마를 기준으로 시즌 holdout / expanding-window 검증과 baseline부터 제출용 실험까지(EXP-000–006)을 재현한다.

이 저장소는 개인 실험 파이프라인을 유지한 채 팀 단위로 실험·리뷰·병합하기 위한 공용 코드베이스다. 협업 규칙의 상세는 `CONTRIBUTING.md`를 따른다.

## 결과 요약 — Public LB와 원인 분리

평가 시즌(2025)에는 학습 때 보지 못한 선수가 많다는 가설로, 선수 ID(`pitcher_id`, `batter_id`)를 뺀 모델과 **ID만 다르고 나머지는 같은 통제 제출**(LB-PROBE)을 따로 냈다. 상세: [`reports/lb_anchor_analysis_001.md`](reports/lb_anchor_analysis_001.md)

| 제출 | 바꾼 것 | 학습 시즌 | 로컬 2024 raw BSS | Public LB |
|---|---|---|---:|---:|
| `EXP-004` | 기준 (ID 포함, 44 feature) | 2019–2023 | 578.18 | 406.95 |
| `LB-PROBE` | 선수 ID만 제거 (42 feature) | 2019–2023 | 739.97 | 676.21 |
| **`EXP-005`** (최종) | ID 제거 + 2024 시즌 추가 학습 | 2019–2024 | 739.97 | **877.60** |

- 전체 향상 +470.65 = **선수 ID 제거 +269.26 (57.2%)** + 2024 시즌 추가 +201.39 (42.8%)
- ID 제거 효과는 로컬 2024 holdout에서 +161.79, Public LB에서 +269.26으로 **1.66배** 커졌다 (이 변화 1건의 전이율이며 일반화하지 않음).
- 이 대회는 Public LB = Private LB라서, 하이퍼파라미터를 LB 점수로 고르지 않고 해석 규칙을 제출 전에 정했다.
- 결과는 수료 기준(549.51) 통과 수준이며, Phase 3 진출 컷(상위 100명)에는 미달했다.

## Competition / Problem

| 항목 | 값 |
|---|---|
| Target | `control_success` (0/1) |
| 제출 키 | `row_id` |
| Metric | Brier Skill Score (공식: `max(0, 100000 * (1 - Brier / (r(1-r))))`) |
| 데이터 | `data/raw/train.csv`, `test.csv`, `sample_submission.csv` (`phase2_open_data_raw_copy`) |
| Config | `configs/baseline.yaml` |

모델 비교·진단에는 official clip 전 **raw BSS** / Brier를 주로 쓴다. 구현: `src/metrics.py`, `src/evaluate.py`.

## Repository 목적

- 공통 validation / metric / seed를 고정한 상태로 실험을 재현한다.
- 실험은 별도 branch에서 수행하고, `main`에는 검증된 공용 코드만 Pull Request로 합친다.
- 실험 로그(`outputs/experiments.csv`)로 팀 전체 `experiment_id`를 한 namespace로 관리한다.
- 대회 raw data와 대형 모델 artifact는 Git에 올리지 않는다.

## Validation (잠정 확정)

`reports/validation_policy.md`, VAL-AUDIT-001 이후:

- **Primary:** train `2019–2023` → valid `2024`
- **Secondary:** expanding  
  `2019–2021→2022` / `2019–2022→2023` / `2019–2023→2024`

시즌 내부 시간순(`order_cols`)은 아직 비어 있고, game-level leakage-free split은 주장하지 않는다.

공통 seed는 `configs/baseline.yaml`의 `runtime.seed` (**42**)다.

## Metric

공식 점수:

```
Brier Score = mean((p_i - y_i)^2)
r = mean(y)
Score = max(0, 100000 * (1 - Brier / (r * (1 - r))))
```

- Config: `configs/baseline.yaml` → `task.metric.name: brier_skill_score`, `clip_at_zero: true`, `scale: 100000`
- Official BSS는 음수 raw를 0으로 clip하므로, 모델 비교는 raw BSS / Brier를 우선한다.

## 실험 상태 확인 위치

| 위치 | 내용 |
|---|---|
| `outputs/experiments.csv` | 실험 레지스트리 (공식 로그) |
| `docs/GPT_HANDOFF.md` | EXP-000–006 수치·결정 상태 |
| `reports/gpt_audit/` | 외부 분석용 요약·핸드오프 |
| `reports/validation_policy.md` | CV lock |

현재 기록된 실험 요약 (출처: 위 레지스트리·핸드오프. Public LB는 `experiments.csv`에 없음. 정식 adopt/reject는 보류 `results_only`):

| ID | 요약 | Primary 2024 raw BSS (대략) |
|---|---|---:|
| EXP-000 | train target mean 기준선 | −827.9 |
| EXP-001A | logistic + provisionally safe 44 feat | −464.6 |
| EXP-002 | LightGBM native cats | 8.2 |
| EXP-003 | CatBoost native cats | 562.3 |
| EXP-004 | CatBoost + season sample weight | 578.2 |
| EXP-005 | L features, full-train 제출 후보 | 740.0 (S1) |
| EXP-006 | Optuna tuned CatBoost (local 최고) | **744.7** |

다음 공식 실험 ID는 `EXP-007`부터 사용한다. 예약·명명 규칙은 `CONTRIBUTING.md`.

## Feature 정책

기본 exclude (`configs/baseline.yaml`):

- `row_id`
- `home_win_expectancy`, `away_win_expectancy`, `li` (구성 확인 전 제외)

EXP-005/006는 추가로 `pitcher_id`, `batter_id`를 제거한 L feature set(42열, `game_type` 유지)을 사용한다. 상세: `reports/feature_safety_audit.md`.

## Repository 구조

```text
configs/          baseline.yaml, synthetic.yaml
data/raw/         대회 CSV (로컬만; .gitkeep만 추적)
data/processed/   합성 smoke 데이터 등 (gitignore)
docs/             GPT_HANDOFF 등
notebooks/        EDA 노트북
outputs/experiments.csv   실험 레지스트리 (추적)
outputs/eda/      요약 md/json (추적)
outputs/models/   학습 모델 (gitignore)
outputs/predictions/  OOF·holdout 예측 (gitignore)
outputs/submissions/  제출 CSV·zip (gitignore)
reports/          validation / feature / gpt_audit
scripts/          EXP·VAL·EDA 러너
src/              공통 파이프라인 (train, predict, metrics, evaluate, ...)
tests/            unittest (대회 CSV 불필요)
```

## 설치

저장소 루트에서:

```powershell
pip install -r requirements.txt
```

CatBoost / LightGBM은 optional이지만 tree 실험에 필요하다. Optuna는 EXP-006 스크립트에서 import한다 (`requirements.txt`에는 아직 없다).

## 데이터 배치

대회 원본 CSV를 로컬에만 둔다 (`.gitignore`: `data/raw/*`). Git에는 `data/raw/.gitkeep`만 있다.

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

경로·컬럼은 `configs/baseline.yaml`의 `data.*`와 맞춘다.

## 테스트

대회 데이터 없이 합성 데이터로 동작한다. clone 후 아래를 먼저 실행한다.

```powershell
python -m unittest discover -s tests -v
```

`python scripts/generate_synthetic_data.py`는 `data/processed/synthetic/`에 smoke CSV를 만든다.
`python -m src.train --config configs/synthetic.yaml`은 공유 `outputs/experiments.csv`에 smoke 행을 추가하므로, 온보딩 확인은 unittest를 사용한다.

## 실행

모든 명령은 **저장소 루트**에서 실행한다. `src`를 import하는 실험 스크립트는 `python scripts/...py`가 아니라 `python -m scripts.<module>`을 사용한다. (`python scripts/run_exp000_target_mean.py`는 `ModuleNotFoundError: No module named 'src'`가 난다.)

공통 파이프라인 (대회 CSV 필요):

```powershell
python -m src.train --config configs/baseline.yaml
python -m src.predict --config configs/baseline.yaml --experiment-id EXPERIMENT_ID
```

`python -m src.train`은 `temporal_YYYYMMDD_...` ID를 만들고, 공식 EXP ID(`EXP-007` 등)가 아니다. 팀 실험은 `scripts/` 러너처럼 ID를 명시한다.

실험별 스크립트 예 (대회 CSV 필요; EXP-005/006는 `--help` 없이 바로 학습을 시작한다):

```powershell
python -m scripts.run_exp000_target_mean
python -m scripts.run_exp001a_logistic
python -m scripts.run_exp002_lightgbm
python -m scripts.run_exp003_catboost
python -m scripts.run_exp004_catboost_weighted
python -m scripts.run_full_train_exp005
python -m scripts.run_exp006_tuning_ensemble
```

합성 데이터 생성:

```powershell
python scripts/generate_synthetic_data.py
```

enriched OOF는 로컬에서만 생성한다 (`reports/gpt_audit/oof/`는 gitignore):

```powershell
python scripts/build_gpt_audit_oof.py
python scripts/verify_gpt_audit_oof_joins.py
```

## Experiment log / submission

| 경로 | 내용 | git |
|---|---|---|
| `outputs/experiments.csv` | 실험 레지스트리 (`author`, `git_branch`, `git_commit`, `git_dirty` 포함) | 추적 |
| `outputs/eda/` | EDA·진단 요약 | 추적 (`_leakage_meta.json` 등 row-level sample은 제외) |
| `outputs/models/` | 학습된 모델 | 무시 |
| `outputs/predictions/` | OOF / holdout 예측 | 무시 |
| `outputs/submissions/` | 제출 CSV·zip | 무시 |
| `reports/` | validation / feature / season audit | 추적 |
| `reports/gpt_audit/` | 외부 분석용 요약 | 추적 (OOF CSV·zip 제외) |
| `docs/GPT_HANDOFF.md` | EXP 핸드오프 | 추적 |

제출 zip은 로컬 `outputs/submissions/`에 생성되며 git에는 올리지 않는다.

## 협업 작업 흐름

팀 운영 규칙이다. 현재 GitHub 저장소 설정만으로는 `main` 직접 push가 기술적으로 막히지 않는다.

1. `main` 최신화
2. 작업 목적 확인 (`outputs/experiments.csv`, Issue, `docs/GPT_HANDOFF.md`)
3. 별도 branch 생성 (`exp/`, `feat/`, `fix/`, `docs/`, `chore/`)
4. 하나의 핵심 가설/작업 수행
5. test 및 필요한 실행
6. experiment log 기록 (`outputs/experiments.csv`)
7. commit
8. push
9. Pull Request
10. review
11. merge

`main`에서 직접 작업하거나 `main`에 직접 push하지 않는다. 상세 규칙·PR 체크리스트는 `CONTRIBUTING.md`와 `.github/pull_request_template.md`.

## 문서 인덱스

- `CONTRIBUTING.md` — 브랜치, 실험, PR 규칙
- `docs/GPT_HANDOFF.md` — EXP-000–006 수치·결정 상태
- `reports/validation_policy.md` — CV lock
- `reports/feature_safety_audit.md` — feature 안전성
- `reports/gpt_audit/README.md` — audit 패키지 안내
