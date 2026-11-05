# LG Aimers 9기 Phase 2 — 투구 단위 제구 성공 확률 예측

[![Unit tests](https://github.com/sooyahh/lgaimers-9th/actions/workflows/tests.yml/badge.svg)](https://github.com/sooyahh/lgaimers-9th/actions/workflows/tests.yml)

**Pitch-level probability forecasting for LG AI Research's LG Aimers 9th hackathon (DACON), done individually.** On 1.47M pitches scored by Brier Skill Score, I locked season-based validation before modeling. I then isolated why the final model improved by submitting a control run that removed only the player-ID features. Removing the IDs added **+269** of the **+471** leaderboard gain. The final score was **877.60**, which passed the completion threshold (549.51) but did not reach the Phase 3 cut.

---

LG AI연구원 LG Aimers 9기(2026-06-22 ~ 09-02)의 Phase 1 AI 교육과 Phase 2 해커톤을 수료했습니다. 이 저장소는 Phase 2 해커톤의 실험 코드와 집계 결과입니다. **개인 참가**이며, 코드·문서·분석은 모두 sooyahh가 작성했습니다.

- 과제: 투구 단위 `control_success`(0/1) 확률 예측 (DACON)
- 데이터: train 1,475,092행 × 49열 (2019~2024 시즌), Trackman 과거 로그 1,793,078행
- 평가: Brier Skill Score = `max(0, 100000 × (1 − Brier / (r(1−r))))`, 평가 시즌은 비공개 2025

## 결과 — Public LB와 원인 분리

평가 시즌(2025)에는 학습 때 보지 못한 선수가 많다는 가설로, 선수 ID(`pitcher_id`, `batter_id`)를 뺀 모델과 **ID만 다르고 나머지는 같은 통제 제출**(LB-PROBE)을 따로 냈습니다. 상세: [`reports/lb_anchor_analysis_001.md`](reports/lb_anchor_analysis_001.md)

| 제출 | 바꾼 것 | 학습 시즌 | 로컬 2024 raw BSS | Public LB |
|---|---|---|---:|---:|
| `EXP-004` | 기준 (ID 포함, 44 feature) | 2019–2023 | 578.18 | 406.95 |
| `LB-PROBE` | 선수 ID만 제거 (42 feature) | 2019–2023 | 739.97 | 676.21 |
| **`EXP-005`** (최종) | ID 제거 + 2024 시즌 추가 학습 | 2019–2024 | — | **877.60** |

- 전체 향상 +470.65 = **선수 ID 제거 +269.26 (57.2%)** + 2024 시즌 추가 +201.39 (42.8%)
- ID 제거 효과는 로컬 2024 holdout에서 +161.79, Public LB에서 +269.26으로 1.66배 커졌습니다. 로컬 점수만 보면 일반화의 가치를 과소평가하게 됩니다. 이 비율은 변화 1건의 관찰이며 일반화하지 않습니다.
- 이 대회는 Public LB = Private LB라서, 하이퍼파라미터를 LB 점수로 고르지 않고 결과 해석 규칙을 제출 전에 정했습니다.
- `EXP-005`는 2024 시즌까지 학습했으므로 로컬 2024 점수를 매기지 않습니다. 같은 피처 구성을 2019–2023으로 학습한 점수가 `LB-PROBE`의 739.97입니다.

## 내가 한 일

1. **검증 체계를 먼저 고정했습니다.** 시즌 단위 검증(2019–2023 학습 → 2024 검증)과 expanding-window 보조 검증 3개를 [`reports/validation_policy.md`](reports/validation_policy.md)로 고정했고, 시즌 내부 무작위 분할은 쓰지 않았습니다.
2. **피처 안전성을 감사했습니다.** 공식 데이터 설명서만 근거로 48개 입력 열을 안전 44 · 누수 검토 필요 3 · 제외 1로 분류했고, 근거가 불분명한 3개(승리 기대값 2개, 상황 중요도)는 기본 제외했습니다 ([`reports/feature_safety_audit.md`](reports/feature_safety_audit.md)).
3. **기준선부터 단계적으로 실험했습니다.** 같은 검증·seed(42)로 아래 실험을 비교했습니다.
4. **원인을 분리했습니다.** 한 번에 한 변수만 바꾸는 통제 제출(LB-PROBE)로 점수 향상을 두 요인으로 나눴습니다.
5. **모든 실험을 기록했습니다.** 브랜치 → PR → 병합 흐름과 실험 레지스트리([`outputs/experiments.csv`](outputs/experiments.csv))로 실험을 ID 하나로 추적했습니다. 혼자 작업한 뒤에도 같은 규칙([`CONTRIBUTING.md`](CONTRIBUTING.md))을 지켰습니다.

### 실험 기록 (로컬 2024 검증, raw BSS)

| ID | 내용 | raw BSS |
| --- | --- | ---: |
| EXP-000 | 학습 데이터 평균 기준선 | −827.9 |
| EXP-001A | 로지스틱 회귀 + 안전 피처 44개 | −464.6 |
| EXP-002 | LightGBM | 8.2 |
| EXP-003 | CatBoost | 562.3 |
| EXP-004 | CatBoost + 시즌 가중치 | 578.2 |
| EXP-005 | 선수 ID 제거(42개 피처) | 740.0 |
| EXP-006 | Optuna 튜닝 CatBoost (로컬 최고) | 744.7 |

이후 Trackman 물리량 기반 피처(TM 계열)와 직전 투구 피처(FE 계열)를 추가하는 실험을 같은 검증으로 진행했고, 제출하지는 않았습니다. 상세: [`docs/STATUS.md`](docs/STATUS.md), [`reports/next_experiment_decision.md`](reports/next_experiment_decision.md)

## 한계

- 결과는 수료 기준 통과 수준이며, Phase 3 진출 컷(상위 100명, 약 1,102점)과는 약 225점 차이가 났습니다.
- 1위도 상수 예측 대비 Brier를 1.2%만 개선한, 신호가 매우 약한 과제였습니다.
- 상위권과의 차이를 만든 요인(Trackman 물리량 활용 등)은 대회 기간 안에 검증하지 못했습니다.
- 시즌 내부 시간순 정렬 정보가 없어 경기 단위 누수가 없다고는 주장하지 않습니다.

## 저장소 구조

```text
configs/      baseline.yaml(검증·metric·seed 고정), synthetic.yaml
src/          공통 파이프라인 (train, predict, metrics, evaluate, ...)
scripts/      실험·검증·EDA 러너 (EXP-000~006, FE/TM, LB-PROBE)
tests/        unittest (대회 데이터 없이 합성 데이터로 실행)
reports/      검증 정책, 피처 안전성, 리더보드·오류 분석
outputs/      실험 레지스트리(experiments.csv), EDA·분석 집계표
docs/         진행 상태(STATUS), 실험 핸드오프, 피처 심층 분석
```

`reports/gpt_audit/`와 `docs/GPT_HANDOFF.md`는 외부 AI 검토용으로 실험 요약을 묶은 패키지입니다. 전략 검토, 문서화, 코드 작성을 서로 다른 AI 도구에 나눠 맡기고, 결과는 산출물과 실험 기록으로만 판단했습니다.

## 실행

모든 명령은 저장소 루트에서 실행합니다.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

테스트는 대회 데이터 없이 합성 데이터로 동작합니다. 실험을 재현하려면 DACON에서 받은 `train.csv`, `test.csv`, `sample_submission.csv`를 `data/raw/`에 넣고 실험 러너를 모듈로 실행합니다.

```bash
python -m scripts.run_exp004_catboost_weighted
python -m scripts.run_full_train_exp005
python -m scripts.run_exp006_tuning_ensemble
```

`src`를 import하므로 `python scripts/....py`가 아니라 `python -m scripts.<module>` 형식을 써야 합니다. 경로·컬럼은 `configs/baseline.yaml`을 따릅니다.

## 공개용 사본 안내

- 이 저장소는 대회 종료 시점(2026-08-22)의 작업 저장소 `main`을 포트폴리오용으로 옮긴 것입니다. 대회 중 잠시 팀을 결성해 협업용으로 구성했다가 개인 참가로 돌아갔으며, 팀원의 실험 브랜치는 넣지 않았습니다.
- 대회 데이터 재배포 금지에 따라 원본 데이터, 행 단위 예측, 모델 파일은 넣지 않았습니다. `outputs/`와 `reports/`의 결과는 모두 집계 통계입니다.
- `outputs/experiments.csv`의 `temporal_…` 행은 파이프라인 점검용 실행 기록입니다. 레지스트리는 추가만 하는 규칙이라 지우지 않았습니다.

## License

코드는 [MIT](LICENSE)입니다. 대회 데이터에는 적용되지 않습니다.
