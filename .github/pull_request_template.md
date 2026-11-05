## 목적 / 가설

<!-- 한 PR에서 핵심 가설 또는 작업은 하나만. 실험이 아니면 N/A. -->

- 목적:
- 핵심 가설:

## Experiment

- experiment_id: <!-- 예: EXP-007 / N/A -->
- 실제 실행 여부: <!-- 실행함 / 실행하지 않음 / N/A -->
- 실행 명령:

```text

```

## 변경

- 변경 파일:
- validation: <!-- 예: primary 2019–2023→2024 + secondary expanding / 변경 없음 / N/A -->
- metric: <!-- 예: brier_skill_score (raw BSS 비교) / 변경 없음 / N/A -->
- seed: <!-- 예: 42 / N/A -->

## 결과

- CV / fold 결과: <!-- mean±std, fold raw BSS / N/A -->
- 요약 결과:
- leakage 검토:
- artifact / log 위치: <!-- 예: outputs/experiments.csv, outputs/predictions/EXP-00x/ / N/A -->

## Tests

- [ ] `python -m unittest discover -s tests -v` 실행함
- [ ] 이 PR은 테스트가 필요 없는 문서/설정 변경이다

## 체크

- [ ] `main`에 직접 작업/push하지 않았다
- [ ] raw data / 대형 artifact를 commit하지 않았다
- [ ] 실험이면 `outputs/experiments.csv`에 기록했다 (실패 포함)
- [ ] 실행하지 않은 결과를 실행했다고 쓰지 않았다
- [ ] 공통 validation / metric / seed를 바꾸면 기존 EXP와 직접 비교하지 않는다

## 미검증 항목

-
