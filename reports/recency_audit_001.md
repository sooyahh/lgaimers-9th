# RECENCY-AUDIT-001

- generated_at: `2026-08-19T20:00:53`
- smoke: `false`
- seed: `42`
- weight_policy: `uniform_1.0`
- feature_set: L42 (`42` features; drop `['pitcher_id', 'batter_id']`; keep `game_type`)
- CatBoost params: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`
- decision: `results_only_no_adopt_reject` (window-policy candidate only)

## Objective

오래된 시즌 데이터를 expanding window에 계속 포함하는 것이 다음 시즌 일반화에
도움이 되는지, 아니면 방해가 되는지 **uniform sample weight (1.0)** 조건에서 측정한다.

## Experimental contract

- 3 valid seasons × 3 windows = 9 CatBoost L42 fits
- 바뀌는 것: training season window, 그리고 모든 학습 행 weight = 1.0
- 바뀌지 않는 것: L42 feature spec, CatBoost hyperparameters, seed 42, metric, preprocessing
- Public LB / 신규 submission 없음
- WEIGHT-LAG-001은 이번 작업 범위 밖

## Canonical L42 controls

- backend: `catboost`
- params lock: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`
- seed lock: `42`
- dropped IDs: `['pitcher_id', 'batter_id']`
- `game_type` kept as categorical
- missing numeric: fold-local median (`prepare_native_frames`)
- categorical: CatBoost native string/`__NA__`
- metric: `src.evaluate.evaluate_predictions` → `src.metrics.compute_brier_metrics`
- comparison metric: **raw BSS** (official clipped BSS is recorded only)

## Uniform-weight scope

> 이 결과는 uniform weighting 조건의 training-window 비교이며,
> 현재 weighted canonical L42의 직접 ablation으로 과대해석하지 않는다.

기존 `SEASON_WEIGHTS = {2019: 1.0, 2020: 1.2, 2021: 1.4, 2022: 1.6, 2023: 2.0}` 는
이번 9 fit에 사용하지 않는다. 구현 자체는 삭제하지 않았다.

## Fold definitions

| valid | A_gap | B_expanding | C_shifted |
|---|---|---|---|
| 2022 | [2019, 2020] | [2019, 2020, 2021] | [2020, 2021] |
| 2023 | [2019, 2020, 2021] | [2019, 2020, 2021, 2022] | [2020, 2021, 2022] |
| 2024 | [2019, 2020, 2021, 2022] | [2019, 2020, 2021, 2022, 2023] | [2020, 2021, 2022, 2023] |

## Fit-level results

| valid | condition | train seasons | train rows | valid rows | raw BSS | Brier | ref Brier | fit_s |
|---|---|---|---|---|---|---|---|---|
| 2022 | A_gap | `2019,2020` | 481500 | 247472 | 1610.805855 | 0.24515007 | 0.24916361 | 69.6 |
| 2022 | B_expanding | `2019,2020,2021` | 728588 | 247472 | 2322.459708 | 0.24337688 | 0.24916361 | 97.5 |
| 2022 | C_shifted | `2020,2021` | 491175 | 247472 | 2296.891287 | 0.24344059 | 0.24916361 | 72.5 |
| 2023 | A_gap | `2019,2020,2021` | 728588 | 245525 | -1061.040781 | 0.25265260 | 0.25000000 | 99.2 |
| 2023 | B_expanding | `2019,2020,2021,2022` | 976060 | 245525 | -1261.539976 | 0.25315385 | 0.25000000 | 125.8 |
| 2023 | C_shifted | `2020,2021,2022` | 738647 | 245525 | -1342.044864 | 0.25335511 | 0.25000000 | 110.1 |
| 2024 | A_gap | `2019,2020,2021,2022` | 976060 | 253507 | -1623.296291 | 0.25386203 | 0.24980693 | 130.9 |
| 2024 | B_expanding | `2019,2020,2021,2022,2023` | 1221585 | 253507 | 734.819045 | 0.24797130 | 0.24980693 | 165.1 |
| 2024 | C_shifted | `2020,2021,2022,2023` | 984172 | 253507 | 724.992243 | 0.24799585 | 0.24980693 | 178.0 |

### Compact raw BSS

| valid | A_gap | B_expanding | C_shifted |
|---|---|---|---|
| 2022 | 1610.805855 | 2322.459708 | 2296.891287 |
| 2023 | -1061.040781 | -1261.539976 | -1342.044864 |
| 2024 | -1623.296291 | 734.819045 | 724.992243 |

## Paired contrasts

정의:

- `B-A` = expanding_history_effect (가장 최근 train season 하나를 기존 history에 추가)
- `C-A` = window_shift_effect (같은 시즌 수에서 window를 한 해 앞으로 이동).
  **순수 recency 효과라고 부르지 않는다.**
- `B-C` = old_history_inclusion_effect (최신 window에 가장 오래된 시즌 하나를 추가). **PRIMARY**

| valid | B-A | C-A | B-C | B-C sign |
|---|---|---|---|---|
| 2022 | 711.653853 | 686.085432 | 25.568421 | positive |
| 2023 | -200.499194 | -281.004083 | 80.504888 | positive |
| 2024 | 2358.115336 | 2348.288534 | 9.826801 | positive |
| mean | 956.423331 | 917.789961 | 38.633370 | |
| std (ddof=1) | 1296.750227 | 1329.872234 | 37.106168 | |
| worst (min) | -200.499194 | -281.004083 | 9.826801 | |
| best (max) | 2358.115336 | 2348.288534 | 80.504888 | |

항등식 `B-A = (C-A) + (B-C)` 는 세 fold 모두 floating-point tolerance `1e-09` 안에서 성립했다.

## Primary contrast

`old_history_inclusion_effect = B - C`

- `D > 0`: 오래된 시즌 포함이 유리
- `D < 0`: 오래된 시즌 제외가 유리

- 2022: `25.568421` (positive)
- 2023: `80.504888` (positive)
- 2024: `9.826801` (positive)
- mean: `38.633370`

## Fold stability

- B-C mean: `38.633370`
- B-C std: `37.106168`
- B-C worst: `9.826801`
- B-C best: `80.504888`

### 2023 behavior

- 2023 B-C: `80.504888` (positive)
- 2022/2024와 부호가 같은가: `true` (2022=positive, 2024=positive)
- B-C mean including 2023: `38.633370`
- B-C mean excluding 2023: `17.697611`
- mean shift from including 2023: `20.935759`
- B-C std including 2023: `37.106168`
- B-C std excluding 2023: `11.131006`

2023 fold는 제외하지 않았다. 위 숫자는 해석용 보조 통계다.

## Predeclared-rule result

**`KEEP_FULL_HISTORY_CANDIDATE`**

이 판정은 최종 canonical 모델 ADOPT/REJECT가 아니다.
uniform-weight 조건에서의 training-window 정책 후보 판정이다.

### SHORTEN_WINDOW_CANDIDATE checks

- `mean_B_minus_C_lt_0`: `False`
- `at_least_two_folds_negative`: `False`
- `C_worst_not_below_B_worst`: `False`

### KEEP_FULL_HISTORY_CANDIDATE checks

- `mean_B_minus_C_gt_0`: `True`
- `at_least_two_folds_positive`: `True`
- `B_worst_not_below_C_worst`: `True`

- mean(B-C): `38.633370`
- negative folds: `0/3`
- positive folds: `3/3`
- B worst-fold raw BSS: `-1261.539976`
- C worst-fold raw BSS: `-1342.044864`
- B-C signs: `{"2022": "positive", "2023": "positive", "2024": "positive"}`

핵심 결론 (이 결과의 해석 상한):

> Uniform-weight L42 조건에서는 오래된 시즌을 제거해 recent window로 단축할 근거가 없으며, B-C가 2022/2023/2024에서 모두 양수이므로 full-history 유지가 training-window policy candidate이다.

이 문장은 canonical model ADOPT가 아니고, weighted L42 우열 비교도 아니며, 2023 shift 해소나 '오래된 시즌은 항상 유익하다'는 일반화도 아니다.

## Limitations

- single seed 42
- uniform weighting experiment; current weighted canonical L42와 직접 같지 않음
- 2024 uniform B와 weighted canonical L42 raw BSS 차이는 이번 audit의 인과 비교가 아니다
- Public LB를 사용하지 않음
- CatBoost L42에서의 결과
- 다른 backend / feature-set / Trackman feature에 자동 일반화 금지
- 2023 regime shift를 포함한 3-fold이며, fold 효과를 평균으로만 읽으면 안 됨
- `load_competition_data`가 test.csv를 같이 열지만, 이 audit은 train만 사용한다. test로 aggregate/encoding/통계를 만들지 않았다

## Next step

RECENCY-AUDIT-001이 main에 반영된 뒤, 최신 origin/main에서 새 브랜치를 만들어 WEIGHT-LAG-001을 시작한다.
WEIGHT-LAG-001은 이번 작업에서 구현하지 않았다.

## Commands actually run

- `python -m unittest tests.test_recency_audit_001 -v`
- `python -m scripts.run_recency_audit_001 --dry-run`
- `python -m scripts.run_recency_audit_001 --smoke`
- `python -m scripts.run_recency_audit_001`

## experiments.csv

이번 audit는 `outputs/experiments.csv`에 append하지 않았다.
VAL-AUDIT-001 / MODEL-TRANSFER-AUDIT-001과 같은 진단 convention이다.
