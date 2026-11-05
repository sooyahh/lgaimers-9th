# EXPERIMENT-LINE-MISMATCH-001 — 실험 라인과 제출 라인의 불일치

- 작성: 2026-08-19 (v2 — §1.2 표현 정정, §3.2 확인 완료)
- 성격: 기존 artifact 재해석. 새로 학습한 모델 없음.
- 출처: `outputs/eda/unseen_id_experiments.md`, `outputs/eda/full_train_experiments.md`, `outputs/eda/model_transfer_audit_001.md`, `outputs/experiments.csv`
- 결론 성격: `results_only`

---

## 1. 불일치의 실체

| | feature | 학습 창 | 로컬 2024 raw BSS |
|---|---|---|---:|
| **실험 비교 baseline (EXP-004)** | 44 (ID 포함) | 2019–2023 | **578.18** |
| **제출 라인 (EXP-005)** | 42 (L, ID 제거) | 2019–**2024** | **739.97** (2019–2023 학습 시) |

FE/TM 계열 실험 6건(FE-001, FE-001A, FE-002, FE-002A, TM-002A, TM-003A)이 전부 **EXP-004(44 feat) 위에서** 판정됐다.

### 1.1 축 1 — feature set (162점)

`outputs/eda/unseen_id_experiments.md`:

| 구성 | 2023 raw BSS | 2024 raw BSS |
|---|---:|---:|
| EXP-004 (44 feat, ID 포함) | −1366.48 | 578.18 |
| `L_drop_ids` (42 feat) | −1306.56 (**+59.92**) | 739.97 (**+161.79**) |

`pitcher_id` / `batter_id` 제거 단독으로 **+161.79**. 이후 모든 FE/TM 실험이 노린 +5 ~ +19의 8배 이상이다.

근거: 2024 holdout 기준 unseen 비율 — `pitcher_id` **19.86%**, `batter_id` 9.30%. 2023 기준 각 13.80% / 9.40%.
2025는 더 높을 것으로 예상된다.

참고 (`model_transfer_audit_001.md` §1, L42 BASE):

| fold | catboost | lightgbm |
|---|---:|---:|
| 2022 | 2321.31 | 2224.49 |
| 2023 | −1308.95 | −1579.99 |
| 2024 | 732.36 | 544.20 |

L42 3-fold mean = **581.57** (EXP-004 490.37 대비 **+91.20**).

### 1.2 축 2 — 학습 창 (미측정, 매우 큼)

제출 모델(EXP-005)은 2019–2024 full-train인데 두 실험 라인 모두 2019–2023이다.
LB 앵커 분석에 따르면 **이 축의 효과가 feature 축보다 클 가능성이 높다.**
상세: `reports/lb_anchor_analysis_001.md` §3.

> **표현 주의:** local primary(`≤2023 → 2024`)와 final submit(`≤2024 → 2025`)은 **구조가 동일하다**
> (직전 시즌까지 학습 → 다음 시즌 예측). primary CV는 제출의 올바른 대리이며 폐기 대상이 아니다.
> 차이는 local이 가장 최신 regime을 학습에 넣을 수 없다는 것뿐이다.

---

## 2. 귀결

1. **TM-003A(유일한 ADOPT)는 44 feat 위에서만 검증됐다.** L42에서 +19.3이 재현되는지 미확인.
   `MODEL-TRANSFER-AUDIT-001`이 L42로 재검증한 것은 FE-001A / FE-002 / FE-002A뿐이며 전부 INCONCLUSIVE_NOISE였다.
2. 6건의 FE/TM 판정은 **제출 라인보다 162점 낮고 학습 창도 다른 baseline** 위에서 내려졌다. 판정 자체가 무효는 아니지만, 제출 라인으로의 전이는 별도 검증 사안이다.
3. 향후 실험은 제출 라인(L42, 그리고 가능하면 학습 창까지 맞춘 설정) 위에서 수행하는 것이 맞다.

---

## 3. 부수 관측 — 2023 fold 붕괴의 원인 후보

`outputs/eda/unseen_id_experiments.md` 실험 `O_drop_game_type`:

| | 2023 raw BSS | 2024 raw BSS |
|---|---:|---:|
| EXP-004 (`game_type` 유지) | −1366.48 | 578.18 |
| `game_type` 제거 | **−384.12 (+982.36)** | 464.49 (**−113.69**) |

`game_type × season` target rate (동일 문서, `reversal_confirmed: True`, sign-flip 유형 `['F','R']`):

| game_type | 2019 | 2020 | 2021 | 2022 | **2023** | 2024 |
|---|---:|---:|---:|---:|---:|---:|
| `F` | 0.6892 | 0.5878 | 0.7038 | 0.7087 | **0.4729** | 0.4593 |
| `R` | 0.5495 | 0.5269 | 0.5128 | 0.5037 | 0.5031 | 0.4897 |

**관측:** `game_type=F`의 성공률이 2022→2023에 0.709 → 0.473으로 붕괴한다.
2023 fold의 −1366은 이 체제 전환을 학습 데이터가 커버하지 못한 결과일 가능성이 높다.
2024는 이미 전환 후 수준(0.459)이고, `game_type` 유지가 2024에서 +113.69 유리하므로 제출 라인의 선택은 타당하다.

### 3.1 하지만 "2023을 판정에서 제외한다"는 결론은 아직 내릴 수 없다

- 2025 target은 비공개다. **2025가 2024와 같은 체제라는 근거가 없다.**
- LB implied Brier(0.2476)가 2024 holdout Brier(0.2480)보다 낮다는 것은 2025에 심각한 drift가 없다는 **약한 신호**일 뿐이다.
- **post-hoc rule change 금지:** TM-003A는 결과를 보기 전에 판정 규칙을 잠갔다(`frozen pre-run rule`, `predeclared_rule_bucket`). 결과를 본 뒤 2023 조건을 빼고 과거 판정을 다시 매기면 저장소의 규율 전체가 무효화된다. Phase 3 코드 검증을 앞두고 실질 리스크다.

**허용되는 절차**

```
VAL-SHIFT-2023-001 실행
  ↓
validation 문제 확인
  ↓
새 validation policy 작성
  ↓
날짜 + commit과 함께 freeze
  ↓
그 이후 실험부터 적용 (과거 판정 소급 변경 없음)
```

과거 REJECT(예: TM-002A)를 다시 보고 싶다면 재판정이 아니라 **새 experiment_id로 재실행**한다.

### 3.2 사전 규칙 원문 확인 (완료)

`exp/tm-003a-expected-ivb:reports/next_experiment_decision.md` §7.11 ADOPT/REJECT 규칙 원문:

```
ADOPT — all of:
  - CV mean raw BSS improves vs EXP-004
  - no meaningful worst-fold degradation
  - specifically, no 2023-fold degradation      <-- 존재 확인
  - leakage / temporal-safety checks PASS
  - the feature is not effectively a duplicate signal ...

REJECT — any of:
  - CV mean raw BSS worsens vs EXP-004
  - 2023 fold or the worst fold clearly degrades  <-- 존재 확인
  ...
```

**2023 조항이 사전 규칙에 명시적으로 존재했음이 확인됐다.**
따라서 §3.1의 post-hoc rule change 금지 원칙은 **확정**이며, 결과를 본 뒤 2023 조건을 제거해 과거 판정을 다시 매기는 것은 허용되지 않는다.

---

## 4. 현 순위에서의 우선순위

이 문서의 §3은 진단 가치가 높지만 **점수 기여가 0**이다.
656/1090등에서는 판정 정확도보다 탐색 범위가 병목이므로, `VAL-SHIFT-2023-001` canonical 실행은 Phase C로 미룬다.
단 §3.1의 post-hoc 금지 원칙은 지금부터 계속 적용한다.
