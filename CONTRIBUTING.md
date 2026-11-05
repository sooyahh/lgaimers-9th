# Contributing

이 저장소는 LG Aimers Phase 2 실험을 팀으로 수행하기 위한 공용 코드다.
`main`에는 검증된 공용 코드만 두고, 실험은 별도 branch와 Pull Request로 합친다.

현재 GitHub 저장소(private, 무료 플랜)에서는 `main` branch protection을 켤 수 없다.
아래 규칙은 **팀 운영 규칙**이다. GitHub가 직접 push를 막아주지는 않는다.

자세한 과제·실행 명령은 `README.md`를 본다.

## 작업 시작

1. `main`을 최신화한다.
   ```powershell
   git checkout main
   git pull origin main
   ```
2. 기존 Issue, `outputs/experiments.csv`, `docs/GPT_HANDOFF.md`에서 목적과 사용 중인 `experiment_id`를 확인한다.
3. `main`에서 바로 작업하지 않고 별도 branch를 만든다.

## Branch naming

무엇을 하는 branch인지 드러나게 한다. 사람 이름만으로 branch를 만들지 않는다.

권장:

- `exp/<topic>` — 실험 (예: `exp/exp007-catboost-depth`)
- `feat/<topic>` — 공용 기능
- `fix/<topic>` — 버그 수정
- `docs/<topic>` — 문서
- `chore/<topic>` — 도구·설정

## 실험 규칙

- 하나의 실험에서 핵심 가설은 **하나만** 변경한다.
- `experiment_id`는 팀 전체 **단일 namespace**다.
  - 공식 모델 실험: `EXP-000`, `EXP-001A`, `EXP-002`, … (이미 사용: `EXP-000`~`EXP-006`)
  - 다음 공식 ID: **`EXP-007`**
  - 검증·진단: `VAL-...` (예: `VAL-AUDIT-001`, `VAL-EXP006-SECONDARY-001`)
  - `python -m src.train`이 만드는 `temporal_YYYYMMDD_...`는 로컬 smoke ID이며 공식 EXP 번호가 아니다.
  - 사용 중인 ID를 재쓰지 않는다. PR에서 예약한다.
- 로그에 **author**를 남긴다. `append_experiment_log()`가 git `user.name` / `GIT_AUTHOR_NAME`을 채울 수 있다. 비어 있으면 수동으로 적는다.
- **seed**를 기록한다. 공통 기본값은 `configs/baseline.yaml`의 `runtime.seed` (42)다. 다른 seed를 쓰면 그 사실을 명시한다.
- **validation**을 기록한다. 기본은 `reports/validation_policy.md` (primary 2019–2023→2024, secondary expanding 3-fold).
- **metric**을 기록한다. 공식 비교는 Brier Skill Score. 모델 비교는 raw BSS / Brier를 우선한다.
- **실제 실행 여부**를 기록한다. 실행하지 않은 결과를 실행했다고 쓰지 않는다.
- 실패 실험도 `outputs/experiments.csv`에 남긴다 (`status`, `error`).
- CV 평균만 쓰지 말고 fold 결과와 변동성(`cv_std`, fold raw BSS)을 남긴다. 정책 요구 항목은 `reports/validation_policy.md`.
- Public LB만으로 채택/폐기하지 않는다. Public LB가 비어 있으면 미기록으로 둔다.
- 기존 validation 또는 metric을 바꾸면 그 실험은 기존 EXP와 **직접 비교하지 않는다**. 비교하려면 동일 validation/metric으로 다시 측정한다.

로그 위치: `outputs/experiments.csv`
스키마는 `src/utils.py`의 `EXPERIMENT_COLUMNS`다. provenance 컬럼: `author`, `git_branch`, `git_commit`, `git_dirty`.

- 공식 성능 비교용 EXP는 가능하면 clean committed state에서 실행한다.
- `git_dirty=true` 실험은 재현성 경고 대상으로 취급한다. 기록된 commit과 실행 코드가 다를 수 있다.
- 채택 후보라면 clean commit에서 재실행하거나 provenance를 명확히 남긴다.

## 금지

- `main`에서 직접 작업하거나 `main`에 직접 push
- `data/raw/` 대회 CSV 수정
- 다른 사람 artifact 덮어쓰기 (`outputs/models/`, `outputs/predictions/`, `outputs/submissions/`의 기존 EXP 디렉터리)
- 로그 없는 실험
- 실행하지 않은 결과를 실행했다고 기록
- raw data, 모델 가중치, 제출 zip, row-level OOF 등 대형 artifact 무단 commit
- 여러 핵심 가설을 한 실험에서 동시에 변경
- 공통 validation / metric / seed lock을 PR 설명 없이 변경

## Pull Request

실험이 아니면 해당 항목에 `N/A`를 적는다. 템플릿: `.github/pull_request_template.md`.

PR에는 최소:

- 목적
- 핵심 가설
- `experiment_id` (실험인 경우)
- 변경 파일
- validation
- metric
- seed
- 실행 명령
- 실제 실행 여부
- 결과
- leakage 위험
- artifact / log path
- 미검증 항목

병합 전:

```powershell
python -m unittest discover -s tests -v
```

실행 명령의 정확한 형태는 `README.md`를 따른다. `src` import가 필요한 스크립트는 `python -m scripts.<module>`이다.
