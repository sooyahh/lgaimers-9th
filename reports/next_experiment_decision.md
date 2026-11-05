# Next Experiment Decision

Prepared from FEATURE-STABILITY-AUDIT, ERR-AUDIT-001, and TM-AUDIT-001B.
No new model was trained. FE/TM candidates below are **not implemented**.

Error-analysis baseline: **EXP-004** CatBoost (season weights), expanding 3-fold OOF 2022/2023/2024.
EXP-006 was **not** used as champion (`is_comparable_cv=false`, secondary verdict **hold**, 2024 is the Optuna tune fold).

**Document layers:** §§1–5 below are the original post-audit plan and are left unchanged. Later execution status (what actually ran vs that plan) is in §6. Do not read FE-001A / FE-002A as completion of FE-001 / FE-002.

---

## 1. Feature Stability 핵심 결과

### 안정적인 신호

- Target-level **count ranks** (`balls_before` as a category) stay consistent across seasons (rank Spearman ~0.84).
- `asof_pitcher_prev3_game_success_rate` and `prev5` success rates keep a **positive** Pearson in every season (mean |r| ~0.07). They still have ~5% NA (pitcher-season cold start).
- `asof_pitcher_success_rate` / `reverse_rate` stay sign-consistent (positive / negative) but **magnitude is UNSTABLE** and weaker in 2023.

### 불안정한 신호

- **SIGN_FLIP:** `asof_pitcher_ball_rate`, `asof_pitcher_strike_rate`, `asof_pitcher_fastball_rate`, plus weak flips on `asof_pitcher_n` / `pitchmix_n`.
- Batter asof success/middle/n: UNSTABLE; 2023 |r| collapses vs 2022.
- `prev1` success: same sign, large magnitude swing (UNSTABLE).

### cold-start 관련 문제

- Rate columns are NA iff `*_n==0` (first observation of that pitcher/batter). Zero-n share is ~0.05–0.15% — structurally real, **tiny**.
- Prev-game asof missing ~2–5% by season (max 5.17%) — larger than n==0, and it is the start-of-proxy-season block.

### 시즌 shift 관련 문제

- Target mean: 2019 0.565 → 2022 0.529 → **2023 0.500** → 2024 0.486.
- 2022→2023 is the largest adjacent drop and coincides with Trackman mix/IVB/HB shifts (`fold_comparison.md`) and asof ball/strike sign flips.

---

## 2. Error Analysis 핵심 결과

Baseline OOF integrity: **PASS** (746,504 rows, unique `row_id`, join 0 unmatched, fold raw BSS matches logged EXP-004 exactly).

### 가장 큰 실패 패턴 TOP 3

| rank | pattern | n | share of OOF | raw BSS | FE로 건드릴 수 있는가 |
|---:|---|---:|---:|---:|---|
| 1 | **2023 entire fold** (overpredict: p=0.520 vs y=0.500) | 245,525 | 32.9% | −1366.5 | 부분만. 2000+ pitcher rows still −494. Not a small-slice bug. Do not retune validation to hide it. |
| 2 | **`prev1_<=0.35`** | 78,607 | 10.5% | −184.2 | **Yes, row-local.** Model moves p down (0.486) but not enough (y=0.463). |
| 3 | **2023 low pitcher history** (`asof_pitcher_n` 1–500) | ~25.9k in 2023 | ~3.5% of all OOF; ~10.5% of 2023 | −6892 (1–100), −5068 (101–500) | **Yes, partially.** n==0 is negligible (232 rows). Batter cold-start is **not** a failure. |

Other notable, not top-3: same-hand 1x1 (12.4%, BSS 346, gap +0.024); count 0-2 (6.2%, BSS 78). High LI and innings 7–9 are **not** primary failures. Extra innings 10+ are weak but only 1.6%.

Predictions never explode to 0/1 (OOF range ~0.33–0.80). Small-n failure is **overprediction**, not degenerate probs.

---

## 3. Trackman 핵심 결과

| item | result |
|---|---|
| Direct pitcher/batter ID overlap | **0 / 0** (reproduced) |
| TM rows / seasons | 1,793,078 / 2019–2024 |
| Hand coding | main `{1,2}` vs TM `{Left, Right}` — **no official map** |
| L1/L2/L3 coverage | **not joinable** without inventing a map |
| L0 count-only coverage | **1.000** on 2022/2023/2024, fallback 0, large group n |
| 2025 main coverage | **not available** (full test not used; local test.csv is a 5-row sample) |

### TM-001 verdict: **HOLD**

Specified hand×count history cannot be built without a hand-code mapping that does not exist in the repo. Count-only TM aggregates are feasible and leakage-safe (`season < y`) but are a **different** experiment and largely duplicate count already in main.

### TM-002 verdict: **PASS**

Prior-season TM group physics (fastball/breaking/offspeed n ≫ 50 in every window) × current-row asof mix is constructible. Mix NA rate ~0.054%. Fold-safe if TM seasons `<` validation year. Expected gain **low–medium** because of collinearity with mix rates; the extra information is *era physics*, which did shift (IVB/HB).

| | TM-001 | TM-002 |
|---|---|---|
| leakage risk | low **if** prior-season only; L1 blocked by hand codes | low **if** prior-season only + official asof mix |
| 구현 비용 | medium (and currently blocked) | medium |
| 우선순위 | HOLD | NEXT (not simultaneous with a main-table FE) |

---

## 4. 다음 실험 후보

Only candidates with data support. Validation stays locked (`reports/validation_policy.md`). Metric stays `src/metrics.py`. Compare on EXP-004-style expanding folds (do not use EXP-006 2024 tune fold as selection).

### FE-001

| 항목 | 내용 |
| --- | --- |
| 가설 | Treating count as a single categorical `count_code=balls-strikes` captures non-monotonic count effects that numeric `balls_before`/`strikes_before` underfit (0-2 is the weakest large count cell). |
| 근거 | ERR-AUDIT count table: 0-2 raw BSS 78 (n=46,276). FEATURE-STABILITY: `balls_before` category ranks STABLE. EXP-004 lists count columns as numeric, not in `categorical_cols`. |
| 변경되는 핵심 요소 | Add **one** CatBoost categorical `count_code`. Do not drop numeric count columns in the same run. |
| validation | 기존 방식 유지 (primary 2024 + secondary 2022/2023) |
| leakage risk | low — same-row pre-pitch fields only |
| 예상 구현 비용 | low |
| 기대 정보가치 | medium (trees can already split the two numerics; this tests explicit combo) |
| 추천 | **NEXT** |

### FE-002

| 항목 | 내용 |
| --- | --- |
| 가설 | Extreme / thin `asof_pitcher_prev1_game_success_rate` overstates last-game noise. A **row-local** shrink toward the same row’s `asof_pitcher_success_rate`, with weight increasing in `asof_pitcher_n`, reduces overprediction on `prev1_<=0.35` and 2023 low-n pitchers. |
| 근거 | Worst large non-season slice: prev1≤0.35, 10.5% of OOF, raw BSS −184, gap +0.023. prev1 is UNSTABLE. 2023 1–500 asof_n is severely negative. Uses only official asof already on the row (no other-row means, no test.csv). |
| 변경되는 핵심 요소 | Replace `asof_pitcher_prev1_game_success_rate` with **one** shrunk column (same name or a single substitute). Do not also change prev3/prev5 in this run. |
| validation | 기존 방식 유지 |
| leakage risk | **low** if and only if the formula is a function of the current row’s asof fields. **medium/high** if anyone computes a fold-wide or test-wide target mean. |
| 예상 구현 비용 | low |
| 기대 정보가치 | medium — directly hits the largest actionable error slice; will **not** fix 2023 2000+ by itself |
| 추천 | **NOW** |

### TM-001

| 항목 | 내용 |
| --- | --- |
| 가설 | Past-season Trackman aggregated on hand×count context transfers to main without player IDs. |
| 근거 | Context columns exist on TM, L0 count-only coverage=1.0, but L1 hand keys **do not match** (`1/2` vs `Left/Right`). ID overlap=0 (reproduced). |
| 변경되는 핵심 요소 | Would be “add TM context aggregates” — blocked as specified. |
| validation | 기존 방식 유지 |
| leakage risk | low if `TM.season < valid_season`; forbidden: current-pitch TM, 2025 TM, test aggregates, invented hand maps |
| 예상 구현 비용 | medium |
| 기대 정보가치 | low until a documented hand map exists; L0 would overlap count |
| 추천 | **HOLD** |

### TM-002

| 항목 | 내용 |
| --- | --- |
| 가설 | League physics given a pitcher’s **asof mix** (`fastball/breaking/offspeed_rate ×` prior-season TM group medians) adds 2023-regime information that mix rates alone lack. |
| 근거 | TM-AUDIT-001B PASS: group n is huge; mix NA ~0.05%; IVB/HB shifted 2022→2023. High collinearity with existing mix features. |
| 변경되는 핵심 요소 | Add **one** expected-physics family (start with a single `tm_expected_ivb` or `tm_expected_speed`, not the full 28-col dump). |
| validation | 기존 방식 유지; rebuild TM profiles per fold from `season < valid_year` |
| leakage risk | low with prior-season filter; never use current-pitch TM |
| 예상 구현 비용 | medium |
| 기대 정보가치 | medium for 2023-style physics shift; low if physics is a linear remix of mix |
| 추천 | **NEXT** (after FE-002; do not stack with FE-002 in one run) |

---

## 5. 최우선 단일 실험

**추천: FE-002 (row-local prev1 shrinkage).**

선정 기준 적용:

1. **2025 일반화:** Hits a slice that is not 2024-specific (10.5% of 2022–2024 OOF) and the 2023 low-history overprediction mode, without betting on SIGN_FLIP ball/strike rates or an unofficial hand map.
2. **Temporal validation:** Measurable on the existing EXP-004 folds (especially secondary_2023 + the `prev1_<=0.35` slice).
3. **Leakage:** Can be implemented as a pure function of columns already on that row. No test.csv, no other-row target encoding, no current-pitch Trackman.
4. **Error pattern:** Directly the #2 failure (largest **actionable** slice). #1 (all of 2023) cannot be “solved” by one FE; this still attacks the part of 2023 that is worst (low n) without pretending to fix 2000+ regime shift.
5. **시간 대비 효과:** Low implementation cost vs TM profile plumbing.
6. **한 가설:** Only prev1 is changed. prev3/prev5, count encoding, and TM stay frozen.

**Do not implement FE-001 / TM-001 / TM-002 in the same run.**
**Do not implement FE-002 until the user explicitly starts that experiment.**

If FE-002 does not move 2023 raw BSS or the prev1≤0.35 slice, the next single experiment should be **FE-001** (cheaper than TM-002) unless the review prefers TM-002 for the 2023 physics-regime hypothesis.

### Explicit hypothesis statement (FE-002)

> Replacing `asof_pitcher_prev1_game_success_rate` with a row-local shrink
> `w(n) * prev1 + (1 − w(n)) * asof_pitcher_success_rate`
> (NA prev1 → fall back to `asof_pitcher_success_rate`; `w` increasing in `asof_pitcher_n`)
> improves EXP-004-style raw BSS on the `prev1_<=0.35` slice and does not degrade primary 2024 or secondary 2022, under the locked season holdouts and unchanged official metric.

Suggested `w(n)` (to be frozen before the run, not tuned on 2024): e.g. `w = n / (n + k)` with a **predeclared** `k` (not Optuna’d in the same experiment).

---

## 6. Plan vs execution status (addendum)

This section does **not** replace §§4–5. Those remain the original recommendation: **FE-001** = add categorical `count_code`; **FE-002** = row-local prev1 shrinkage.

### 6.1 Original plan (from §§4–5; not rewritten)

| ID | Planned change | Status |
|---|---|---|
| FE-001 | Add one CatBoost categorical `count_code=balls-strikes`. Do not drop numeric count columns in the same run. | **미실행** |
| FE-002 | Replace `asof_pitcher_prev1_game_success_rate` with one row-local shrink toward the same row’s `asof_pitcher_success_rate` (weight increasing in `asof_pitcher_n`). Do not change prev3/prev5 in that run. | **미실행** |

No `configs/fe001.yaml` / `configs/fe002.yaml` and no `experiment_id` `FE-001` or `FE-002` row in `outputs/experiments.csv`.

### 6.2 What actually ran (different experiment IDs)

These are **not** FE-001 or FE-002.

| ID | Actual change | Sources | Verdict |
|---|---|---|---|
| FE-001A | Drop only `asof_pitcher_ball_rate` (44→43). Not `count_code`. | `configs/fe001a.yaml`; `outputs/predictions/FE-001A/comparison_vs_exp004.json`; `reports/fe_001a_ball_rate_ablation.md` §10; `outputs/experiments.csv` (`verdict` in notes) | **REJECT** |
| FE-002A | Drop only `asof_pitcher_prev1_game_success_rate` (44→43). No shrinkage / smoothing column. | `configs/fe002a.yaml`; `outputs/predictions/FE-002A/comparison_vs_exp004.json`; `reports/fe_002a_prev1_ablation.md` §14; `outputs/experiments.csv` (`verdict` in notes) | **REJECT** |

Artifact `verdict` strings (copied, not re-derived):

- FE-001A `comparison_vs_exp004.json`: `"verdict": "REJECT"` — `"2023 did not improve in a material way; mean and worst fold did not improve."`
- FE-002A `comparison_vs_exp004.json`: `"verdict": "REJECT"` — `"Removing prev1 did not improve CV mean or the prev1<=0.35 slice; 2023 fold change is not a material recovery."`

### 6.3 Plan vs execution mismatch

- FE-001A ≠ FE-001 (`count_code` add vs `asof_pitcher_ball_rate` drop).
- FE-002A ≠ FE-002 (prev1 shrinkage vs prev1 drop).
- Therefore FE-001 and FE-002 are **not** recorded as executed or completed.
- FE-001A / FE-002A REJECT does not decide the still-unrun FE-001 / FE-002 designs.

### 6.4 Why the plan changed

**reason unknown / 기록 없음** in this repository: this file’s original recommendation was FE-002 shrinkage then FE-001 `count_code`; no in-repo decision note explains why FE-001A / FE-002A ablations were run instead. Do not infer a reason from the A-experiment hypotheses.

---

## 7. TM-003A pre-run decision

**Status of this section: pre-run.** At the time this section was written, **TM-003A has not been implemented, trained, or cross-validated.** No `tm_expected_ivb_by_count` column exists, no config exists, no CV numbers exist. Nothing below is a result; every number below is a *pre-declared control*, not an observation. Sections §§1–6 above are unchanged.

**Why this section exists.** §6.3/§6.4 record a plan-vs-execution mismatch (FE-001A ≠ FE-001, FE-002A ≠ FE-002) whose reason is `기록 없음`. This section exists so that mismatch cannot repeat silently: the hypothesis, the exact feature definition, the fallback constant, the controls, the diagnostics, and the ADOPT/REJECT/HOLD rule are all frozen **before** the run. If what actually executes differs from §7, the difference must be recorded here as an explicit amendment, with a reason, before the result is interpreted.

### 7.1 Decision

**The next single experiment is `TM-003A` — count-conditioned expected IVB.** One experiment, one feature.

Chain that produced this decision:

`TM-002A` → **REJECT** → `TM-AUDIT-002` → contextual Trackman signal found → `TM-003A`

Rationale:

1. TM-002A (`tm_expected_rel_speed` = global prior-season pitch-group mean × as-of mix) was REJECT: CV mean raw BSS ≈ **−1.67** vs EXP-004, and the 2023 fold ≈ **−25.06** (`reports/tm_002a_expected_velocity.md` §8, §15). That failure is evidence against **global pitch-group physics × pitch-mix**, not against Trackman as a source.
2. TM-AUDIT-002 verdict is **`CONTEXTUAL_TM_PROMISING`** (`reports/trackman_audit_002.md` §16): Trackman physics differences concentrate in **count** context; `count × breaking × IVB` and `count × fastball × rel_speed` are the only two combinations that survive the stronger **non-overlapping independent-season** re-check (not just the overlapping expanding cutoffs).
3. `count × breaking × IVB` is the strongest and most stable of those (audit §14: effect_ratio 0.69, deviation up to 9.3 IVB units from the global mean, median pairwise Spearman 0.965 across independent seasons, 0.976 after sparse-cell removal).
4. TM-003A is the single feature hypothesis that follows **directly** from that audit finding — it is the audit's own nominated next experiment (`reports/trackman_audit_002.md` §17).

**This decision does not retire FE-001 or FE-002.** Both remain **미실행** exactly as recorded in §6.1, and both remain valid future candidates. TM-003A is chosen as the *next* experiment, not as a replacement for them.

Baseline for comparison is **EXP-004** (the current controlled-experiment baseline). **EXP-006 is not used as champion baseline**: 2024 is its Optuna tuning fold and its secondary validation verdict is **hold** (see the header of this file).

### 7.2 Hypothesis

> Combining a **prior-season Trackman `count × pitch_type_group` IVB profile** with the **current row's official as-of pitch mix** into exactly one new feature, added to EXP-004, improves generalization on the locked expanding temporal validation (2022 / 2023 / 2024) relative to EXP-004, which sees count and pitch-mix features but contains **no movement information at all**.

Honest framing, carried from `reports/trackman_audit_002.md` §13 and §17 and **not** softened here:

- This is **not** "completely new orthogonal information." It is a **handcrafted contextual feature that injects an external Trackman physics profile into the count × pitch-mix interaction** that CatBoost can already partially learn from features it holds.
- Redundancy risk is **MEDIUM** by pre-run assessment.
- The one thing the main table cannot supply is the **external physics magnitude** (main has no IVB).
- Expected BSS gain is **UNKNOWN**. This experiment is designed to be able to fail.
- The audit's count effect is an **observational association**, likely driven by pitch-subtype and pitcher composition inside the broad `breaking` group (audit §12.5). TM-003A injects *the profile*; it does not assert that count causes a pitch's IVB to change.

### 7.3 Exact feature definition

Exactly **one** new model feature is added. Nothing is removed.

**Name:** `tm_expected_ivb_by_count`

**Definition:**

```
tm_expected_ivb_by_count = Σ_g  asof_mix_g × IVB_hist(count, g)

g     ∈ {fastball, breaking, offspeed}          # `other` group excluded
count = f"{balls_before}-{strikes_before}"      # 12 legal counts, 0-0 … 3-2
```

where:

| term | definition |
|---|---|
| `asof_mix_g` | the current row's **official** as-of pitch-mix rate for group `g` (`asof_pitcher_fastball_rate` and its breaking / offspeed counterparts), as already used by EXP-004 / TM-002A. No re-derivation, no new mix estimator. |
| `IVB_hist(count, g)` | `mean(induced_vert_break)` over Trackman rows with that `count` and that `pitch_type_group`, computed **per fold** from Trackman seasons `season < valid_year` only. |
| `count` | built from the main row's `balls_before` / `strikes_before` — pre-pitch fields already on the row. |

Hard constraints on the definition:

- The **current pitch's** Trackman values are **never** used. `IVB_hist` is a historical aggregate profile, keyed only on `(count, pitch_type_group)`.
- The profile is **rebuilt independently for each fold** under that fold's cutoff. There is no single shared profile across folds.
- `pitch_type_group == other` is excluded from the profile, consistent with TM-AUDIT-002 §5 and TM-002A.
- Feature count goes **44 → 45**, matching the TM-002A pattern (one added column, none dropped).

*Implementation note (recorded, not a change to the definition):* the definition above is the plain weighted sum `Σ_g asof_mix_g × IVB_hist(count, g)`. TM-002A used the same numerator with a `Σ_g asof_mix_g` denominator as a numerical guard, and recorded that the observed as-of mix rates already sum to 1 up to float noise ~1e-6 (`reports/tm_002a_expected_velocity.md` §4). The two forms therefore coincide to float noise. Whichever form is implemented must be stated in the TM-003A report; it is not a free parameter to be chosen after seeing CV.

### 7.4 Pre-declared fallback

Fixed **before** the run and **before** any CV output is seen.

**Frozen constant: `min_n = 1000`.**

| condition | behaviour |
|---|---|
| prior-history `count × pitch_type_group` cell has `n ≥ 1000` | use that cell's count-conditioned mean IVB |
| prior-history `count × pitch_type_group` cell has `n < 1000` | **fall back** to that `pitch_type_group`'s **count-agnostic** mean IVB, computed from the **same** fold's prior-history window |
| `asof_pitcher_pitchmix_n == 0`, or a required as-of mix rate is missing | `tm_expected_ivb_by_count = NaN` |

Missing-value handling after that point is **fold-local only**, identical to EXP-004 / TM-002A (fold-local median impute + CatBoost native categoricals). No global impute, no cross-fold statistic, no test-informed impute.

Why `min_n = 1000`: it follows the **sparse threshold already used in TM-AUDIT-002** (§6: SPARSE = 1,000, a round value just below the p10 cell size of 6,809), and it is the audit's own recommended fallback shape (§17). The audit also established which cells this will actually bite: the thin cells are rare 3-ball breaking/offspeed counts (e.g. 3-0 breaking, n ≈ 102–276 per season), while the median cell for those same `count × breaking/offspeed` combinations is still 16k–24k.

**`min_n = 1000` is frozen.** It is not re-selected, swept, or tuned after seeing CV results. If the build-time cell counts turn out to make 1000 clearly unworkable, that is an **amendment recorded here with a reason before the CV result is read** — never a post-hoc adjustment.

### 7.5 Locked experiment controls

| item | locked value |
|---|---|
| Baseline | **EXP-004** (not EXP-006) |
| Validation | expanding, locked: 2019–2021 → **2022**; 2019–2022 → **2023**; 2019–2023 → **2024** |
| Trackman cutoff | per fold, strictly `trackman.season < valid_year` |
| Seed | **42**, single seed |
| Model | EXP-004 CatBoost recipe unchanged (native cats, iterations 300, lr 0.05, depth 6, Logloss) |
| Season weights | identical to EXP-004 (2019:1.0 … 2023:2.0) |
| Metric | existing `src/metrics.py` Brier Skill Score, unmodified |
| Missing handling | fold-local median impute + native cats, as EXP-004 / TM-002A |
| **The only change** | **+1 model feature: `tm_expected_ivb_by_count`** |

Explicitly **forbidden** in this run:

- removing or altering any existing EXP-004 feature
- any hyperparameter change
- any validation change
- any metric change
- any seed change
- any season-weight change
- adding any second Trackman feature to the model input (the diagnostic in §7.7 is **not** a model input)

### 7.6 Leakage guardrails

Forbidden without exception:

- current-pitch Trackman values
- 2025 Trackman
- any main ↔ Trackman 1:1 row join
- pitcher / batter ID guess-matching (ID overlap is 0/0; TM-AUDIT-001B)
- unofficial hand mapping (`1/2` vs `Left/Right`; no official map exists in-repo)
- any aggregate computed from inside `test.csv`
- any rolling / expanding statistic across test rows
- test-frequency features
- target encoding
- any Trackman season `≥` the fold's validation year

Positive requirements: `assert_trackman_temporal_safety` (or equivalent) must pass for every fold; only the official as-of mix already present on the row may be used as the mix source; EXP-004 / FE-001A / FE-002A / TM-002A artifacts must remain unmodified with fingerprints unchanged.

### 7.7 Falsification diagnostics

A diagnostic-only companion quantity is planned:

`tm_expected_ivb_global` = count-agnostic historical `pitch_type_group` IVB × as-of mix (i.e. the TM-002A structure applied to IVB).

**`tm_expected_ivb_global` is never a model input.** It exists only to test whether TM-003A is the TM-002A trap again.

To be reported after the run:

1. `corr(tm_expected_ivb_by_count, tm_expected_ivb_global)`
2. correlation of `tm_expected_ivb_by_count` with **each** as-of pitch-mix rate (fastball / breaking / offspeed)
3. **fallback usage rate** (share of rows served by the count-agnostic fallback under `min_n = 1000`)
4. **NaN rate** of `tm_expected_ivb_by_count` (pre-impute, valid rows)
5. prior-history `count × pitch_type_group` cell `n`: **min / p10 / median**

Purpose: determine whether the count-conditioned feature is effectively a **duplicate signal** of either the existing pitch-mix rates or its own count-agnostic version. TM-002A correlated **+0.98** with `asof_pitcher_fastball_rate` — a near-clone of an existing feature. Per the audit (§17), if TM-003A turns out to be a near-clone in the same way, that is grounds for **REJECT regardless of the CV number**.

### 7.8 Pre-declared decision rule

Fixed now; **not** revised after seeing results.

**ADOPT_CANDIDATE** — all of:

- CV mean raw BSS **improves** vs EXP-004
- **no meaningful worst-fold degradation**
- **specifically, no 2023-fold degradation**
- leakage / temporal-safety checks **PASS**
- the feature is **not** effectively a duplicate signal of `tm_expected_ivb_global` or of the existing as-of pitch-mix rates (§7.7)

**REJECT** — any of:

- CV mean raw BSS **worsens** vs EXP-004
- 2023 fold or the worst fold **clearly degrades**
- leakage / temporal-safety **failure**
- the feature is confirmed to be a near-complete **duplicate signal** (§7.7)

**HOLD** — any of:

- mean change is very small
- fold directions disagree
- results are unstable enough that adopt/reject cannot be responsibly decided

`ADOPT_CANDIDATE` means *candidate*: it authorizes recording TM-003A as a viable feature, **not** a Trackman bundle, not tuning, not a submission. A REJECT does not condemn Trackman generally, exactly as TM-002A's REJECT did not.

### 7.9 Scope

**In scope — exactly one thing:**

- TM-003A: add `tm_expected_ivb_by_count` to EXP-004, evaluate on the locked folds, report §7.7 diagnostics, apply the §7.8 rule.

**Out of scope for this run:**

- FE-001 (still **미실행**, not withdrawn)
- FE-002 (still **미실행**, not withdrawn)
- any additional Trackman feature (including `count × fastball × rel_speed`, the audit's secondary candidate)
- hyperparameter tuning
- multi-seed runs
- seed ensembling
- model ensembling
- stacking
- Public LB submission
- any audit of `exp/taewook-ensemble`

### 7.10 Status before execution

**`APPROVED_TO_RUN / NOT YET EXECUTED`**

As of the writing of this section:

- TM-003A model training has **not** been run.
- TM-003A cross-validation has **not** been run.
- No `tm_expected_ivb_by_count` feature has been built.
- No TM-003A config, script, prediction, or `outputs/experiments.csv` row exists.
- **No TM-003A result numbers exist anywhere in this repository, and none are estimated or predicted in this section.**

This section is a **pre-run decision record only**. When TM-003A does execute, its results belong in a separate TM-003A report; §7 above must not be retroactively edited to match the outcome. Any deviation from §§7.3–7.6 during implementation must be recorded here as a dated amendment with its reason, so that §6.4's `reason unknown / 기록 없음` does not recur.

---

## 7.11 Pre-run amendment — 2026-08-17

**This is the dated amendment that §7.10 requires.** It is written **before execution**: as of 2026-08-17, TM-003A is still **not implemented, not trained, not cross-validated**, and no TM-003A config, script, feature column, output, or `outputs/experiments.csv` row exists. No result number appears below.

**§§1–7.10 above are unchanged and are preserved as the historical pre-run record.** This amendment does not rewrite them. It tightens four things that were either too permissive (A), imprecise in wording (B), or under-specified in a way that would leave room for post-hoc interpretation (C, D).

**Precedence rule:** for the actual TM-003A implementation and report, **§7.11 overrides any conflicting wording in §§7.3–7.8.** Where §7.11 is silent, §7 stands unchanged. Where they conflict, §7 is read as the original plan and §7.11 as the adopted rule.

Amended items: **A** row-level as-of temporal rule · **B** terminology correction · **C** objective decision rule · **D** exact fallback diagnostics.

### 7.11.A Row-level as-of temporal rule (supersedes the fold-level wording in §7.3 / §7.5)

**Problem with the original wording.** §7.3 and §7.5 locked Trackman safety at the **fold** level: `trackman.season < valid_year`. That is sufficient for validation rows but **too permissive for training rows**. Under the fold rule, a 2020 training row inside the 2022 fold would be allowed to see Trackman from 2020 and 2021 — i.e. **its own season and its future**. That is a point-in-time violation on the train side even though the fold-level assertion passes.

**Adopted rule (row-level, point-in-time):**

> For a main row with `season = s`, the Trackman profile used to build that row's TM feature must be computed from **`trackman.season < s` only**.

No row ever sees Trackman information from its own season or later.

| main row season | Trackman seasons used for that row |
|---|---|
| 2019 | **none** → TM feature = `NaN` |
| 2020 | 2019 |
| 2021 | 2019–2020 |
| 2022 | 2019–2021 |
| 2023 | 2019–2022 |
| 2024 | 2019–2023 |
| 2025 (final inference) | 2019–2024 |

Properties of this rule:

- **It applies identically to train rows and validation rows.** There is no separate train-side relaxation.
- **It is strictly stricter than the old fold rule, and implies it.** Train rows satisfy `trackman.season < s < valid_year`; validation rows satisfy `trackman.season < s = valid_year`. So the existing fold-level assertion `trackman.season < valid_year` still holds everywhere and must still be run — it is now a *necessary but not sufficient* check. A **row-level** assertion (`max(trackman.season used) < row.season` for every row) must be added and must pass.
- **On validation rows the two rules coincide** (valid rows have `s = valid_year`), so this amendment does not change the validation-side feature values. It changes the **train side**, which is exactly where the leak was.
- **Profiles are keyed by row season, not by fold.** The `count × pitch_type_group` IVB profile is built once per distinct main season `s` (cutoff `< s`) and reused wherever rows of season `s` appear, rather than once per fold. A given train season's profile is therefore identical across all folds that contain it.

**2019 rows.** They have no prior history and their TM feature is `NaN` by construction. Handling stays exactly as already locked in §7.4 / §7.5: **fold-local missing handling only** (fold-local median impute + CatBoost native categoricals), identical to EXP-004 / TM-002A. No global impute, no cross-fold statistic, no backfill from a later season, no dropping of 2019 rows.

**Reporting consequence (clarification, not a new decision).** Because all 2019 rows are now `NaN` by design, the `NaN` rate in §7.7 item 4 must be reported **separately for validation rows and training rows**, and the training-side figure must state the structural 2019 share so a large train-side `NaN` rate is not misread as a defect. The validation-side `NaN` rate remains comparable to TM-002A's (`0.000311`), since validation rows are unaffected by this amendment.

**Reason for adopting before execution:** to prevent train rows from seeing future Trackman. Recorded here, before any CV number exists, so this cannot later look like a post-hoc adjustment.

### 7.11.B Terminology correction

The phrase **`prior-season Trackman`**, used in §7 and inherited from TM-002A / TM-AUDIT-002 wording, is ambiguous and does not match what TM-003A actually does.

Adopted terminology for all TM-003A implementation and reporting:

| use this | meaning |
|---|---|
| **`prior-history Trackman`** | default term for TM-003A |
| **`historical Trackman available before the row season`** | long form, when precision matters |
| `previous season` / `prior-season` | **reserved** — use *only* when exactly one season `s-1` is used |

TM-003A uses **every allowed earlier season** (`trackman.season < row.season`), not the single preceding season. It must therefore **never** be described as a "직전 한 시즌" / "previous-season" experiment. A 2024 row's profile is built from 2019–2023, not from 2023 alone.

This is a naming correction only. It changes no data, no cutoff, and no computation. Where §7 says `prior-season`, read `prior-history` under §7.11.A's row-level cutoff.

### 7.11.C Objective pre-declared decision rule (supersedes §7.8)

**Problem with the original wording.** §7.8 used judgement words — `meaningful`, `clearly`, `very small`, `effectively duplicate`, `near-complete duplicate` — that could be re-interpreted after the numbers are known. The rule below removes that discretion.

Definition used throughout:

```
Δ fold raw BSS = TM-003A fold raw BSS − EXP-004 fold raw BSS
```

evaluated on the three locked folds 2022 / 2023 / 2024, using the existing `src/metrics.py`.

**Duplicate diagnostics** = these four correlations (per §7.7; `tm_expected_ivb_global` is diagnostic-only and never a model input):

- `corr(tm_expected_ivb_by_count, tm_expected_ivb_global)`
- `corr(tm_expected_ivb_by_count, asof_pitcher_fastball_rate)`
- `corr(tm_expected_ivb_by_count, asof_pitcher_breaking_rate)`
- `corr(tm_expected_ivb_by_count, asof_pitcher_offspeed_rate)`

#### ADOPT_CANDIDATE — **all five** must hold

1. 3-fold CV mean `Δ raw BSS` **> 0**
2. **at least 2 of the 3** folds (2022 / 2023 / 2024) have `Δ raw BSS` **> 0**
3. 2023 fold `Δ raw BSS` **≥ 0**
4. leakage / temporal-safety checks **PASS** (including the new row-level assertion from §7.11.A)
5. **all four** duplicate diagnostics satisfy `|corr| < 0.98`

#### REJECT — **any one** is sufficient

1. 3-fold CV mean `Δ raw BSS` **< 0** **AND** 2023 fold `Δ raw BSS` **< 0**
2. leakage / temporal-safety check **FAIL**
3. **any** duplicate diagnostic has `|corr| ≥ 0.98`

#### HOLD

Every case that is neither ADOPT_CANDIDATE nor REJECT.

Notes:

- The `0.98` threshold is the TM-002A observed near-clone level (`tm_expected_rel_speed` vs `asof_pitcher_fastball_rate` = **+0.9867 / +0.9777 / +0.9786** by fold; `reports/tm_002a_expected_velocity.md` §11). It is set here **before** any TM-003A correlation is computed.
- Whether a correlation is evaluated per fold or pooled must be stated in the TM-003A report; if reported per fold, the rule applies to the **worst (largest `|corr|`)** fold. This is fixed now, not chosen later.
- These thresholds and the logical structure are **frozen before CV**. They are not modified after results are seen. Any change requires a further dated amendment written **before** reading the affected result.
- `ADOPT_CANDIDATE` means **candidate only**. It does not mean champion, does not authorize a Trackman bundle, does not authorize tuning or multi-seed, and does not authorize a Public LB submission.

### 7.11.D Fallback diagnostics — exact definition (supersedes §7.7 item 3)

**Problem with the original wording.** §7.7 item 3 said "fallback usage rate" without specifying whether it counted **rows** or **mix components**. Those give very different numbers: a row can be 1% breaking-mix and still be counted as "using fallback." Both are now defined and **both must be reported**.

Both are computed over **validation rows**, per fold and overall. Both use the frozen `min_n = 1000` from §7.4. Neither is a model input; both are experiment diagnostics that measure how much of the feature's mass actually came from the **count-conditioned** profile rather than the count-agnostic fallback.

For a validation row and group `g ∈ {fastball, breaking, offspeed}`, define
`I(g)` = 1 if that row's `count × g` prior-history cell had `n < 1000` and the count-agnostic same-group IVB fallback was used; else 0. (Under §7.11.A, cell `n` is computed from `trackman.season < row.season`.)

**1. `fallback_any_row_rate`** — row-level incidence

```
fallback_any_row     = 1 if any g has I(g) = 1, else 0
fallback_any_row_rate = mean(fallback_any_row) over validation rows
```

**2. `fallback_weight_share`** — mix-weighted mass

```
fallback_weight_share_row = Σ_g  asof_mix_g × I(g)
fallback_weight_share     = mean(fallback_weight_share_row) over validation rows
```

**Worked example** (illustrative only — not a prediction of the actual rates):

| group | as-of mix | cell status | `I(g)` |
|---|---:|---|---:|
| fastball | 0.60 | `n ≥ 1000`, count-conditioned | 0 |
| breaking | 0.30 | `n < 1000`, fallback | 1 |
| offspeed | 0.10 | `n < 1000`, fallback | 1 |

→ `fallback_any_row = 1`, `fallback_weight_share_row = 0.30 + 0.10 = 0.40`.

Interpretation fixed in advance: `fallback_any_row_rate` can be large while `fallback_weight_share` stays small — that is the expected shape, since the audit found the thin cells are rare 3-ball breaking/offspeed counts carried at small mix weight (TM-AUDIT-002 §6, §14). A **high** `fallback_weight_share` would mean TM-003A is largely running as the count-agnostic version, i.e. close to the TM-002A structure. This is a **diagnostic for interpretation**, not an additional pass/fail gate — the pass/fail gates are exactly and only those in §7.11.C.

### 7.11.E Unchanged by this amendment

Explicitly still locked exactly as in §7, and **not** touched here:

- baseline **EXP-004** (not EXP-006); one experiment, one feature
- feature name and formula `tm_expected_ivb_by_count = Σ_g asof_mix_g × IVB_hist(count, g)`, `g ∈ {fastball, breaking, offspeed}`, `count = balls_before-strikes_before`, `other` group excluded, 44 → 45 features, nothing dropped
- `min_n = 1000`, frozen; fallback to the same-group **count-agnostic** mean from the **same prior-history window**; cold start (`asof_pitcher_pitchmix_n == 0` or missing mix) → `NaN`
- locked expanding validation 2022 / 2023 / 2024; seed **42**, single seed; EXP-004 CatBoost recipe; EXP-004 season weights; `src/metrics.py` unchanged
- all §7.6 leakage guardrails (no current-pitch TM, no 2025 TM, no 1:1 row join, no ID guess-matching, no unofficial hand map, no test.csv aggregates, no test-row rolling/expanding, no test frequency, no target encoding)
- §7.9 scope: FE-001 and FE-002 remain **미실행** and are **not** withdrawn; no second Trackman feature, no tuning, no multi-seed, no ensembling, no stacking, no Public LB, no `exp/taewook-ensemble` audit

### 7.11.F Status after this amendment

**`APPROVED_TO_RUN / NOT YET EXECUTED`**

As of 2026-08-17, with amendments A–D adopted:

- TM-003A feature generation has **not** been run.
- TM-003A model training has **not** been run.
- TM-003A cross-validation has **not** been run.
- No config, script, prediction file, or `outputs/experiments.csv` row for TM-003A exists.
- **No TM-003A result numbers exist in this repository, and none are estimated, predicted, or implied in this amendment.**

When TM-003A executes, it must follow **§7 as amended by §7.11**, with §7.11 taking precedence on every conflicting point. Results go in a separate TM-003A report. Neither §7 nor §7.11 may be retroactively edited to match an outcome; any further change requires another dated amendment appended below this one, written before the affected result is read.

---

## 7.12 Pre-run clarification — 2026-08-17

Written **before execution**: TM-003A is still not implemented, not trained, not cross-validated. No result number appears below.

**Purpose:** remove the two remaining interpretive degrees of freedom left in §7.11.C and §7.11.D. This adds no new gate and changes no threshold — it only fixes *how* the already-frozen rules are computed.

### 7.12.A Duplicate-correlation evaluation rule (supersedes §7.11.C's "per fold or pooled, state it in the report")

§7.11.C allowed the report to choose between per-fold and pooled correlation. **That choice is removed.**

Adopted rule:

1. Duplicate correlations are computed **separately in each validation fold**.
2. The folds are the locked ones: **2022, 2023, 2024**.
3. For each diagnostic, the decision value is the **maximum absolute** Pearson correlation across those three folds.

```text
max_fold_abs_corr(X, Z) = max( |corr_2022(X, Z)|, |corr_2023(X, Z)|, |corr_2024(X, Z)| )
```

with `X = tm_expected_ivb_by_count` and `Z` the same four targets as §7.11.C:

- `tm_expected_ivb_global` (diagnostic-only; never a model input)
- `asof_pitcher_fastball_rate`
- `asof_pitcher_breaking_rate`
- `asof_pitcher_offspeed_rate`

**Duplicate gate:**

- all four have `max_fold_abs_corr < 0.98` → duplicate gate **PASS**
- any one has `max_fold_abs_corr ≥ 0.98` → **REJECT**

Fixed additionally:

- **Pooled correlation is not used for the verdict.** Reporting it for reference is allowed, but it has **no** influence on ADOPT / HOLD / REJECT.
- Correlation type is fixed to **Pearson**.
- Each fold's correlation uses only rows in that validation fold where **both** variables are finite / non-missing.
- The per-fold-vs-pooled choice and the `0.98` threshold are **not** revised after results are seen.

### 7.12.B Constructible vs non-constructible validation rows

§7.11.D did not say how rows with missing as-of mix enter the fallback denominator. Fixed here.

**Constructible row** — all of:

- `asof_pitcher_pitchmix_n > 0`
- `asof_pitcher_fastball_rate`, `asof_pitcher_breaking_rate`, `asof_pitcher_offspeed_rate` are **all** finite / non-missing

`tm_expected_ivb_by_count` is computable on such a row, so whether it used a `n < 1000` fallback is a meaningful question.

**Non-constructible row** — any of:

- `asof_pitcher_pitchmix_n == 0`
- one or more required as-of mix rates missing / non-finite

Such a row keeps `tm_expected_ivb_by_count = NaN` per §7.4 and is **not counted as fallback** — neither in the numerator nor in the denominator. It is reported only in the separate NaN diagnostic below.

### 7.12.C Exact denominators

All three are computed over validation rows, per fold and overall.

**`fallback_any_row_rate`**

```text
fallback_any_row_rate
= n_constructible_rows_with_any_fallback / n_constructible_validation_rows
```

- denominator: **constructible validation rows only**
- numerator: constructible rows where **at least one** of fastball / breaking / offspeed had its `count × pitch_type_group` prior-history cell `n < 1000` and therefore used the same-group **count-agnostic** fallback

**`fallback_weight_share`**

```text
fallback_weight_share_row = Σ_g asof_mix_g × I(g used fallback)

fallback_weight_share = mean(fallback_weight_share_row)
                        over constructible validation rows only
```

Same **constructible-row denominator** as `fallback_any_row_rate`.

**`tm_feature_nan_rate`** (separate diagnostic)

```text
tm_feature_nan_rate
= n_rows_where_tm_expected_ivb_by_count_is_NaN / n_all_validation_rows
```

Denominator is **all** validation rows. Numerator includes the non-constructible rows (mix cold-start / missing). The separate train-side NaN reporting required by §7.11.A (structural 2019 share) is unchanged by this clarification.

**Interpretation, fixed in advance:**

- **fallback** = the feature *can* be built, but the count-specific Trackman cell was too thin (`n < 1000`).
- **NaN** = the required as-of mix does not exist, so the feature *cannot* be built at all.
- These are different phenomena and must not be summed or reported as one rate.
- Both fallback diagnostics remain **interpretive only**. They add **no** pass/fail gate.

### 7.12.D Precedence and status

- §§1–7.11 are preserved unchanged as the historical pre-run record.
- For the actual TM-003A implementation and report:
  - **§7.12.A takes precedence** over §7.11.C's correlation-evaluation wording.
  - **§7.12.B–C take precedence** over §7.11.D's fallback-denominator ambiguity.
- Everything else in §7 and §7.11 stands exactly as written. **The ADOPT / HOLD / REJECT rule of §7.11.C is unchanged** — its five ADOPT conditions, three REJECT conditions, HOLD residual, and the `0.98` threshold all stay as frozen; §7.12.A only fixes how condition 5 / REJECT-3 is computed.

**`APPROVED_TO_RUN / NOT YET EXECUTED`**

As of 2026-08-17: no TM-003A feature generation, no training, no CV, no config, no script, no prediction file, no `outputs/experiments.csv` row — and no TM-003A result numbers anywhere in this repository, estimated or otherwise. Any further change requires another dated amendment appended below this one, written before the affected result is read.

---

## 8. TM-003A post-run decision — 2026-08-18

This section does **not** replace or rewrite §§7–7.12. Those remain the historical pre-run record.

The `APPROVED_TO_RUN / NOT YET EXECUTED` line in §7.12 is the status **as of 2026-08-17**, when that clarification was written. It is not the current status. **For current TM-003A status, this §8 post-run decision takes precedence.**

### 8.1 Execution status

TM-003A was implemented and evaluated. The canonical run is recorded.

- implementation commit: `642a127 feat: finalize TM-003A contextual expected IVB`
- canonical CV commit: `7429fea exp: record canonical TM-003A CV`
- report: `reports/tm_003a_expected_ivb.md`
- log: canonical `TM-003A` row in `outputs/experiments.csv` (`canonical=true`)

### 8.2 Team verdict

**ADOPT**

This is the team decision. It is **not** the same string as the mechanical pre-run bucket `ADOPT_CANDIDATE` in §7.11.C / the TM-003A report's "Predeclared rule signals" section. That bucket remains a frozen-rule signal only.

`ADOPT` applies only to the single added feature `tm_expected_ivb_by_count` on the locked EXP-004 controls. It does **not** mean champion, Public LB submission, adoption of a Trackman bundle, tuning, multi-seed, or any second Trackman feature.

### 8.3 Canonical comparison vs EXP-004 (copied, not re-derived)

Copied from the canonical TM-003A report / canonical `outputs/experiments.csv` notes:

| Fold | Δ raw BSS |
|---|---:|
| 2022 | −2.3462452333424153 |
| 2023 | +17.26371326633557 |
| 2024 | +19.317255966999596 |
| mean | +11.411574666664194 |

- 2023 and 2024 improved; 2022 worsened slightly; the three-fold mean improved.
- `canonical=true`
- `leakage_ok=true`
- the canonical rerun reproduced the pre-canonical fold deltas
- Codex independent audit: `PASS_WITH_NOTES`
- Codex did not find an issue that invalidates feature calculation, temporal leakage, baseline comparability, row alignment, or the metric
- reporting/reproducibility notes raised in that audit were addressed before the canonical rerun

Do not read the mean Δ of +11.41 as a large guaranteed lift. Record the split: two folds up, one fold slightly down, mean up, leakage checks passed, canonical reproduction held.

### 8.4 What this does not authorize

- TM-003A hyperparameter tuning or ablation
- adding `count × fastball × rel_speed` or any other Trackman feature in the same step
- Public LB submit
- declaring a new champion in place of the EXP-004 comparison baseline
- retiring FE-001 or FE-002

### 8.5 Next

TM-003A as a single-feature experiment is **closed**. Return to the unimplemented feature backlog in §6.1. Both remain **미실행**. Validate each as an independent hypothesis, not as a pair in one run:

1. **FE-001** — categorical `count_code`
2. **FE-002** — row-local prev1 shrinkage

---

## 9. FE-002 canonical execution (append-only, 2026-08-18)

This section does **not** rewrite §§1–6. Those remain the original plan and the FE-001A / FE-002A addendum. FE-002A (prev1 drop) is still not FE-002 (prev1 shrinkage).

### 7.1 What ran

| item | value |
|---|---|
| experiment_id | `FE-002` |
| branch | `exp/fe-002-prev1-shrinkage` (from `main`, not from FE-001) |
| change | in-place k=500 row-local shrinkage of `asof_pitcher_prev1_game_success_rate` toward `asof_pitcher_success_rate`; same column name; raw prev1 not used in the model |
| not changed | EXP-004 validation, metric, seed 42, CatBoost 300/0.05/6, season weights, prev3/prev5, count columns, Trackman, `count_code` |
| baseline | existing EXP-004 artifacts (not retrained) |
| Public LB | not submitted |

Sources: `configs/fe002.yaml`; `outputs/predictions/FE-002/comparison_vs_exp004.json`; `reports/fe_002_prev1_shrinkage.md`; `outputs/experiments.csv` row `FE-002`.

### 7.2 Canonical scores vs EXP-004

| Fold | EXP-004 raw BSS | FE-002 raw BSS | Δ |
|---|---:|---:|---:|
| 2022 | 2259.398328 | 2251.442006 | −7.956 |
| 2023 | −1366.481067 | −1340.347907 | +26.133 |
| 2024 | 578.184416 | 576.586121 | −1.598 |
| 3-fold mean | 490.367225 | 495.893407 | +5.526 |

Original `prev1 <= 0.35` (raw `train.csv` prev1, n=78,607): −184.202 → −209.752, **Δ −25.550**. Season × that slice: 2022 −41.0, 2023 −6.0, 2024 −31.5.

### 7.3 Final verdict

**REJECT**

The frozen core condition failed: original prev1<=0.35 raw BSS got worse. Mean +5.526 does not override that. The reject rule was not rewritten after seeing 2023 or n<=500 diagnostics.

### 7.4 Diagnostic finding (not an adopt argument)

- `n<=500`: Δ **+41.4**
- 2023 × `n<=500`: Δ **+201.4**

Interpretation:

k=500 global row-local prev1 shrinkage는 low-prev1 문제 해결에는 실패했다.
다만 low-history(n<=500) 집단에서 개선 신호가 관찰되었으며,
이는 별도의 후속 가설일 뿐 FE-002 채택 근거가 아니다.

---

## 10. WEIGHT-LAG-001 — POST-RUN DECISION

This section does **not** rewrite §§1–9. Those remain historical plan / TM-003A / FE-002 records.

Numbers below are copied from `outputs/analysis/weight_lag_001/contrasts.csv` and `reports/weight_lag_001.md`. No model was retrained for this decision.

### Decision

**REJECT**

### Scope

REJECT applies only to:

- L42
- CatBoost, seed 42
- iterations=300, learning_rate=0.05, depth=6, Logloss
- expanding full-history windows (RECENCY-AUDIT-001 `B_expanding`)
- geometric validation-relative season-lag weighting
- `m ∈ {1.5, 2.0, 3.0, 4.0}`

This is **not** a rejection of recency weighting as a general idea.

### Keep

- full-history training windows
- uniform weight (`m = 1.0`)

`m=1.0` CONTROL reproduced RECENCY-AUDIT-001 `B_expanding` on 2022/2023/2024 (exact raw BSS match; reproduction gate `PASS`).

### Why

- all treatment mean deltas vs `m=1.0` are negative:
  - m=1.5: −44.995460
  - m=2.0: −40.100129
  - m=3.0: −37.576280
  - m=4.0: −54.992200
- 2023 consistently deteriorates substantially under every `m>1`:
  - m=1.5: −122.982228
  - m=2.0: −100.260745
  - m=3.0: −101.686079
  - m=4.0: −120.108073
- 2024 gains at m=2.0 (+5.924867) and m=3.0 (+6.071192) are small and do not offset the 2023 loss; three-fold means remain negative
- no stable dose-response / monotonic improvement as `m` increases

### Do not do next

- `WEIGHT-LAG-002` fine sweep
- post-hoc `m=1.1/1.2/…` tuning
- LB-based weight selection

### Next analytical direction

Investigate the 2023 feature / context / regime shift.
Do not automatically launch a new training experiment from this closeout.

The 2023 deterioration under recent-season up-weighting weakens the claim that 2023 is simply old-data contamination or a simple recency problem.
