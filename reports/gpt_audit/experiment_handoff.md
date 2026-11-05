# Experiment Handoff for GPT Data Analysis

Prepared: 2026-08-07  
Updated: 2026-08-13 (VAL-EXP006-SECONDARY-001 logged; EXP-006 vs EXP-005 secondary verdict = hold)
Scope: EXP-000 ~ EXP-006 + VAL-EXP006-SECONDARY-001
Rule: facts from repository artifacts only; no estimated LB or unlogged decisions.

---

## 1. Problem definition

Binary probability prediction of `control_success` for baseball pitches (LG Aimers Phase 2).

- Train seasons present: 2019–2024 (`data/raw/train.csv`, 1,475,092 rows, 49 cols)
- Test: sample format in `data/raw/test.csv` (5 rows in local copy per `data_integrity.md`; not full evaluation labels)
- Submission key: `row_id` → predict `control_success` in `[0, 1]`
- Forbidden / unused: current-pitch Trackman fields; participant-built test aggregates (`feature_safety_audit.md`)

---

## 2. Official metric

From `src/metrics.py` (DACON / Competition Spec §2 citation in docstring):

```
Brier = mean((p - y)^2)
r = mean(y)
Score = max(0, 100000 * (1 - Brier / (r*(1-r))))
```

Maximize. If `r*(1-r)==0`, local code raises `ValueError`.

---

## 3. Metric implementation

| Piece | Location |
|---|---|
| Core BSS | `src/metrics.py` → `compute_brier_metrics` returns `brier_score`, `raw_brier_skill_score`, `official_brier_skill_score` |
| Wrapper | `src/evaluate.py` → `evaluate_predictions` / `score_metric` |
| Config | `configs/baseline.yaml` → `brier_skill_score`, `clip_at_zero: true`, `scale: 100000` |

**Analysis note:** when predictions are worse than climatology, official BSS is 0 and cannot rank models. Use **raw BSS** and Brier for diagnostics (`validation_policy.md`, EXP reports).

---

## 4. Primary validation

- Train: 2019–2023  
- Valid: 2024  
- Status: **잠정 확정** (`reports/validation_policy.md`)

---

## 5. Secondary validation

Expanding window:

1. 2019–2021 → 2022  
2. 2019–2022 → 2023  
3. 2019–2023 → 2024  

Logged as fold raw BSS mean±std and concatenated OOF metrics for EXP-001A–004.

EXP-005/006 **2022 and 2023** scores now exist from `VAL-EXP006-SECONDARY-001` (frozen EXP-006 params vs EXP-005 untuned; identical L features and season weights). That run did **not** treat 2024 as a new secondary fold or selection criterion (2024 was EXP-006’s Optuna holdout).

---

## 6. Why this validation was chosen

Evidence-based (not invented here):

- No `game_id` / event datetime → within-season chronological CV not supported (`validation_structure_audit.md`)
- Season axis available and monotonic in file order
- VAL-AUDIT-001 (`season_validation_audit.md`): target mean drifts across seasons; constant-mean fails on all expanding folds → single 2024 holdout alone is risky; expanding raw BSS reported as auxiliary
- Policy explicitly does **not** claim game-leakage-free or within-season order (`validation_policy.md` Non-goals)

---

## 7. EXP-000 → EXP-006 flow

1. **EXP-000** — train mean constant → primary raw −827.9, official 0.  
2. **EXP-001A** — safe 44 features + LR → primary raw −464.6 (better, still bad); fold unstable.  
3. **EXP-002** — LightGBM → primary raw +8.2; sec2023 −2208.  
4. **EXP-003** — CatBoost → primary raw +562.3; tree baseline.  
5. **EXP-004** — season weights → primary +578.2; sec2023 still ~−1366.  
6. Diagnostics — regime / season / unseen-id / gap (see `outputs/eda/*`).  
7. **EXP-005** — drop IDs (L), keep `game_type`, full-train submit; S1 primary raw BSS 739.97.
8. **EXP-006** — Optuna CB/LGB + ensemble; select tuned CB primary raw BSS 744.68; full-train submit.
9. **VAL-EXP006-SECONDARY-001** — re-score frozen EXP-006 vs EXP-005 on untouched 2022/2023. Verdict: **hold**.

Formal adopt/reject: EXP-000–005 remain `results_only_no_adopt_reject`. EXP-006 vs EXP-005 on secondary folds is recorded as **hold** (not adopt, not reject).

---

## 8. Highest observed primary score (not a finalized best model)

**EXP-006 tuned CatBoost** still has the **highest observed primary 2024 raw BSS** among logged EXPs: **744.675900** (`outputs/eda/tuning_ensemble.md`). That 2024 number is **not** independent evidence: it is the Optuna tune fold.

Secondary validation is **no longer pending**. `VAL-EXP006-SECONDARY-001` (`outputs/predictions/VAL-EXP006-SECONDARY-001/summary.json`) compared frozen EXP-006 params to EXP-005 untuned on 2022 and 2023 only.

### EXP-005 vs EXP-006 scores (do not mix metric kinds)

**raw BSS** (ranking metric for this comparison)

| model | 2022 | 2023 | secondary 2-fold mean | secondary 2-fold std (ddof=0) | primary 2024 (reference) |
|---|---|---|---|---|---|
| EXP-005 untuned | 2317.878894 | −1306.558749 | 505.660072 | 1812.218822 | 739.969723 |
| EXP-006 tuned | 2312.821927 | −1287.270639 | 512.775644 | 1800.046283 | 744.675900 |
| Δ (006 − 005) | **−5.056967** | **+19.288111** | **+7.115572** | — | +4.706177 |

**official BSS** (clipped `max(0, raw)`)

| model | 2022 | 2023 | primary 2024 |
|---|---|---|---|
| EXP-005 | 2317.878894 | **0** | 739.969723 |
| EXP-006 | 2312.821927 | **0** | 744.675900 |
| Δ (006 − 005) | −5.056967 | **0** | +4.706177 |

2023 official BSS is 0 for both because raw BSS is negative. It cannot rank the two models.

**Brier** (lower is better)

| model | 2022 | 2023 | primary 2024 |
|---|---|---|---|
| EXP-005 | 0.243388 | 0.253266 | (not stored on EXP-005 S1 summary) |
| EXP-006 | 0.243401 | 0.253218 | 0.247947 |
| Δ (006 − 005) | +0.000013 (worse) | −0.000048 (better) | — |

**quantile-10 ECE** (VAL only; lower is better): 2022 0.004323 → 0.004762 (worse); 2023 0.025617 → 0.025228 (better).

### Verdict for EXP-006 as current main model: **hold**

Not **adopt**: 2022 raw BSS did not improve; calibration also mixed; 2024 gain is the tune fold.
Not **reject**: 2023 raw BSS improved (Δ +19.29) and the 2-fold mean Δ is positive (+7.12).

- `is_comparable_cv=false` remains: EXP-006 `cv_raw_bss_mean` still repeats the 2024 tune holdout, not a 3-fold CV mean
- No Public LB in `outputs/experiments.csv`
- Among EXPs with original expanding 3-fold scores (`is_comparable_cv=true`), highest primary is still EXP-004 (578.18) with still-negative secondary_2023

### EXP-006 tuning (from existing artifacts only)

Sources: `outputs/eda/tuning_ensemble.md`, `outputs/predictions/EXP-006/summary.json`, `outputs/predictions/VAL-EXP006-SECONDARY-001/summary.json`

| Item | Recorded value |
|---|---|
| Tuning objective | maximize **raw BSS** (Optuna; stated in `tuning_ensemble.md`) |
| Validation dataset | single season holdout: train `[2019–2023]` → valid `[2024]` (not TimeSeriesSplit / not secondary expanding folds) |
| Trial count | **35** trials each for CatBoost and LightGBM (`n_trials: 35`) |
| Best CatBoost params (selected submit) | `depth=7`, `learning_rate=0.028199996263578355`, `l2_leaf_reg=1.2151617026673374`, `iterations=361`, `bagging_temperature=0.9656320330745594` |
| Best LightGBM params (holdout only; not selected) | `n_estimators=134`, `learning_rate=0.040079006339060044`, `num_leaves=21`, `max_depth=5`, `min_child_samples=60`, `subsample=0.6710161397413721`, `colsample_bytree=0.9220383092635855`, `reg_lambda=0.20785880614346974` |
| Early stopping / pruning policy | **unknown** (not stated as a named policy beyond logged `best_iteration` on trials) |
| Search space bounds | **unknown** in handoff sense beyond observed trial params in `summary.json` |
| Secondary-fold re-evaluation of best params | **done** — `VAL-EXP006-SECONDARY-001` (2022/2023 only; 2024 not re-used as selection) |

---

## 9. Currently adopted / operational decisions

Documented as provisional, not formal “채택” votes:

| Decision | Status |
|---|---|
| Primary/secondary season policy | 잠정 확정 |
| Exclude WE / LI / `row_id` from features | In `baseline.yaml` + all model EXPs |
| Prefer CatBoost over LR/untuned LGB | Implied by scores; no formal adopt text |
| Season sample weights | Used EXP-004→006 |
| Drop `pitcher_id`/`batter_id` (L); keep `game_type` | EXP-005/006 submit path |
| EXP-006 tuned params as current main model vs EXP-005 | **hold** (`VAL-EXP006-SECONDARY-001`) |
| `order_cols: []` | Remains empty |

---

## 10. Leakage-related checks

Confirmed in `outputs/eda/leakage_check.md` / audits:

- Season holdouts disjoint; no shuffle in split construction
- asof `prev*_game_*` sanity: no mid-season NA re-entry under proxy order
- WE/LI excluded pending construction confirmation

Open risks:

- No game_id → cannot block within-season game leakage
- Pitcher ID overlaps across years (~35% in all of 2022–24)
- Trackman not joinable (`pitcher_id ∩ pitcher_trackman_id = 0`)

---

## 11. Unresolved suspicions

1. Secondary_2023 failure may preview future-season risk (`gap_diagnosis.md`)  
2. `game_type` drop helps 2023 a lot but hurts 2024 (`unseen_id_experiments.md` O vs S)  
3. Absolute `season` feature importance high; drop collapses 2024 (experiments D/G)  
4. Target regime shift 2022→2023 (target rate + Trackman margins) — join-limited  
5. Public LB unknown → CV–LB gap unknown  
6. EXP-006 holdout-tuned hyperparameters: secondary 2022/2023 now measured; **hold** (2022 raw BSS Δ −5.06, 2023 Δ +19.29)

---

## 12. Next experiment candidates (from logged next_actions / open diagnostics)

1. Submit EXP-006 (or EXP-005) and **record** Public LB into `experiments.csv`  
2. ~~Re-evaluate tuned EXP-006 params on secondary_2022/2023~~ **done** (`VAL-EXP006-SECONDARY-001`; verdict **hold**)
3. Resolve game_type vs 2023/2024 tradeoff under L+weights  
4. Formal adopt/reject of L / weights / CatBoost / validation lock  
5. WE/LI only after organizer construction clarity  
6. Trackman FE blocked until mapping exists  

---

## 13. Package pointers

- Scores table: `reports/gpt_audit/experiment_summary.csv`  
- Config linkage: `reports/gpt_audit/config_manifest.csv`  
- OOF: `reports/gpt_audit/oof/` + `oof_inventory.csv`  
- Originals: `reports/gpt_audit/SOURCE_INDEX.md`  

### Score-scope / source columns (`experiment_summary.csv`)

- `score_scope`, `n_validation_folds`, `is_comparable_cv`: prevent treating EXP-005/006 `cv_raw_bss_mean` as 3-fold CV means (`is_comparable_cv=false`). EXP-005/006 secondary_2022/2023 columns are filled from `VAL-EXP006-SECONDARY-001`.
- `source`: for EXP-000, secondary folds + `cv_raw_bss_mean` come from **VAL-AUDIT-001** (`reports/season_validation_audit.md`), not from an EXP-000 OOF artifact. EXP-000 itself only persisted primary-2024 `holdout_predictions.csv`. For EXP-005/006 secondary folds, source includes `outputs/predictions/VAL-EXP006-SECONDARY-001/summary.json`.
