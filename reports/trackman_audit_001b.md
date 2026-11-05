# TM-AUDIT-001B

- purpose: feasibility of TM-001 / TM-002 **without** 1:1 Trackman↔main join
- Trackman is a history table, not a current-pitch feature table
- no current-pitch TM, no 2025 TM, no test.csv row aggregates
- existing EDA Trackman artifacts were read, not overwritten

## 3-1. Minimum linkage re-check

Prior `target_regime_check.md` reported |pitcher_id ∩ pitcher_trackman_id| = 0. This re-check: numeric overlap=0, string overlap=0. Reproduced.

| check | value |
|---|---|
| TM rows | 1,793,078 |
| TM seasons | 2019,2020,2021,2022,2023,2024 |
| main pitcher_id ∩ pitcher_trackman_id (numeric) | 0 |
| main batter_id ∩ batter_trackman_id (numeric) | 0 |
| same ∩ as strings | pitcher=0, batter=0 |
| train pitcher_hand values | 1, 2 |
| TM pitcher_hand values | Left, Right |
| train batter_hand values | 1, 2 |
| TM batter_hand values | Left, Right |
| hand coding after normalize | NO_SHARED_HAND_CODES |
| shared hand keys | none |
| pitch_type_group missing rate | 0.0000 |

Seasonal TM sample size:

| season | n |
|---|---|
| 2019 | 255,957 |
| 2020 | 279,126 |
| 2021 | 301,032 |
| 2022 | 307,637 |
| 2023 | 315,100 |
| 2024 | 334,226 |

Physical-column missing rates (all TM rows):

| column | missing_rate |
|---|---|
| `rel_speed` | 0.0042 |
| `spin_rate` | 0.0070 |
| `induced_vert_break` | 0.0056 |
| `horz_break` | 0.0058 |
| `extension` | 0.0043 |
| `rel_height` | 0.0042 |
| `rel_side` | 0.0042 |
| `zone_speed` | 0.0044 |

pitch_type_group share (all TM rows):

| group | share |
|---|---|
| fastball | 0.5193 |
| breaking | 0.2860 |
| offspeed | 0.1823 |
| other | 0.0124 |

Direct player matching is **not** the goal. Overlap 0 means TM-001/002 must be context-level or league-level, never pitcher_id join.

## 3-2. TM-001 feasibility

Design: for validation season y, aggregate Trackman rows with `season < y` only.

| valid y | TM seasons used |
|---|---|
| 2022 | 2019–2021 |
| 2023 | 2019–2022 |
| 2024 | 2019–2023 |
| 2025 | 2019–2024 |

Minimum group n for 'no fallback' = 50 (feasibility parameter, not a model lock).

| valid_year | level | n_main | n_tm_hist | n_groups | coverage | fallback_rate | median group n | note |
|---|---|---|---|---|---|---|---|---|
| 2022 | L1_hand_count | 247,472 | 836,115 | missing | missing | 1.0000 | missing | hand codes do not overlap; this level cannot be joined |
| 2022 | L2_pitcher_hand_count | 247,472 | 836,115 | missing | missing | 1.0000 | missing | hand codes do not overlap; this level cannot be joined |
| 2022 | L3_pitcher_hand | 247,472 | 836,115 | missing | missing | 1.0000 | missing | hand codes do not overlap; this level cannot be joined |
| 2022 | L0_count_only | 247,472 | 836,115 | 12 | 1.0000 | 0.0000 | 57283.0 | prior-season TM only |
| 2023 | L1_hand_count | 245,525 | 1,143,752 | missing | missing | 1.0000 | missing | hand codes do not overlap; this level cannot be joined |
| 2023 | L2_pitcher_hand_count | 245,525 | 1,143,752 | missing | missing | 1.0000 | missing | hand codes do not overlap; this level cannot be joined |
| 2023 | L3_pitcher_hand | 245,525 | 1,143,752 | missing | missing | 1.0000 | missing | hand codes do not overlap; this level cannot be joined |
| 2023 | L0_count_only | 245,525 | 1,143,752 | 14 | 1.0000 | 0.0000 | 65486.5 | prior-season TM only |
| 2024 | L1_hand_count | 253,507 | 1,458,852 | missing | missing | 1.0000 | missing | hand codes do not overlap; this level cannot be joined |
| 2024 | L2_pitcher_hand_count | 253,507 | 1,458,852 | missing | missing | 1.0000 | missing | hand codes do not overlap; this level cannot be joined |
| 2024 | L3_pitcher_hand | 253,507 | 1,458,852 | missing | missing | 1.0000 | missing | hand codes do not overlap; this level cannot be joined |
| 2024 | L0_count_only | 253,507 | 1,458,852 | 14 | 1.0000 | 0.0000 | 83389.0 | prior-season TM only |
| 2025 | L1_hand_count | not available | 1,793,078 | missing | missing | missing | missing | 2025 main-row coverage not available without full test labels/rows |
| 2025 | L2_pitcher_hand_count | not available | 1,793,078 | missing | missing | missing | missing | 2025 main-row coverage not available without full test labels/rows |
| 2025 | L3_pitcher_hand | not available | 1,793,078 | missing | missing | missing | missing | 2025 main-row coverage not available without full test labels/rows |
| 2025 | L0_count_only | not available | 1,793,078 | missing | missing | missing | missing | 2025 main-row coverage not available without full test labels/rows |

Estimated raw feature count per level: **28** (4 mix shares + 8 physics × 3 stats).
A production implementation should keep a much smaller subset (e.g. medians + mix shares) to avoid collinearity.

Audit runtime: 22.3s on full TM+train context columns. Groupby cost is **low**.

Leakage risks if implemented as specified:

- **Allowed:** TM rows with season < validation season; context keys that exist on the current main row (`pitcher_hand`, `batter_hand`, `balls_before`, `strikes_before`).
- **Forbidden:** current-pitch TM physics, same-season TM after the pitch, 2025 TM, test-wide frequency/target encoding, player ID matching that isn't actually available.
- Residual risk: Trackman context may not be sampled from the same games as main (row counts differ by season). This is distribution shift, not row leakage.

## 3-3. TM-002 feasibility

Candidate: `expected_x = asof_fastball_rate * hist_fastball_x + ...` using TM physics from `season < y` only.

Main asof pitchmix missingness (all train):

| item | rate |
|---|---|
| fastball | 0.00054 |
| breaking | 0.00054 |
| offspeed | 0.00054 |
| pitchmix_n_zero | 0.00054 |

Prior-window TM group sizes:

| valid_year | hist n | fastball n | breaking n | offspeed n | other n | core groups ≥ min_n | main mix NA |
|---|---|---|---|---|---|---|---|
| 2022 | 836,115 | 481,844 | 215,163 | 123,233 | 15,875 | True | 0.00036 |
| 2023 | 1,143,752 | 628,687 | 306,493 | 189,732 | 18,840 | True | 0.00026 |
| 2024 | 1,458,852 | 774,775 | 404,308 | 259,360 | 20,409 | True | 0.00032 |
| 2025 | 1,793,078 | 931,120 | 512,851 | 326,809 | 22,298 | True | missing |

Example medians (2024-valid window = TM 2019–2023):

| group | rel_speed median | spin_rate median | IVB median | HB median | extension median |
|---|---|---|---|---|---|
| fastball | 142.77 | 2244.7 | 44.83 | 21.35 | 1.813 |
| breaking | 128.40 | 2313.9 | 6.25 | -5.30 | 1.703 |
| offspeed | 130.52 | 1721.5 | 22.79 | 30.55 | 1.770 |

Interaction with asof mix NA: when `asof_pitcher_pitchmix_n==0` the mix rates are NA (`data_integrity.md`). TM-002 must fall back (e.g. league mix or skip feature) on that tiny slice. That fallback is the same cold-start set as FEATURE-STABILITY-AUDIT.

Fold-safe: yes, if each fold rebuilds TM profiles from `season < valid_season` and uses only the current row's asof mix (already as-of).

## 3-4. Verdict

### TM-001: **HOLD**

Specified L1 (`pitcher_hand × batter_hand × balls × strikes`) **cannot be joined**: main hands are `{1,2}`, Trackman hands are `{Left, Right}`. No official mapping is in the repo, so this audit does **not** invent one.

Count-only fallback (**L0**) is a different design: coverage **1.000** and fallback_rate **0** on 2022/2023/2024 with large group n (median 57k–83k). That variant is temporally feasible and leakage-safe if `season < y`, but it is not the L1 spec and overlaps count already in the main model. Player ID overlap remains 0, matching `target_regime_check.md`.

| criterion | assessment |
|---|---|
| leakage safety | acceptable **if** prior-season filter is enforced in code |
| temporal feasibility | yes for 2022/2023/2024 folds; 2025 uses 2019–2024 TM (main coverage not measurable here) |
| coverage | see table; L3 should be near-complete if hand codes match |
| missingness | physical columns ~0.3–1% NA; group medians still defined |
| expected information gain | medium/low: league context, not pitcher identity; may help mix/physics regime shift |
| computational cost | low |
| overlap with existing features | partial with asof pitchmix and count/hand already in the model |

### TM-002: **PASS**

Past-season TM group physical profiles exist for fastball/breaking/offspeed, and main asof mix rates exist on almost all rows. Combination is fold-safe if TM seasons < valid year. Information is redundant with mix rates themselves; value depends on whether physics means shifted.

| criterion | assessment |
|---|---|
| leakage safety | acceptable **if** TM profiles use season < y and mix is official asof |
| temporal feasibility | yes |
| coverage | mix observed on ~all non-cold-start rows; TM group n is large |
| missingness | cold-start mix NA needs fallback |
| expected information gain | low–medium; collinear with asof mix if physics means are stable, more useful if physics shifted (IVB/HB did) |
| computational cost | low |
| overlap with existing features | **high** with `asof_pitcher_*_rate`; TM-002 is a physics-weighted remix of mix |

## What this audit does not do

- No model was trained.
- No 1:1 player or pitch join was forced.
- test.csv was not loaded for aggregates (local copy is a 5-row format sample).
- Existing `outputs/eda/fold_comparison.*` / `target_regime_check.md` were not overwritten.

```json
{
  "TM-001": "HOLD",
  "TM-002": "PASS",
  "pitcher_overlap": 0,
  "batter_overlap": 0,
  "hand_coding": "NO_SHARED_HAND_CODES",
  "min_group_n": 50,
  "runtime_seconds": 22.334048299999267
}
```
