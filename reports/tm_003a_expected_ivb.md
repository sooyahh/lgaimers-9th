# TM-003A Expected IVB by Count

**Run status: CANONICAL.** This report is from the explicit `--canonical` rerun.

- generated_at: `2026-08-18T03:17:03`
- experiment_id: `TM-003A`
- baseline: `EXP-004` (existing artifacts reused; not overwritten)
- feature: `tm_expected_ivb_by_count`
- seed: `42`
- run_kind: `canonical`
- canonical: `True`
- git_commit: `642a127a79f16f7b5c1840836500b58e7a015264`
- git_dirty: `false`

## Experiment definition

- experiment_id: `TM-003A`
- baseline: `EXP-004`
- feature: `tm_expected_ivb_by_count`
- seed: `42`
- model/controls: CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss;
  EXP-004 season weights 2019:1.0 … 2023:2.0; fold-local median impute
- temporal cutoff: `trackman.season < row.season`
- code commit: `642a127a79f16f7b5c1840836500b58e7a015264`
- config path: `configs/tm003a.yaml`
- min_n: `1000` (frozen, not tunable)
- aggregation: plain sum `Σ_g asof_mix_g × IVB_hist(count, g)`
- pitch groups: fastball / breaking / offspeed (`other` excluded)
- count cells: `balls_before-strikes_before`, legal 0-0 … 3-2
- sparse fallback: cell `n < 1000` → same-group count-agnostic mean
- non-finite defensive fallback: `n >= 1000` but non-finite cell mean → global fallback

## 1. Hypothesis

A prior-history Trackman `count × pitch_type_group` IVB profile combined with the
current row's official as-of pitch mix adds movement information that EXP-004 does
not have. Exactly one model feature is added.

## 2. What changed

- Added exactly `tm_expected_ivb_by_count`
- Feature count: 44 → 45
- TM-003A − EXP-004 = `['tm_expected_ivb_by_count']`
- EXP-004 − TM-003A = `[]`
- `tm_expected_ivb_global` is diagnostic-only and is **not** a model input
- Trackman cutoff: row-level `trackman.season < row.season`
- Sparse fallback: `min_n = 1000` to same-group count-agnostic mean
- Combination: plain sum `Σ_g asof_mix_g × IVB_hist(count, g)`

## 3. What stayed fixed

- CatBoost native cats, iterations=300, lr=0.05, depth=6, Logloss
- seed 42
- season weights 2019:1.0 … 2023:2.0
- expanding 2022/2023/2024 holdouts
- `src.metrics` unchanged
- fold-local median impute + native cats
- no current-pitch Trackman, no player-ID join, no 2025 TM, no test aggregates

## CV comparison

| Fold | n_train | n_valid | EXP-004 raw BSS | TM-003A raw BSS | Δ raw BSS | TM-003A official | TM-003A Brier | runtime_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 728,588 | 247,472 | 2259.398328 | 2257.052082 | -2.346245 | 2257.052082 | 0.243540 | 129.8 |
| 2023 | 976,060 | 245,525 | -1366.481067 | -1349.217354 | 17.263713 | 0.000000 | 0.253373 | 158.9 |
| 2024 | 1,221,585 | 253,507 | 578.184416 | 597.501672 | 19.317256 | 597.501672 | 0.248314 | 472.1 |

- TM-003A CV mean (ddof=0): `501.778800` (EXP-004 `490.367225`)
- TM-003A CV std (ddof=0): `1473.808437` (EXP-004 `1481.560943`)
- TM-003A worst fold: `-1349.217354`
- TM-003A best fold: `2257.052082`
- mean Δ raw BSS: `11.411575`
- fold direction: `-, +, +` (2022/2023/2024)

## Fallback diagnostics

Fold-level rates keep the constructible-row denominator from §7.12.
Overall rates are pooled from those fold counts; they are not hard-coded.

| Fold | n_valid | n_constructible | fallback_any_row_rate | fallback_weight_share | contextual_weight_share | nan_rate |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 247,472 | 247,384 | 0.011355 | 0.004902 | 0.995098 | 0.000356 |
| 2023 | 245,525 | 245,462 | 0.011643 | 0.005149 | 0.994851 | 0.000257 |
| 2024 | 253,507 | 253,426 | 0.012682 | 0.003770 | 0.996230 | 0.000320 |

| Overall | n_valid | n_constructible | fallback_any_row_rate | fallback_weight_share | tm_feature_nan_rate |
|---|---:|---:|---:|---:|---:|
| all folds | 746,504 | 746,272 | 0.01190049 | 0.00459912 | 0.00031078 |

- non-finite `n >= 1000` cell-mean fallback cells: `0`
- that branch is defensive only; it does not change the frozen sparse rule

Train-side NaN rate includes structural 2019 rows (no prior-history Trackman).

| Fold | train nan_rate | train n_2019 | train share_2019 |
|---|---:|---:|---:|
| 2022 | 0.326135 | 237,413 | 0.325854 |
| 2023 | 0.243536 | 237,413 | 0.243236 |
| 2024 | 0.194640 | 237,413 | 0.194348 |

## Cell-n diagnostics

Cell `n` is the **full cell row count**, including IVB-missing rows.
Values are reused from `fold_extra.cell_n_valid_profile` (no second aggregator).

| Fold | min | p10 | median | n_cells |
|---|---:|---:|---:|---:|
| 2022 | 227.0 | 2552.0 | 16950.0 | 36 |
| 2023 | 582.0 | 4150.5 | 24763.0 | 36 |
| 2024 | 858.0 | 5907.0 | 33129.5 | 36 |

## Duplicate-signal diagnostics (validation, Pearson, per fold)

| Fold | target | pearson | n |
|---|---|---:|---:|
| 2022 | tm_expected_ivb_global | 0.902923 | 247,384 |
| 2022 | asof_pitcher_fastball_rate | 0.731324 | 247,384 |
| 2022 | asof_pitcher_breaking_rate | -0.788830 | 247,384 |
| 2022 | asof_pitcher_offspeed_rate | 0.092349 | 247,384 |
| 2023 | tm_expected_ivb_global | 0.898108 | 245,462 |
| 2023 | asof_pitcher_fastball_rate | 0.729995 | 245,462 |
| 2023 | asof_pitcher_breaking_rate | -0.787391 | 245,462 |
| 2023 | asof_pitcher_offspeed_rate | 0.162158 | 245,462 |
| 2024 | tm_expected_ivb_global | 0.914260 | 253,426 |
| 2024 | asof_pitcher_fastball_rate | 0.774503 | 253,426 |
| 2024 | asof_pitcher_breaking_rate | -0.801916 | 253,426 |
| 2024 | asof_pitcher_offspeed_rate | 0.131710 | 253,426 |

## Predeclared rule signals

These are mechanical signals from the frozen pre-run rule. They are **not** a
team ADOPT / HOLD / REJECT decision.

- leakage_ok: `True`
- predeclared_rule_bucket: `ADOPT_CANDIDATE`
- duplicate max |corr|: `{"asof_pitcher_breaking_rate": 0.8019163959832131, "asof_pitcher_fastball_rate": 0.7745029343742039, "asof_pitcher_offspeed_rate": 0.1621579512267726, "tm_expected_ivb_global": 0.914259569216095}`

## Integrity/reproducibility

- git_commit: `642a127a79f16f7b5c1840836500b58e7a015264`
- git_dirty: `false`
- git_branch: `exp/tm-003a-expected-ivb`
- config path: `configs/tm003a.yaml`
- command: `python -m scripts.run_tm003a_expected_ivb --canonical`
- artifact path: `outputs/predictions/TM-003A`
- report path: `reports/tm_003a_expected_ivb.md`
- run_kind: `canonical`
- canonical: `True`

## Team decision

**ADOPT**

This is the **team** verdict for the single feature `tm_expected_ivb_by_count`.
It is distinct from the mechanical `predeclared_rule_bucket: ADOPT_CANDIDATE`
in the previous section. That bucket is a frozen-rule signal only.

Copied from this canonical report and the canonical `TM-003A` row in
`outputs/experiments.csv` (not re-derived):

- 2022 Δ raw BSS vs EXP-004: `-2.3462452333424153`
- 2023 Δ raw BSS vs EXP-004: `+17.26371326633557`
- 2024 Δ raw BSS vs EXP-004: `+19.317255966999596`
- mean Δ raw BSS vs EXP-004: `+11.411574666664194`
- `canonical`: `true`
- `leakage_ok`: `true`

2023 and 2024 improved; 2022 worsened slightly; the three-fold mean improved.
The canonical rerun reproduced the pre-canonical fold deltas. Codex independent
audit: `PASS_WITH_NOTES`. Codex did not find an issue that invalidates feature
calculation, temporal leakage, baseline comparability, row alignment, or the
metric. Reporting/reproducibility notes were addressed before the canonical
rerun.

`ADOPT` means this one feature is accepted on the locked EXP-004 controls.
It does **not** mean champion, Public LB submit, a Trackman bundle, tuning,
or any other Trackman feature. No further TM-003A tuning or ablation.

Next: return to the unimplemented feature backlog, each as its own hypothesis
— **FE-001** (`count_code` categorical), then **FE-002** (prev1 shrinkage).

## Re-run

Default rerun is **not** canonical:

```powershell
python -m scripts.run_tm003a_expected_ivb
```

Canonical rerun (only after an explicit instruction, on a committed tree):

```powershell
python -m scripts.run_tm003a_expected_ivb --canonical
```

- summary: `outputs/predictions/TM-003A/summary.json`
- OOF: `outputs/predictions/TM-003A/oof_holdout_predictions.csv`
- comparison: `outputs/predictions/TM-003A/comparison_vs_exp004.json`
