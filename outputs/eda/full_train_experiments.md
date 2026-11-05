# Full-train experiments (S → R → EXP-005)

- generated_at: `2026-08-07T07:29:38`
- **no adopt/reject judgment — results only**

## Experiment S (config check on primary_2024)

Common: CatBoost; train `2019–2023`; valid `2024`; EXP-004 season weights; **L** (drop `pitcher_id`, `batter_id`).

| run | game_type | raw BSS | official BSS | Brier |
|---|---|---:|---:|---:|
| S-1 L only (from unseen_id_experiments L_drop_ids) | kept | 739.969723 | 739.969723 | — |
| S-2 L + drop game_type | dropped | 703.144718 | 703.144718 | 0.248050 |

- Δ (S-2 − S-1) raw BSS: `-36.825005`
- mechanical select for submit (higher raw BSS): **`R_L_keep_game_type`**

### S-2 details

- n_train / n_valid: `1,221,585` / `253,507`
- n_features: `41`
- elapsed_s: `136.5`

## Experiment R (full train 2019–2024)

- weights: `{"2019": 1.0, "2020": 1.2, "2021": 1.4, "2022": 1.6, "2023": 2.0, "2024": 2.4}`
- L applied on both variants; no holdout metrics

| variant | game_type | n_features | n_train | model path | elapsed_s |
|---|---|---:|---:|---|---:|
| R_L_keep_game_type | kept | 42 | 1,475,092 | `outputs\models\EXP-005\R_L_keep_game_type\catboost_full.joblib` | 172.2 |
| R_L_drop_game_type | dropped | 41 | 1,475,092 | `outputs\models\EXP-005\R_L_drop_game_type\catboost_full.joblib` | 163.9 |

## EXP-005 submit.zip

- selected variant: `R_L_keep_game_type`
- zip: `outputs\submissions\EXP-005_submit.zip`
- zip entries: `['script.py', 'requirements.txt', 'model/catboost_full.joblib', 'model/feature_meta.json']`
- smoke test (5-row official test): rc=`0`, rows=`5`, cols=`['row_id', 'control_success']`, pred_mean=`0.4687281328602285`, wall_s=`2.687`

- summary: `outputs\predictions\EXP-005\summary.json`

