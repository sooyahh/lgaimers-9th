# EXP-006 Tuning + Ensemble

- generated_at: `2026-08-07T09:54:41`
- holdout: train `[2019, 2020, 2021, 2022, 2023]` → valid `[2024]` (single season holdout, not TimeSeriesSplit)
- features: `42` with L (drop ['pitcher_id', 'batter_id']), game_type kept
- weights holdout: `{2019: 1.0, 2020: 1.2, 2021: 1.4, 2022: 1.6, 2023: 2.0}`; full: `{2019: 1.0, 2020: 1.2, 2021: 1.4, 2022: 1.6, 2023: 2.0, 2024: 2.4}`
- Optuna trials: `35` each; maximize raw BSS
- **no adopt/reject judgment — results only**

## CatBoost Optuna

- best trial raw BSS: `744.675900`
- refit holdout raw BSS: `744.675900`
- best params: `{"depth": 7, "learning_rate": 0.028199996263578355, "l2_leaf_reg": 1.2151617026673374, "iterations": 361, "bagging_temperature": 0.9656320330745594}`
- tune elapsed_s: `5849.3`

## LightGBM Optuna

- best trial raw BSS: `613.278850`
- refit holdout raw BSS: `613.278850`
- best params: `{"n_estimators": 134, "learning_rate": 0.040079006339060044, "num_leaves": 21, "max_depth": 5, "min_child_samples": 60, "subsample": 0.6710161397413721, "colsample_bytree": 0.9220383092635855, "reg_lambda": 0.20785880614346974}`
- tune elapsed_s: `369.3`

## Ensemble (2024 holdout preds)

- corr(pred_cb, pred_lgb): `0.922101`

| combo | w_cb | w_lgb | raw BSS | official BSS | Brier |
|---|---:|---:|---:|---:|---:|
| tuned_catboost | 1.0 | 0.0 | 744.675900 | 744.675900 | 0.247947 |
| tuned_lightgbm | 0.0 | 1.0 | 613.278850 | 613.278850 | 0.248275 |
| ensemble_w_cb=0.3 | 0.3 | 0.7 | 674.466999 | 674.466999 | 0.248122 |
| ensemble_w_cb=0.5 | 0.5 | 0.5 | 704.892892 | 704.892892 | 0.248046 |
| ensemble_w_cb=0.7 | 0.7 | 0.3 | 727.025819 | 727.025819 | 0.247991 |

- selected for full-train/submit: **`tuned_catboost`** (raw BSS `744.675900`)

## Full train + submit

- train seasons: 2019–2024 full
- zip: `outputs\submissions\EXP-006_submit.zip`
- zip entries: `['requirements.txt', 'script.py', 'model/catboost_full.joblib', 'model/feature_meta.json']`
- smoke: ok=`True` rc=`0` rows=`5` pred_mean=`0.4700110579883023` wall_s=`3.149`

- summary: `outputs\predictions\EXP-006\summary.json`

