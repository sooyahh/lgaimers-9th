# Season Extrapolation Alternatives (D / E / F)

- generated_at: `2026-08-07T01:43:37`
- common: CatBoost EXP-003 params; train=`2019–2023`; valid=`2024`
- params: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`; seed=`42`
- **no adopt/reject judgment — results only**

## Comparison (raw BSS primary)

| experiment | train | season handling | n_features | raw BSS | official BSS | Brier | Δ raw vs EXP-003 |
|---|---|---|---:|---:|---:|---:|---:|
| EXP-003 baseline | 2019–2023 | season absolute year | 44 | 562.278018 | 562.278018 | 0.248402 | +0.000000 |
| A sample_weight | 2019–2023 weighted | season kept + weights | 44 | 578.184416 | 578.184416 | 0.248363 | +15.906398 |
| D drop season | 2019–2023 | season removed | 43 | -61.627806 | 0.000000 | 0.249961 | -623.905824 |
| E season_since_start | 2019–2023 | season→0..4 (valid=5) | 44 | 562.278018 | 562.278018 | 0.248402 | +0.000000 |
| F no season + weights | 2019–2023 weighted | season removed + A weights | 43 | 116.035290 | 116.035290 | 0.249517 | -446.242727 |

## Experiment D — season removed

- features: `43` (excluded `season`)
- n_train / n_valid: `1,221,585` / `253,507`
- raw BSS: `-61.627806`
- official BSS: `0.000000`
- Brier: `0.249961`
- elapsed_s: `769.8`

## Experiment E — `season_since_start = season - 2019`

- replaces absolute `season` with elapsed years from 2019
- train range: `[0, 4]` (expect `[0, 4]`); valid range: `[5, 5]` (expect `[5, 5]`)
- features: `44`
- n_train / n_valid: `1,221,585` / `253,507`
- raw BSS: `562.278018`
- official BSS: `562.278018`
- Brier: `0.248402`
- elapsed_s: `306.5`

## Experiment F — no season + sample_weight (A weights)

- `season` removed from features; weights `2019=1.0 … 2023=2.0`
- features: `43`
- n_train / n_valid: `1,221,585` / `253,507`
- raw BSS: `116.035290`
- official BSS: `116.035290`
- Brier: `0.249517`
- elapsed_s: `218.1`

- machine-readable: `outputs\eda\season_extrapolation_experiments.json`

## Follow-up: Experiment G (season categorical; H skipped)

| EXP-003 | A | G |
|---:|---:|---:|
| 562.278018 | 578.184416 | -357.891912 |

Details: `season_categorical_experiment_G.md`
