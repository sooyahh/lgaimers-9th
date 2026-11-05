# Season Categorical Experiment G (H skipped)

- generated_at: `2026-08-07T02:50:53`
- common: CatBoost; train=`2019–2023`; valid=`2024`
- **H skipped**: season clipping has no effect when valid year is train_max+1 (2024).
- **no adopt/reject judgment — results only**

## Comparison (raw BSS primary)

| experiment | season handling | raw BSS | official BSS | Brier | Δ raw vs EXP-003 |
|---|---|---:|---:|---:|---:|
| EXP-003 baseline | numeric absolute year | 562.278018 | 562.278018 | 0.248402 | +0.000000 |
| A sample_weight | numeric season + weights | 578.184416 | 578.184416 | 0.248363 | +15.906398 |
| G season categorical | `season` in categorical_cols | -357.891912 | 0.000000 | 0.250701 | -920.169929 |
| H season clip | skipped | — | — | — | — |

## Experiment G details

- n_features: `44`
- categorical includes season: `True`
- numeric includes season: `False`
- train season levels (as cat strings): `['2019', '2020', '2021', '2022', '2023']`
- valid season levels: `['2024']`
- unseen in valid vs train: `['2024']`
- n_train / n_valid: `1,221,585` / `253,507`
- raw BSS: `-357.891912`
- official BSS: `0.000000`
- Brier: `0.250701`
- elapsed_s: `1537.7`

- machine-readable: `outputs\eda\season_categorical_experiment_G.json`

