# Seed ensemble (Task E)

- generated_at: `2026-08-07T17:12:18`
- holdout: train 2019–2023 → valid 2024; L 42 features; current season weights
- seeds: `[42, 1, 7, 123, 2024]`
- ensemble: arithmetic mean of positive-class probabilities, then clip to [0,1]
- n1 = seed 42 only; n3 = seeds 42,1,7; n5 = all five
- **results only — no adopt/reject**

## EXP-005 params

- params: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`

| seed | raw BSS | official BSS | Brier | elapsed_s |
|---:|---:|---:|---:|---:|
| 42 | 739.969723 | 739.969723 | 0.247958 | 131.6 |
| 1 | 733.738086 | 733.738086 | 0.247974 | 140.0 |
| 7 | 723.379950 | 723.379950 | 0.248000 | 132.6 |
| 123 | 722.855627 | 722.855627 | 0.248001 | 133.5 |
| 2024 | 723.114344 | 723.114344 | 0.248001 | 137.2 |

- single-model mean / std: `728.611546` / `7.842338`

| ensemble | n | seeds | raw BSS | official BSS | Brier |
|---|---:|---|---:|---:|---:|
| n1_seed_42 | 1 | `[42]` | 739.969723 | 739.969723 | 0.247958 |
| n3_seeds_42_1_7 | 3 | `[42, 1, 7]` | 742.083248 | 742.083248 | 0.247953 |
| n5_all_seeds | 5 | `[42, 1, 7, 123, 2024]` | 739.851903 | 739.851903 | 0.247959 |

## EXP-006 Optuna best CatBoost params

- params: `{"depth": 7, "learning_rate": 0.028199996263578355, "l2_leaf_reg": 1.2151617026673374, "iterations": 361, "bagging_temperature": 0.9656320330745594, "loss_function": "Logloss"}`

| seed | raw BSS | official BSS | Brier | elapsed_s |
|---:|---:|---:|---:|---:|
| 42 | 744.606222 | 744.606222 | 0.247947 | 178.8 |
| 1 | 726.061741 | 726.061741 | 0.247993 | 178.5 |
| 7 | 729.916966 | 729.916966 | 0.247984 | 179.7 |
| 123 | 726.485069 | 726.485069 | 0.247992 | 179.6 |
| 2024 | 728.614578 | 728.614578 | 0.247987 | 180.3 |

- single-model mean / std: `731.136915` / `7.692037`

| ensemble | n | seeds | raw BSS | official BSS | Brier |
|---|---:|---|---:|---:|---:|
| n1_seed_42 | 1 | `[42]` | 744.606222 | 744.606222 | 0.247947 |
| n3_seeds_42_1_7 | 3 | `[42, 1, 7]` | 741.405626 | 741.405626 | 0.247955 |
| n5_all_seeds | 5 | `[42, 1, 7, 123, 2024]` | 739.737435 | 739.737435 | 0.247959 |

## Side-by-side ensemble raw BSS

| ensemble | EXP-005 params | EXP-006 tuned params |
|---|---:|---:|
| n1_seed_42 | 739.969723 | 744.606222 |
| n3_seeds_42_1_7 | 742.083248 | 741.405626 |
| n5_all_seeds | 739.851903 | 739.737435 |

## Saved predictions

- `outputs\predictions\task_de\exp005_seed_preds_2024.csv`
- `outputs\predictions\task_de\exp006_seed_preds_2024.csv`
