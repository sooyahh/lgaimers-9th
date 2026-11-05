# Seed variance (Task A)

- generated_at: `2026-08-07T15:46:04`
- setup: EXP-005 holdout (CatBoost L + game_type kept + current season weights; train 2019–2023 → valid 2024)
- params: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`
- n_features: `42`
- **results only — no adopt/reject**

| seed | raw BSS | official BSS | Brier | elapsed_s |
|---:|---:|---:|---:|---:|
| 42 | 739.969723 | 739.969723 | 0.247958 | 187.2 |
| 1 | 733.738086 | 733.738086 | 0.247974 | 187.5 |
| 7 | 723.379950 | 723.379950 | 0.248000 | 185.8 |
| 123 | 722.855627 | 722.855627 | 0.248001 | 429.8 |
| 2024 | 723.114344 | 723.114344 | 0.248001 | 236.0 |

- mean raw BSS: `728.611546`
- std raw BSS (sample, ddof=1): `7.842338`
- min / max: `722.855627` / `739.969723`
- range (max−min): `17.114096`
