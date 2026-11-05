# Season weight schedule grid (Task B)

- generated_at: `2026-08-07T15:55:11`
- setup: CatBoost L + game_type kept; seed=`42`; train 2019–2023 → valid 2024
- params: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`
- **results only — no adopt/reject**

| schedule | weights | raw BSS | Δ vs B-1 | official BSS | Brier | elapsed_s |
|---|---|---:|---:|---:|---:|---:|
| B-1_current | `{2019: 1.0, 2020: 1.2, 2021: 1.4, 2022: 1.6, 2023: 2.0}` | 739.969723 | +0.000000 | 739.969723 | 0.247958 | 142.2 |
| B-2_mild | `{2019: 1.0, 2020: 1.1, 2021: 1.2, 2022: 1.3, 2023: 1.5}` | 725.401003 | -14.568720 | 725.401003 | 0.247995 | 135.7 |
| B-3_aggressive | `{2019: 1.0, 2020: 1.3, 2021: 1.7, 2022: 2.2, 2023: 3.0}` | 735.940909 | -4.028814 | 735.940909 | 0.247968 | 137.1 |
| B-4_extreme | `{2019: 0.5, 2020: 0.7, 2021: 1.0, 2022: 1.5, 2023: 3.0}` | 719.870339 | -20.099384 | 719.870339 | 0.248009 | 131.4 |

- B-1 (current) raw BSS: `739.969723`
