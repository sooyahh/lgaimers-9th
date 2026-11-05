# Task C multi-seed recheck (Task D)

- generated_at: `2026-08-07T16:57:18`
- C-0: EXP-005 L 42 features; C-4: +li +home_win_expectancy +away_win_expectancy (45)
- params: `{"iterations": 300, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss"}`
- seeds: `[42, 1, 7, 123, 2024]`
- paired Δ = C-4 raw BSS − C-0 raw BSS (same seed)
- **results only — no adopt/reject**

| seed | C-0 raw BSS | C-4 raw BSS | paired Δ (C4−C0) |
|---:|---:|---:|---:|
| 42 | 739.969723 | 745.759315 | +5.789592 |
| 1 | 733.738086 | 726.247708 | -7.490378 |
| 7 | 723.379950 | 725.006544 | +1.626593 |
| 123 | 722.855627 | 722.313517 | -0.542109 |
| 2024 | 723.114344 | 727.827971 | +4.713627 |

- paired Δ mean: `+0.819465`
- paired Δ std (sample, ddof=1): `5.277914`
- C-0 mean raw BSS: `728.611546`
- C-4 mean raw BSS: `729.431011`
