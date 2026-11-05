# Provisional Validation Policy

Status: **잠정 확정** (VAL-AUDIT-001 / EXP-000 채택 이후)

## Primary

- Train seasons: `2019–2023`
- Valid seasons: `2024`

## Secondary (expanding-window)

1. `2019–2021` → `2022`
2. `2019–2022` → `2023`
3. `2019–2023` → `2024`

## Model-selection logging (required)

Per fold:

- Brier Score
- raw BSS
- official BSS (`max(0, raw)`)
- valid target rate
- prediction mean / std / min / max

Across secondary folds:

- fold raw BSS mean ± std
- concatenated OOF Brier
- concatenated OOF raw BSS
- concatenated OOF official BSS

Always also report:

- **primary 2024** official / raw BSS / Brier

## Non-goals of this lock

- Does **not** finalize within-season chronology (`order_cols` remain `[]`)
- Does **not** claim game-leakage-free splits (no game id)
- Does **not** use within-season random split as default
