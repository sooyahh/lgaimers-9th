from __future__ import annotations

"""Audit train.csv temporal / group key structure. Does not invent order_cols."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
REPORT_PATH = ROOT / "reports" / "validation_structure_audit.md"


def _dup_stats(df: pd.DataFrame, cols: list[str]) -> dict:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return {"missing_cols": missing}
    n = len(df)
    nunique_keys = int(df.groupby(cols, dropna=False).ngroups)
    n_dup_rows = int(df.duplicated(cols, keep=False).sum())
    n_dup_groups = int((df.groupby(cols, dropna=False).size() > 1).sum())
    max_group = int(df.groupby(cols, dropna=False).size().max()) if n else 0
    is_unique = nunique_keys == n and df[cols].isna().sum().sum() == 0
    return {
        "cols": cols,
        "n_rows": n,
        "n_unique_keys": nunique_keys,
        "n_duplicate_rows": n_dup_rows,
        "n_duplicate_groups": n_dup_groups,
        "max_group_size": max_group,
        "has_null": bool(df[cols].isna().any().any()),
        "is_unique": bool(is_unique),
    }


def _fmt_stats(s: dict) -> str:
    if "missing_cols" in s:
        return f"columns missing: {s['missing_cols']}"
    return (
        f"unique={s['is_unique']}, n_unique_keys={s['n_unique_keys']}, "
        f"dup_rows={s['n_duplicate_rows']}, dup_groups={s['n_duplicate_groups']}, "
        f"max_group_size={s['max_group_size']}, has_null={s['has_null']}"
    )


def main() -> None:
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    cols = list(df.columns)

    # row_id order analysis
    row_id = df["row_id"].astype(str)
    sorted_lex = row_id.sort_values(kind="mergesort")
    lex_equals_file_order = bool((sorted_lex.to_numpy() == row_id.to_numpy()).all())
    # numeric suffix if pattern like PREFIX_digits
    suffixes = row_id.str.extract(r"_(\d+)$", expand=False)
    has_numeric_suffix = bool(suffixes.notna().all())
    if has_numeric_suffix:
        suffix_num = suffixes.astype(int)
        suffix_monotonic = bool(suffix_num.is_monotonic_increasing)
        suffix_unique = bool(suffix_num.is_unique)
    else:
        suffix_monotonic = False
        suffix_unique = False

    # candidate game-id-like columns (exclude calendar fields and asof_* history features)
    game_like = [
        c
        for c in cols
        if ("game_id" in c.lower() or c.lower() in {"game_id", "game"})
        and not c.startswith("asof_")
    ]
    calendar_game_fields = [c for c in ("game_month", "game_dayofweek", "game_type") if c in cols]
    id_cols = [c for c in cols if c.endswith("_id") or c == "row_id"]

    key_candidates = [
        ["row_id"],
        ["season"],
        ["pitcher_id"],
        ["batter_id"],
        ["season", "pitcher_id"],
        ["season", "batter_id"],
        ["season", "pitcher_id", "batter_id"],
        ["season", "pitcher_id", "batter_id", "inning", "top_bottom"],
        [
            "season",
            "pitcher_id",
            "batter_id",
            "inning",
            "top_bottom",
            "balls_before",
            "strikes_before",
            "outs_before",
        ],
        [
            "season",
            "game_month",
            "game_dayofweek",
            "pitcher_id",
            "batter_id",
            "inning",
            "top_bottom",
            "balls_before",
            "strikes_before",
            "outs_before",
            "base_state",
            "num_runners_on",
        ],
        [
            "season",
            "game_month",
            "game_dayofweek",
            "pitcher_team_id",
            "batter_team_id",
            "pitcher_id",
            "batter_id",
            "inning",
            "top_bottom",
            "balls_before",
            "strikes_before",
            "outs_before",
            "base_state",
            "li",
            "home_win_expectancy",
            "away_win_expectancy",
            "score_diff_home",
            "score_diff_pitcher_team",
            "run_total_before",
        ],
    ]

    stats = {tuple(k): _dup_stats(df, k) for k in key_candidates}

    # Within pitcher-season: can we recover pitch order from context alone?
    # Count max collisions for pitcher-season + rich context
    rich = [
        "season",
        "pitcher_id",
        "batter_id",
        "inning",
        "top_bottom",
        "balls_before",
        "strikes_before",
        "outs_before",
        "base_state",
        "game_month",
        "game_dayofweek",
        "li",
    ]
    rich_stats = _dup_stats(df, rich)

    # Season-level temporal split feasibility
    seasons = sorted(df["season"].dropna().unique().tolist())
    season_counts = df["season"].value_counts().sort_index().to_dict()

    # File order vs season order: does file appear season-sorted?
    season_monotonic = bool(df["season"].is_monotonic_increasing)

    lines: list[str] = []
    lines.append("# Validation Structure Audit")
    lines.append("")
    lines.append("Generated from `data/raw/train.csv` only. No order keys were invented.")
    lines.append("Validation strategy is **not** finalized by this report.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- train shape: `{df.shape}`")
    lines.append(f"- seasons present: `{seasons}`")
    lines.append(f"- season counts: `{season_counts}`")
    lines.append(f"- explicit game_id-like columns: `{game_like or 'none'}`")
    lines.append(f"- calendar/game-context fields (not game ids): `{calendar_game_fields}`")
    lines.append(f"- id-like columns: `{id_cols}`")
    lines.append(f"- official event datetime column: **none** (`time_col` remains null)")
    lines.append("")
    lines.append("## row_id as ordering signal")
    lines.append("")
    lines.append(f"- `row_id` is unique: `{stats[('row_id',)]['is_unique']}`")
    lines.append(f"- file order equals lexicographic `row_id` sort: `{lex_equals_file_order}`")
    lines.append(f"- all `row_id` have numeric suffix `_*digits`: `{has_numeric_suffix}`")
    lines.append(f"- numeric suffixes unique: `{suffix_unique}`")
    lines.append(f"- numeric suffixes monotonic in file order: `{suffix_monotonic}`")
    lines.append(f"- `season` monotonic in file order: `{season_monotonic}`")
    lines.append("")
    lines.append(
        "**Interpretation:** uniqueness of `row_id` does **not** by itself prove pitch chronology. "
        "Unless official docs state that `row_id` encodes time order, do not treat it as a temporal key."
    )
    lines.append("")
    lines.append("## Available time-related fields")
    lines.append("")
    lines.append("| Field | Role | Notes |")
    lines.append("|---|---|---|")
    lines.append("| `season` | coarse time axis | usable for season holdout / expanding-window |")
    lines.append("| `game_month` | within-season month | not a full date; ties across games possible |")
    lines.append("| `game_dayofweek` | weekday | weak; many collisions |")
    lines.append("| event datetime | — | **absent** |")
    lines.append("| pitch sequence within game | — | **no `pitch_no` / game_id in train** |")
    lines.append("")
    lines.append("## Available group keys")
    lines.append("")
    lines.append("| Key | Present | Notes |")
    lines.append("|---|---|---|")
    lines.append("| `pitcher_id` | yes | entity for pitcher-level diagnostics |")
    lines.append("| `batter_id` | yes | entity for matchup diagnostics |")
    lines.append("| `pitcher_team_id` / `batter_team_id` | yes | team groups |")
    lines.append("| game id | **no** | cannot block game leakage via explicit game groups |")
    lines.append("")
    lines.append("## Uniqueness / duplicate audit")
    lines.append("")
    lines.append("| Candidate key | Result |")
    lines.append("|---|---|")
    for cols_k, s in stats.items():
        lines.append(f"| `{list(cols_k)}` | {_fmt_stats(s)} |")
    lines.append(f"| `{rich}` | {_fmt_stats(rich_stats)} |")
    lines.append("")
    lines.append("## Pitch-order recoverability within a game")
    lines.append("")
    if game_like:
        lines.append(f"Candidate game columns found: {game_like}. Further inspection required.")
    else:
        lines.append(
            "No explicit game identifier exists in `train.csv`. "
            "Same pitcher–batter–inning–count context can still collide across games/days. "
            "**Within-game pitch order cannot be reconstructed from provided columns alone.**"
        )
    lines.append("")
    lines.append(f"Rich context key uniqueness: `{_fmt_stats(rich_stats)}`")
    if not rich_stats.get("is_unique", False):
        lines.append(
            f"Even a rich situational key still has "
            f"`{rich_stats.get('n_duplicate_rows')}` duplicate rows / "
            f"`{rich_stats.get('n_duplicate_groups')}` colliding groups "
            f"(max group size `{rich_stats.get('max_group_size')}`). "
            "Do **not** invent a synthetic order to break ties."
        )
    lines.append("")
    lines.append("## Temporal validation feasibility")
    lines.append("")
    lines.append("- **Season-level temporal split: possible.** `season` is present, non-null for audit purposes, and covers 2019–2024.")
    lines.append("- Expanding-window / latest-season holdout can be built on `season` without requiring pitch order.")
    lines.append("- Within-season chronological CV is **not supported** by available keys (no date/time/pitch_no/game_id).")
    lines.append("- `order_cols` in config remains `[]` until an officially unique chronological key is confirmed.")
    lines.append("")
    lines.append("## Game leakage blocking")
    lines.append("")
    lines.append("- **Not possible with an explicit game group key** (no game id).")
    lines.append("- Season split reduces cross-season leakage but does not prevent same-game rows from landing in both train and valid within a season.")
    lines.append("- Approximate proxies (team+month+dow+inning+...) are **not unique** and must not be treated as game ids.")
    lines.append("")
    lines.append("## Still unknown / not confirmed")
    lines.append("")
    lines.append("- Whether `row_id` numeric order matches true pitch chronology (official docs do not state this).")
    lines.append("- Mapping between main-data anonymized IDs and Trackman IDs (direct intersection was 0; out of scope here).")
    lines.append("- True within-game sequence for 2025 evaluation rows.")
    lines.append("- Final choice among season expanding / holdout / other schemes (explicitly **not finalized**).")
    lines.append("- Whether any undocumented tie-breaker exists outside the CSV columns.")
    lines.append("")
    lines.append("## Recommendation (non-final)")
    lines.append("")
    lines.append("1. Keep `data.order_cols: []` and `data.time_col: null` until a unique chronological key is proven.")
    lines.append("2. Prefer season-based temporal splits for local CV candidates.")
    lines.append("3. Do not fabricate pitch order from `row_id` or colliding situational keys.")
    lines.append("4. Do not claim game-leakage-free validation without a game identifier.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(f"seasons={seasons}")
    print(f"row_id unique={stats[('row_id',)]['is_unique']}")
    print(f"lex_equals_file_order={lex_equals_file_order}")
    print(f"suffix_monotonic={suffix_monotonic}")
    print(f"game_like={game_like}")
    print(f"rich_unique={rich_stats.get('is_unique')} dup_rows={rich_stats.get('n_duplicate_rows')}")


if __name__ == "__main__":
    main()
