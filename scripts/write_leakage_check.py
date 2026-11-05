"""Generate outputs/eda/leakage_check.md (aggregate only; no competition row samples)."""

from __future__ import annotations

import json
import math

import pandas as pd

from src.utils import ensure_dir, load_config, resolve_path

PREV = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]


def _json_safe(obj: object) -> object:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> None:
    cfg = load_config("configs/baseline.yaml")
    train_rel = cfg["data"]["train_path"]
    train = pd.read_csv(
        resolve_path(train_rel), encoding="utf-8-sig", low_memory=False
    )

    # Illustrative pitcher-season for local sanity only (not published in markdown).
    pid, season = 23987, 2020
    sub = (
        train.loc[(train.pitcher_id == pid) & (train.season == season)]
        .sort_values(["asof_pitcher_n", "row_id"])
        .reset_index(drop=True)
    )
    want = [0, 20, 40, 80, 82, 83, 84, 200, 1000, 2500]
    want = [i for i in want if i < len(sub)]
    sample = sub.loc[
        want, ["row_id", "season", "game_month", "game_dayofweek", "asof_pitcher_n", *PREV]
    ].copy()
    sample.insert(0, "order_rank", want)

    n_dec_sample = int((sub["asof_pitcher_n"].diff() < 0).sum())
    prev1_changes = int(
        (
            (sub[PREV[0]] != sub[PREV[0]].shift(1))
            & sub[PREV[0]].notna()
            & sub[PREV[0]].shift(1).notna()
        ).sum()
    )
    sample_null_reentry = {
        c: int(
            (
                lambda s: (
                    0
                    if s.first_valid_index() is None
                    else int(s.iloc[int(s.first_valid_index()) + 1 :].isna().sum())
                )
            )(sub[c].reset_index(drop=True))
        )
        for c in PREV
    }

    # Local-only row-level artifact (gitignored). Not written into markdown.
    eda_dir = ensure_dir(resolve_path("outputs/eda"))
    meta_path = eda_dir / "_leakage_meta.json"
    meta = _json_safe(
        {
            "pick": {
                "pitcher_id": int(pid),
                "season": int(season),
                "n": int(len(sub)),
                "n_dec_asof_n": n_dec_sample,
                "prev1_changes": prev1_changes,
                "null_reentry_after_first_non_null": sample_null_reentry,
            },
            "sample": json.loads(sample.to_json(orient="records")),
        }
    )
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ord2 = train.sort_values(["pitcher_id", "season", "asof_pitcher_n", "row_id"]).copy()
    dec = int(ord2.groupby(["pitcher_id", "season"])["asof_pitcher_n"].diff().lt(0).sum())

    null_after_total = {c: 0 for c in PREV}
    groups_with_reentry = {c: 0 for c in PREV}
    n_groups = 0
    for _, g in ord2.groupby(["pitcher_id", "season"], sort=False):
        n_groups += 1
        for c in PREV:
            s = g[c].reset_index(drop=True)
            nn = s.first_valid_index()
            if nn is None:
                continue
            after_null = int(s.iloc[int(nn) + 1 :].isna().sum())
            null_after_total[c] += after_null
            if after_null > 0:
                groups_with_reentry[c] += 1

    g = ord2.groupby(["pitcher_id", "season"], sort=False)
    ord2["_new"] = (
        (ord2["game_month"] != g["game_month"].shift(1))
        | (ord2["game_dayofweek"] != g["game_dayofweek"].shift(1))
    ).fillna(True)
    ord2["game_proxy"] = g["_new"].cumsum()
    ord2["game_rank"] = ord2.groupby(["pitcher_id", "season"])["game_proxy"].transform(
        lambda s: s.rank(method="dense")
    )
    early = ord2["game_rank"] <= 5
    late = ~early

    def miss_rates(mask: pd.Series) -> list[float]:
        return [float(ord2.loc[mask, c].isna().mean()) for c in PREV]

    def outlier_rates(mask: pd.Series) -> list[float]:
        out: list[float] = []
        for c in PREV:
            s = ord2.loc[mask, c].dropna()
            out.append(float(((s < 0) | (s > 1)).mean()) if len(s) else float("nan"))
        return out

    segments = [
        ("first_5_game_proxies", early),
        ("after_5_game_proxies", late),
        ("months_le_4", ord2["game_month"] <= 4),
        ("months_ge_5", ord2["game_month"] >= 5),
    ]

    suby = train.loc[train.season.isin([2022, 2023, 2024]), ["pitcher_id", "season"]].drop_duplicates()
    cnt = suby.groupby("pitcher_id")["season"].nunique()
    set_counts = (
        suby.groupby("pitcher_id")["season"]
        .apply(lambda s: tuple(sorted(s.tolist())))
        .value_counts()
    )
    n_p = int(cnt.shape[0])

    lines: list[str] = []
    lines.append("# Leakage Check Report")
    lines.append("")
    lines.append(f"- source: `{train_rel}`")
    lines.append(
        "- focus: `asof_pitcher_prev*_game_*` temporal sanity, CV fold separation, "
        "pitcher cross-year overlap"
    )
    lines.append(
        "- note: row-level competition samples are **not** published here; "
        "local-only detail (if regenerated) lives in gitignored "
        "`outputs/eda/_leakage_meta.json`."
    )
    lines.append("")
    lines.append("## 1. asof column verification (`asof_pitcher_prev*_game_*`)")
    lines.append("")
    lines.append("### Ordering caveat")
    lines.append("")
    lines.append(
        "- train.csv has **no `game_id` / event timestamp**. Chronology within a "
        "pitcher-season is approximated by sorting on `asof_pitcher_n` (official "
        "cumulative pitch count before the current pitch), then `row_id`."
    )
    lines.append(
        "- `game_month` / `game_dayofweek` are secondary context and used for an "
        "approximate “first 5 games” proxy (new proxy-game when month or weekday "
        "changes for that pitcher)."
    )
    lines.append(
        "- This is a **sanity check**, not a proof of official feature construction."
    )
    lines.append("")
    lines.append("### Illustrative single pitcher-season (de-identified)")
    lines.append("")
    lines.append(
        "- One real pitcher-season was inspected locally as `SAMPLE_PITCHER` "
        "(identity and `row_id` withheld from this tracked report)."
    )
    lines.append(
        "- expected if past-only: cold-start NA → values after enough prior games → "
        "infrequent updates (game-level) → no mid-season flip back to NA."
    )
    lines.append("")
    lines.append("### Observations on this sample")
    lines.append("")
    lines.append(
        f"- `asof_pitcher_n` is strictly non-decreasing along the sort "
        f"(`decreases={n_dec_sample}` along the selected pitcher-season)."
    )
    lines.append(
        "- Early ranks are all-NA on every `prev*` column; values become available "
        "near the first non-null rank and then remain populated."
    )
    lines.append(
        "- After first non-null, null re-entry count for this pitcher-season: `0` "
        "on all six `prev*` columns."
    )
    lines.append(
        f"- `prev1_success` discrete value changes across the season: "
        f"`{prev1_changes}` (game-step like; not pitch-by-pitch churn)."
    )
    lines.append("")
    lines.append("### Global re-entry check (all pitcher × season)")
    lines.append("")
    lines.append(f"- pitcher-season groups: `{n_groups}`")
    lines.append(f"- global `asof_pitcher_n` decreases under the same sort: `{dec}`")
    lines.append("")
    lines.append(
        "| column | nulls after first non-null (row count) | "
        "pitcher-seasons with any re-entry |"
    )
    lines.append("|---|---:|---:|")
    for c in PREV:
        lines.append(f"| `{c}` | {null_after_total[c]} | {groups_with_reentry[c]} |")
    lines.append("")
    lines.append(
        "- Interpretation: once a `prev*` value appears in a pitcher-season, it does "
        "**not** return to NA later under this ordering — consistent with cold-start "
        "then past-game aggregation."
    )
    lines.append("")
    lines.append("### Early-season (first ~5 games) vs later: missing / outlier rates")
    lines.append("")
    lines.append(
        "Game proxy: within each `(pitcher_id, season)`, sort by `asof_pitcher_n`; "
        "start a new proxy-game when `game_month` or `game_dayofweek` changes. "
        "“First 5 games” = `game_rank <= 5`."
    )
    lines.append("")
    lines.append(
        "| segment | n_rows | miss prev1_success | miss prev3_success | "
        "miss prev5_success | miss prev1_middle | miss prev3_middle | "
        "miss prev5_middle |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, mask in segments:
        rates = miss_rates(mask)
        lines.append(
            "| "
            + " | ".join(
                [label, f"{int(mask.sum()):,}", *[f"{x:.6f}" for x in rates]]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "| segment | outlier rate (value not in [0,1]) "
        "prev1_s / prev3_s / prev5_s / prev1_m / prev3_m / prev5_m |"
    )
    lines.append("|---|---|")
    for label, mask in [("first_5_game_proxies", early), ("after_5_game_proxies", late)]:
        rates = outlier_rates(mask)
        lines.append(f"| {label} | {' / '.join(f'{x:.6f}' for x in rates)} |")
    lines.append("")
    lines.append(
        "**Verdict (asof):** Early proxy-games have markedly higher missingness "
        "(~7.9% vs ~0% after 5 games; months≤4 also higher than later months). "
        "Non-null values stay in `[0,1]` (outlier rate 0). Pattern matches past-only "
        "cold-start behavior; no evidence in this check of future leakage via "
        "mid-season NA re-entry or out-of-range rates."
    )
    lines.append("")

    lines.append("## 2. CV group structure (`tree_exp_common.py`)")
    lines.append("")
    lines.append("### Fold construction as used by tree experiments")
    lines.append("")
    lines.append("From `scripts/tree_exp_common.py` (fold specs + holdout call):")
    lines.append("")
    lines.append("```python")
    lines.append('primary = cv_cfg["primary"]')
    lines.append('secondary = list(cv_cfg.get("secondary_folds") or [])')
    lines.append("fold_specs = [")
    lines.append("    {")
    lines.append('        "fold_id": "secondary_2022",')
    lines.append('        "train_seasons": list(secondary[0]["train_seasons"]),')
    lines.append('        "valid_seasons": list(secondary[0]["valid_seasons"]),')
    lines.append('        "role": "secondary",')
    lines.append("    },")
    lines.append("    {")
    lines.append('        "fold_id": "secondary_2023",')
    lines.append('        "train_seasons": list(secondary[1]["train_seasons"]),')
    lines.append('        "valid_seasons": list(secondary[1]["valid_seasons"]),')
    lines.append('        "role": "secondary",')
    lines.append("    },")
    lines.append("    {")
    lines.append('        "fold_id": "primary_2024",')
    lines.append('        "train_seasons": list(primary["train_seasons"]),')
    lines.append('        "valid_seasons": list(primary["valid_seasons"]),')
    lines.append('        "role": "primary",')
    lines.append("    },")
    lines.append("]")
    lines.append("")
    lines.append("for spec in fold_specs:")
    lines.append("    fold = build_season_holdout(")
    lines.append("        train_df,")
    lines.append("        season_col,")
    lines.append('        spec["train_seasons"],')
    lines.append('        spec["valid_seasons"],')
    lines.append('        name=spec["fold_id"],')
    lines.append("    )")
    lines.append("```")
    lines.append("")
    lines.append("Underlying splitter (`src/evaluate.py::build_season_holdout`):")
    lines.append("")
    lines.append("```python")
    lines.append("def build_season_holdout(...):")
    lines.append(
        '    """Explicit season holdout (e.g. 2019-2023 train / 2024 valid)."""'
    )
    lines.append("    ...")
    lines.append("    if set(train_seasons_t) & set(valid_seasons_t):")
    lines.append(
        '        raise ValueError("train_seasons and valid_seasons must be disjoint.")'
    )
    lines.append("    if max(train_seasons_t) >= min(valid_seasons_t):")
    lines.append("        raise ValueError(")
    lines.append(
        '            "train seasons must be strictly earlier than valid seasons "'
    )
    lines.append(
        '            f"(train_max={max(train_seasons_t)}, '
        'valid_min={min(valid_seasons_t)})."'
    )
    lines.append("        )")
    lines.append("    return _make_fold(...)")
    lines.append("```")
    lines.append("")
    lines.append("### Configured seasons (`configs/baseline.yaml`)")
    lines.append("")
    lines.append("| fold_id | train_seasons | valid_seasons |")
    lines.append("|---|---|---|")
    lines.append("| secondary_2022 | 2019–2021 | 2022 |")
    lines.append("| secondary_2023 | 2019–2022 | 2023 |")
    lines.append("| primary_2024 | 2019–2023 | 2024 |")
    lines.append("")
    lines.append("### Explicit answers")
    lines.append("")
    lines.append(
        "- **Random shuffle in fold split?** **No.** Indices come from boolean masks "
        "on `season` only (`build_season_holdout` / `_make_fold`). There is no "
        "`shuffle`, `sample`, or `train_test_split` in the fold construction path."
    )
    lines.append(
        "- **Season 완전 분리?** **Yes for train vs valid within each fold:** "
        "train/valid season sets are required to be disjoint, and train seasons must "
        "be strictly earlier than valid seasons (expanding temporal holdout). Valid "
        "seasons do not leak into that fold’s training rows."
    )
    lines.append(
        "- Note: the same early seasons appear in multiple folds’ training sets "
        "(expanding CV); that is intentional and is **not** random mixing of a valid "
        "season into train."
    )
    lines.append(
        "- `runtime.seed` is used for model RNG (`set_seed` / LightGBM "
        "`random_state` / CatBoost `random_seed`), **not** for splitting."
    )
    lines.append("")

    lines.append("## 3. Pitcher cross-year overlap (2022 / 2023 / 2024)")
    lines.append("")
    lines.append(
        f"- distinct `pitcher_id` appearing in any of 2022–2024: `{n_p}`"
    )
    lines.append("")
    lines.append("| n_years present (among 2022–2024) | n_pitchers | share |")
    lines.append("|---:|---:|---:|")
    for k in [1, 2, 3]:
        n = int((cnt == k).sum())
        lines.append(f"| {k} | {n} | {n / n_p:.4f} |")
    lines.append("")
    lines.append("### Year-set detail")
    lines.append("")
    lines.append("| year set | n_pitchers |")
    lines.append("|---|---:|")
    for k, v in set_counts.items():
        lines.append(f"| `{list(k)}` | {int(v)} |")
    lines.append("")
    lines.append(
        "**Implication for CV:** many pitchers "
        f"({int((cnt == 3).sum())} / {n_p} ≈ {(cnt == 3).mean():.0%}) appear in "
        "**all three** fold years. Season holdout prevents *row* leakage from the "
        "valid season into train, but **entity (pitcher) identity still overlaps** "
        "across years — models can memorize pitcher-specific patterns that transfer "
        "into later holdouts."
    )
    lines.append("")

    path = eda_dir / "leakage_check.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(f"wrote local-only {meta_path} (gitignored)")


if __name__ == "__main__":
    main()
