"""Target regime check: correlations, pitch-type blocker, count grids."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ensure_dir, load_config, resolve_path

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
RATE_COLS = [
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
]


def corr_pair(y: pd.Series, x: pd.Series) -> dict[str, float]:
    mask = y.notna() & x.notna()
    n = int(mask.sum())
    if n < 2 or y.loc[mask].nunique() < 2 or x.loc[mask].nunique() < 2:
        return {"n": n, "pearson": np.nan, "spearman": np.nan}
    a = y.loc[mask].astype(float)
    b = x.loc[mask].astype(float)
    return {
        "n": n,
        "pearson": float(a.corr(b, method="pearson")),
        "spearman": float(a.corr(b, method="spearman")),
    }


def main() -> None:
    cfg = load_config("configs/baseline.yaml")
    train = pd.read_csv(
        resolve_path(cfg["data"]["train_path"]),
        encoding="utf-8-sig",
        low_memory=False,
        usecols=[
            "season",
            "control_success",
            "balls_before",
            "strikes_before",
            "pitcher_id",
            *RATE_COLS,
        ],
    )
    tm = pd.read_csv(
        resolve_path("data/raw/trackman_history.csv"),
        encoding="utf-8-sig",
        low_memory=False,
        usecols=["pitcher_trackman_id", "season", "pitch_type_group"],
    )

    inter = len(set(train["pitcher_id"]) & set(tm["pitcher_trackman_id"]))

    lines: list[str] = []
    lines.append("# Target Regime Check (2022→2023 hypothesis inputs)")
    lines.append("")
    lines.append(
        "- hypothesis under inspection: target construction may have changed "
        "between 2022 and 2023"
    )
    lines.append("- **no adopt/reject judgment in this document — results only**")
    lines.append(f"- train source: `{resolve_path(cfg['data']['train_path'])}`")
    lines.append("")

    # --- 1 correlations ---
    lines.append("## 1. Year-wise correlation: `control_success` vs asof pitcher rates")
    lines.append("")
    lines.append(
        "Pairwise Pearson / Spearman within each season "
        "(rows with NA in either side dropped)."
    )
    lines.append("")
    lines.append(
        "| season | feature | n | pearson | spearman |"
    )
    lines.append("|---:|---|---:|---:|---:|")

    corr_rows = []
    for y in YEARS:
        part = train.loc[train["season"] == y]
        for c in RATE_COLS:
            stats = corr_pair(part["control_success"], part[c])
            corr_rows.append({"season": y, "feature": c, **stats})
            lines.append(
                f"| {y} | `{c}` | {stats['n']:,} | "
                f"{stats['pearson']:.6f} | {stats['spearman']:.6f} |"
            )

    lines.append("")
    lines.append("### Wide view (Pearson)")
    lines.append("")
    wide_p = (
        pd.DataFrame(corr_rows)
        .pivot(index="feature", columns="season", values="pearson")
        .reindex(RATE_COLS)
    )
    lines.append("| feature | " + " | ".join(str(y) for y in YEARS) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(YEARS)) + "|")
    for feat in RATE_COLS:
        vals = [f"{float(wide_p.loc[feat, y]):.6f}" for y in YEARS]
        lines.append(f"| `{feat}` | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("### Wide view (Spearman)")
    lines.append("")
    wide_s = (
        pd.DataFrame(corr_rows)
        .pivot(index="feature", columns="season", values="spearman")
        .reindex(RATE_COLS)
    )
    lines.append("| feature | " + " | ".join(str(y) for y in YEARS) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(YEARS)) + "|")
    for feat in RATE_COLS:
        vals = [f"{float(wide_s.loc[feat, y]):.6f}" for y in YEARS]
        lines.append(f"| `{feat}` | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("### 2022 vs 2023 Pearson delta (2023 − 2022)")
    lines.append("")
    lines.append("| feature | pearson_2022 | pearson_2023 | delta |")
    lines.append("|---|---:|---:|---:|")
    for feat in RATE_COLS:
        p22 = float(wide_p.loc[feat, 2022])
        p23 = float(wide_p.loc[feat, 2023])
        lines.append(f"| `{feat}` | {p22:.6f} | {p23:.6f} | {p23 - p22:.6f} |")

    # --- 2 pitch type ---
    lines.append("")
    lines.append(
        "## 2. `control_success=1` rate by `pitch_type_group` × season"
    )
    lines.append("")
    lines.append(
        "**Not computable at pitch level with current keys.** "
        "`control_success` lives only in `train.csv`; `pitch_type_group` lives only "
        "in `trackman_history.csv`. There is no shared join key:"
    )
    lines.append("")
    lines.append(
        f"- `|pitcher_id ∩ pitcher_trackman_id| = {inter}` "
        "(distinct ID namespaces)"
    )
    lines.append("- no `game_id` / `trackman_game_id` on train")
    lines.append(
        "- training pipeline does not join Trackman into train "
        "(see prior Trackman-join check)"
    )
    lines.append("")
    lines.append(
        "Therefore a year × `pitch_type_group` cross-tab of "
        "`P(control_success=1)` **cannot be produced** without an external mapping "
        "that is not in the released files."
    )
    lines.append("")
    lines.append("### Contextual year-level margins (not a joint cross-tab)")
    lines.append("")
    lines.append(
        "Shown separately so distributions can still be compared side-by-side; "
        "these are **not** success rates within pitch type."
    )
    lines.append("")
    lines.append("| season | train P(y=1) | tm share fastball | breaking | offspeed | other | tm n |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    tm["_pt"] = tm["pitch_type_group"].astype(str).str.lower().str.strip()
    for y in YEARS:
        tr = train.loc[train["season"] == y, "control_success"]
        part = tm.loc[tm["season"] == y]
        n = len(part)
        shares = {
            cat: float((part["_pt"] == cat).mean()) if n else np.nan
            for cat in ["fastball", "breaking", "offspeed", "other"]
        }
        lines.append(
            f"| {y} | {float(tr.mean()):.6f} | {shares['fastball']:.6f} | "
            f"{shares['breaking']:.6f} | {shares['offspeed']:.6f} | "
            f"{shares['other']:.6f} | {n:,} |"
        )

    # --- 3 count grids ---
    lines.append("")
    lines.append(
        "## 3. `P(control_success=1)` by `balls_before` × `strikes_before` "
        "(2022 vs 2023)"
    )
    lines.append("")

    def count_table(year: int) -> pd.DataFrame:
        part = train.loc[train["season"] == year]
        g = (
            part.groupby(["balls_before", "strikes_before"], dropna=False)[
                "control_success"
            ]
            .agg(n="size", rate="mean")
            .reset_index()
        )
        return g

    t22 = count_table(2022).rename(columns={"n": "n_2022", "rate": "rate_2022"})
    t23 = count_table(2023).rename(columns={"n": "n_2023", "rate": "rate_2023"})
    merged = t22.merge(t23, on=["balls_before", "strikes_before"], how="outer")
    merged["delta_rate_2023_minus_2022"] = merged["rate_2023"] - merged["rate_2022"]
    merged = merged.sort_values(["balls_before", "strikes_before"])

    lines.append(
        "| balls_before | strikes_before | n_2022 | rate_2022 | "
        "n_2023 | rate_2023 | Δ rate (2023−2022) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in merged.iterrows():
        def f(v: object, nd: int = 6) -> str:
            if pd.isna(v):
                return ""
            if nd == 0:
                return f"{int(v):,}"
            return f"{float(v):.{nd}f}"

        lines.append(
            f"| {int(r['balls_before']) if pd.notna(r['balls_before']) else ''} | "
            f"{int(r['strikes_before']) if pd.notna(r['strikes_before']) else ''} | "
            f"{f(r['n_2022'], 0)} | {f(r['rate_2022'])} | "
            f"{f(r['n_2023'], 0)} | {f(r['rate_2023'])} | "
            f"{f(r['delta_rate_2023_minus_2022'])} |"
        )

    lines.append("")
    lines.append("### Margin by `balls_before` only")
    lines.append("")
    lines.append("| balls_before | rate_2022 | n_2022 | rate_2023 | n_2023 | Δ |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for b in sorted(train.loc[train["season"].isin([2022, 2023]), "balls_before"].dropna().unique()):
        r22 = train.loc[(train["season"] == 2022) & (train["balls_before"] == b), "control_success"]
        r23 = train.loc[(train["season"] == 2023) & (train["balls_before"] == b), "control_success"]
        lines.append(
            f"| {int(b)} | {r22.mean():.6f} | {len(r22):,} | "
            f"{r23.mean():.6f} | {len(r23):,} | {r23.mean() - r22.mean():.6f} |"
        )

    lines.append("")
    lines.append("### Margin by `strikes_before` only")
    lines.append("")
    lines.append("| strikes_before | rate_2022 | n_2022 | rate_2023 | n_2023 | Δ |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for s in sorted(train.loc[train["season"].isin([2022, 2023]), "strikes_before"].dropna().unique()):
        r22 = train.loc[(train["season"] == 2022) & (train["strikes_before"] == s), "control_success"]
        r23 = train.loc[(train["season"] == 2023) & (train["strikes_before"] == s), "control_success"]
        lines.append(
            f"| {int(s)} | {r22.mean():.6f} | {len(r22):,} | "
            f"{r23.mean():.6f} | {len(r23):,} | {r23.mean() - r22.mean():.6f} |"
        )

    lines.append("")
    lines.append("### Overall rates")
    lines.append("")
    lines.append("| season | n | P(control_success=1) |")
    lines.append("|---:|---:|---:|")
    for y in YEARS:
        s = train.loc[train["season"] == y, "control_success"]
        lines.append(f"| {y} | {len(s):,} | {float(s.mean()):.6f} |")

    lines.append("")
    path = ensure_dir(resolve_path("outputs/eda")) / "target_regime_check.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
