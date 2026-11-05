"""Expand fold_comparison EDA to seasons 2019-2024."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dir, load_config, resolve_path

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
YEAR_COLS = [str(y) for y in YEARS]


def add_row(
    rows: list[dict],
    *,
    section: str,
    source: str,
    feature_group: str,
    feature: str,
    metric: str,
    values_by_year: dict[int, float],
    note: str = "",
) -> None:
    r: dict = {
        "section": section,
        "source": source,
        "feature_group": feature_group,
        "feature": feature,
        "metric": metric,
        "note": note,
    }
    for y in YEARS:
        r[str(y)] = values_by_year.get(y, np.nan)
    # refs used for change detection
    v22, v23 = r.get("2022"), r.get("2023")
    if pd.notna(v22) and pd.notna(v23):
        r["diff_2023_minus_2022"] = float(v23) - float(v22)
        r["abs_rel_diff_2022_to_2023"] = abs(float(v23) - float(v22)) / (abs(float(v22)) + 1e-12)
    else:
        r["diff_2023_minus_2022"] = np.nan
        r["abs_rel_diff_2022_to_2023"] = np.nan

    early = [r[str(y)] for y in (2019, 2020, 2021, 2022) if pd.notna(r[str(y)])]
    late = [r[str(y)] for y in (2023, 2024) if pd.notna(r[str(y)])]
    if early and late:
        early_mean = float(np.mean(early))
        late_mean = float(np.mean(late))
        early_std = float(np.std(early, ddof=0))
        r["early_2019_2022_mean"] = early_mean
        r["late_2023_2024_mean"] = late_mean
        r["diff_late_minus_early"] = late_mean - early_mean
        r["abs_rel_diff_regime"] = abs(late_mean - early_mean) / (abs(early_mean) + 1e-12)
        r["early_cv"] = early_std / (abs(early_mean) + 1e-12)
    else:
        r["early_2019_2022_mean"] = np.nan
        r["late_2023_2024_mean"] = np.nan
        r["diff_late_minus_early"] = np.nan
        r["abs_rel_diff_regime"] = np.nan
        r["early_cv"] = np.nan
    rows.append(r)


def is_bounded(r: dict) -> bool:
    vals = [r[c] for c in YEAR_COLS]
    return all(pd.notna(v) and 0.0 <= float(v) <= 1.0001 for v in vals)


def flag_jump_2022_2023(r: dict) -> bool:
    if pd.isna(r.get("abs_rel_diff_2022_to_2023")):
        return False
    m = r["metric"]
    d = abs(r["diff_2023_minus_2022"]) if pd.notna(r["diff_2023_minus_2022"]) else 0.0
    ar = float(r["abs_rel_diff_2022_to_2023"])
    if m == "missing_rate":
        return d >= 0.005
    if m == "count":
        return ar >= 0.05
    if m == "std":
        return ar >= 0.05
    if is_bounded(r):
        return d >= 0.015 or ar >= 0.05
    return ar >= 0.03


def flag_regime_2023(r: dict) -> bool:
    """Stable 2019-2022, different from 2023+."""
    if pd.isna(r.get("abs_rel_diff_regime")) or pd.isna(r.get("early_cv")):
        return False
    m = r["metric"]
    # require early years relatively stable
    early_stable = float(r["early_cv"]) <= 0.08
    if m == "missing_rate":
        early_stable = float(r["early_cv"]) <= 0.5 or abs(r["early_2019_2022_mean"]) < 0.01
        return early_stable and abs(r["diff_late_minus_early"]) >= 0.005
    if m == "count":
        return early_stable and float(r["abs_rel_diff_regime"]) >= 0.05
    if m == "std":
        return early_stable and float(r["abs_rel_diff_regime"]) >= 0.05
    if is_bounded(r):
        return early_stable and (
            abs(r["diff_late_minus_early"]) >= 0.015 or float(r["abs_rel_diff_regime"]) >= 0.05
        )
    return early_stable and float(r["abs_rel_diff_regime"]) >= 0.03


def main() -> None:
    cfg = load_config("configs/baseline.yaml")
    train = pd.read_csv(
        resolve_path(cfg["data"]["train_path"]), encoding="utf-8-sig", low_memory=False
    )
    tm = pd.read_csv(
        resolve_path("data/raw/trackman_history.csv"), encoding="utf-8-sig", low_memory=False
    )
    train = train.loc[train["season"].isin(YEARS)].copy()
    tm = tm.loc[tm["season"].isin(YEARS)].copy()

    rows: list[dict] = []

    n_train = {int(y): int((train["season"] == y).sum()) for y in YEARS}
    n_tm = {int(y): int((tm["season"] == y).sum()) for y in YEARS}
    add_row(
        rows,
        section="sample_count",
        source="train.csv",
        feature_group="sample",
        feature="n_rows",
        metric="count",
        values_by_year=n_train,
        note="fold/season row count (main label table)",
    )
    add_row(
        rows,
        section="sample_count",
        source="trackman_history.csv",
        feature_group="sample",
        feature="n_rows",
        metric="count",
        values_by_year=n_tm,
        note="Trackman log rows (not 1:1 with train)",
    )

    tgt: dict[int, dict[str, float]] = {}
    for y in YEARS:
        s = train.loc[train["season"] == y, "control_success"]
        tgt[y] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "rate_0": float((s == 0).mean()),
            "rate_1": float((s == 1).mean()),
            "missing_rate": float(s.isna().mean()),
        }
    for metric in ["mean", "std", "rate_0", "rate_1", "missing_rate"]:
        add_row(
            rows,
            section="target",
            source="train.csv",
            feature_group="target",
            feature="control_success",
            metric=metric,
            values_by_year={y: tgt[y][metric] for y in YEARS},
            note="BSS label; mean == P(y=1)",
        )

    for c in ["balls_before", "strikes_before", "outs_before"]:
        means, stds, miss = {}, {}, {}
        for y in YEARS:
            s = train.loc[train["season"] == y, c]
            means[y] = float(s.mean())
            stds[y] = float(s.std(ddof=0))
            miss[y] = float(s.isna().mean())
        add_row(rows, section="feature", source="train.csv", feature_group="count", feature=c, metric="mean", values_by_year=means)
        add_row(rows, section="feature", source="train.csv", feature_group="count", feature=c, metric="std", values_by_year=stds)
        add_row(rows, section="feature", source="train.csv", feature_group="count", feature=c, metric="missing_rate", values_by_year=miss)

    tm_groups = {
        "release_point": ["rel_height", "rel_side", "extension", "rel_speed"],
        "spin": ["spin_rate", "induced_vert_break", "horz_break"],
        "location": ["zone_speed"],
    }
    for group, cols in tm_groups.items():
        note_base = (
            "PlateLocX/Y absent; zone_speed used as plate-adjacent proxy"
            if group == "location"
            else ""
        )
        for c in cols:
            means, stds, miss = {}, {}, {}
            for y in YEARS:
                s = pd.to_numeric(tm.loc[tm["season"] == y, c], errors="coerce")
                means[y] = float(s.mean())
                stds[y] = float(s.std(ddof=0))
                miss[y] = float(s.isna().mean())
            add_row(rows, section="feature", source="trackman_history.csv", feature_group=group, feature=c, metric="mean", values_by_year=means, note=note_base)
            add_row(rows, section="feature", source="trackman_history.csv", feature_group=group, feature=c, metric="std", values_by_year=stds, note=note_base)
            add_row(rows, section="feature", source="trackman_history.csv", feature_group=group, feature=c, metric="missing_rate", values_by_year=miss, note=note_base)

    tm["_pt"] = tm["pitch_type_group"].astype(str).str.lower().str.strip()
    for cat in ["fastball", "breaking", "offspeed", "other"]:
        rates = {y: float((tm.loc[tm["season"] == y, "_pt"] == cat).mean()) for y in YEARS}
        add_row(
            rows,
            section="feature",
            source="trackman_history.csv",
            feature_group="pitch_type",
            feature=f"pitch_type_group__{cat}_rate",
            metric="mean",
            values_by_year=rates,
            note="share of pitches",
        )
    add_row(
        rows,
        section="feature",
        source="trackman_history.csv",
        feature_group="pitch_type",
        feature="pitch_type_group",
        metric="missing_rate",
        values_by_year={y: float(tm.loc[tm["season"] == y, "pitch_type_group"].isna().mean()) for y in YEARS},
    )

    for c in ["tagged_pitch_type", "auto_pitch_type"]:
        miss = {y: float(tm.loc[tm["season"] == y, c].isna().mean()) for y in YEARS}
        mode_share = {}
        for y in YEARS:
            s = tm.loc[tm["season"] == y, c]
            mode_share[y] = (
                float(s.value_counts(normalize=True).iloc[0]) if not s.dropna().empty else np.nan
            )
        add_row(
            rows,
            section="feature",
            source="trackman_history.csv",
            feature_group="pitch_type",
            feature=f"{c}__top_category_share",
            metric="mean",
            values_by_year=mode_share,
            note="share of most frequent category",
        )
        add_row(
            rows,
            section="feature",
            source="trackman_history.csv",
            feature_group="pitch_type",
            feature=c,
            metric="missing_rate",
            values_by_year=miss,
        )

    df = pd.DataFrame(rows)
    df["jump_2022_to_2023"] = df.apply(flag_jump_2022_2023, axis=1)
    df["regime_shift_from_2023"] = df.apply(flag_regime_2023, axis=1)

    jump = df.loc[df["jump_2022_to_2023"]].sort_values(
        "abs_rel_diff_2022_to_2023", ascending=False
    )
    regime = df.loc[df["regime_shift_from_2023"]].sort_values(
        "abs_rel_diff_regime", ascending=False
    )

    summary_rows: list[dict] = []
    for _, r in jump.iterrows():
        direction = "up" if r["diff_2023_minus_2022"] > 0 else "down"
        summary_rows.append(
            {
                **{k: r[k] for k in df.columns},
                "section": "SUMMARY_JUMP_2022_TO_2023",
                "note": (
                    f"2022→2023 {direction}; |rel|={r['abs_rel_diff_2022_to_2023']:.4f}. "
                    + (r["note"] or "")
                ),
            }
        )
    for _, r in regime.iterrows():
        direction = "higher" if r["diff_late_minus_early"] > 0 else "lower"
        summary_rows.append(
            {
                **{k: r[k] for k in df.columns},
                "section": "SUMMARY_REGIME_FROM_2023",
                "note": (
                    f"2019-2022 stable (cv={r['early_cv']:.4f}); 2023+ {direction} "
                    f"(|rel|={r['abs_rel_diff_regime']:.4f}). " + (r["note"] or "")
                ),
            }
        )

    order_section = {"sample_count": 0, "target": 1, "feature": 2}
    order_group = {
        "sample": 0,
        "target": 1,
        "count": 2,
        "release_point": 3,
        "spin": 4,
        "location": 5,
        "pitch_type": 6,
    }
    detail = df.copy()
    detail["_s"] = detail["section"].map(order_section).fillna(9)
    detail["_g"] = detail["feature_group"].map(order_group).fillna(9)
    detail = detail.sort_values(["_s", "_g", "feature", "metric"]).drop(columns=["_s", "_g"])

    out = pd.concat([pd.DataFrame(summary_rows), detail], ignore_index=True) if summary_rows else detail
    cols = [
        "section",
        "source",
        "feature_group",
        "feature",
        "metric",
        *YEAR_COLS,
        "diff_2023_minus_2022",
        "abs_rel_diff_2022_to_2023",
        "early_2019_2022_mean",
        "late_2023_2024_mean",
        "diff_late_minus_early",
        "abs_rel_diff_regime",
        "early_cv",
        "jump_2022_to_2023",
        "regime_shift_from_2023",
        "note",
    ]
    out = out[cols]

    eda = ensure_dir(resolve_path("outputs/eda"))
    csv_path = eda / "fold_comparison.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Highlight table for requested metrics
    highlight_keys = [
        ("control_success", "mean"),
        ("pitch_type_group__other_rate", "mean"),
        ("tagged_pitch_type__top_category_share", "mean"),
        ("pitch_type_group__offspeed_rate", "mean"),
        ("rel_height", "mean"),
        ("rel_height", "std"),
        ("rel_height", "missing_rate"),
        ("rel_side", "mean"),
        ("rel_side", "std"),
        ("rel_side", "missing_rate"),
        ("extension", "mean"),
        ("extension", "std"),
        ("extension", "missing_rate"),
        ("rel_speed", "mean"),
        ("rel_speed", "std"),
        ("rel_speed", "missing_rate"),
        ("spin_rate", "mean"),
        ("spin_rate", "std"),
        ("spin_rate", "missing_rate"),
        ("induced_vert_break", "mean"),
        ("induced_vert_break", "std"),
        ("induced_vert_break", "missing_rate"),
        ("horz_break", "mean"),
        ("horz_break", "std"),
        ("horz_break", "missing_rate"),
        ("zone_speed", "mean"),
        ("zone_speed", "std"),
        ("zone_speed", "missing_rate"),
    ]

    md: list[str] = []
    md.append("# Fold Comparison (2019–2024)")
    md.append("")
    md.append(f"- csv: `{csv_path}`")
    md.append("- sources: `train.csv` (target/count), `trackman_history.csv` (release/spin/location/pitch_type)")
    md.append("")
    md.append("## Highlight: requested year trends")
    md.append("")
    md.append(
        "| feature | metric | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | "
        "Δ2023−2022 | jump? | regime_from_2023? |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for feat, metric in highlight_keys:
        hit = df.loc[(df["feature"] == feat) & (df["metric"] == metric)]
        if hit.empty:
            continue
        r = hit.iloc[0]

        def f(v: object) -> str:
            if pd.isna(v):
                return ""
            return f"{float(v):.6g}"

        md.append(
            "| "
            + " | ".join(
                [
                    f"`{feat}`",
                    metric,
                    *[f(r[str(y)]) for y in YEARS],
                    f(r["diff_2023_minus_2022"]),
                    str(bool(r["jump_2022_to_2023"])),
                    str(bool(r["regime_shift_from_2023"])),
                ]
            )
            + " |"
        )

    md.append("")
    md.append("## Sample counts")
    md.append("")
    md.append("| season | train n_rows | trackman n_rows |")
    md.append("|---:|---:|---:|")
    for y in YEARS:
        md.append(f"| {y} | {n_train[y]:,} | {n_tm[y]:,} |")

    md.append("")
    md.append("## 2022→2023 sharp changes")
    md.append("")
    if jump.empty:
        md.append("- (none under thresholds)")
    else:
        md.append("| feature | metric | 2022 | 2023 | Δ | |rel| |")
        md.append("|---|---|---:|---:|---:|---:|")
        for _, r in jump.iterrows():
            md.append(
                f"| `{r['feature']}` | {r['metric']} | {float(r['2022']):.6g} | "
                f"{float(r['2023']):.6g} | {float(r['diff_2023_minus_2022']):.6g} | "
                f"{float(r['abs_rel_diff_2022_to_2023']):.4f} |"
            )

    md.append("")
    md.append("## Stable 2019–2022, different from 2023+")
    md.append("")
    md.append(
        "Criterion: early years relatively stable (`early_cv` low) and "
        "mean(2023,2024) differs from mean(2019–2022) beyond thresholds."
    )
    md.append("")
    if regime.empty:
        md.append("- (none under thresholds)")
    else:
        md.append(
            "| feature | metric | early_mean | late_mean | Δ | |rel| | early_cv |"
        )
        md.append("|---|---|---:|---:|---:|---:|---:|")
        for _, r in regime.iterrows():
            md.append(
                f"| `{r['feature']}` | {r['metric']} | "
                f"{float(r['early_2019_2022_mean']):.6g} | "
                f"{float(r['late_2023_2024_mean']):.6g} | "
                f"{float(r['diff_late_minus_early']):.6g} | "
                f"{float(r['abs_rel_diff_regime']):.4f} | "
                f"{float(r['early_cv']):.4f} |"
            )

    md.append("")
    md_path = eda / "fold_comparison.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"jump={len(jump)} regime={len(regime)}")
    print("target means", {y: round(tgt[y]["mean"], 6) for y in YEARS})


if __name__ == "__main__":
    main()
