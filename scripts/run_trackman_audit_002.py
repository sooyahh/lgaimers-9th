"""TM-AUDIT-002: Contextual Trackman Signal Audit.

Question: does Trackman physics vary by main-shared context (count / outs /
inning) beyond the global pitch-group mean, and is that variation stable across
temporal cutoffs 2022 / 2023 / 2024?

This is an AUDIT, not a model experiment. It:
  - reads only Trackman history (season < valid_year, never 2025),
  - uses only context columns that also exist on the main row
    (balls_before, strikes_before, outs_before, inning, top_bottom),
  - never joins player IDs, never maps hand, never touches test.csv,
  - never trains a model and never writes an EXP/FE/TM artifact.

Outputs:
  outputs/analysis/trackman_context_profiles_002.csv
  outputs/analysis/trackman_context_stability_002.csv
  outputs/analysis/trackman_context_summary_002.csv
  outputs/analysis/trackman_audit_002/*.png  (<= 4 plots)
  reports/trackman_audit_002.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackman_profiles import (  # reuse fold-safe cutoff logic; do not modify it
    allowed_trackman_seasons,
    assert_trackman_temporal_safety,
)

TRACKMAN_PATH = ROOT / "data/raw/trackman_history.csv"
ANALYSIS_DIR = ROOT / "outputs" / "analysis"
PLOT_DIR = ANALYSIS_DIR / "trackman_audit_002"

CUTOFFS = (2022, 2023, 2024)  # validation-year perspectives; TM season < cutoff
# Independent single seasons for the non-overlapping stability check. These are
# exactly the seasons that ever enter an allowed profile (season < 2024). 2024 is
# never used because the largest cutoff (<2024) only consumes 2019-2023, and 2025
# is forbidden everywhere.
NONOVERLAP_SEASONS = (2019, 2020, 2021, 2022, 2023)
TOP_CANDIDATES = (  # (context_type, pitch_group, variable) prioritised in the report
    ("count", "breaking", "IVB"),
    ("count", "fastball", "rel_speed"),
    ("count", "fastball", "IVB"),
)
CORE_GROUPS = ("fastball", "breaking", "offspeed")  # 'other' excluded

# Trackman schema is the source of truth for physical column names.
PHYSICS = {
    "IVB": "induced_vert_break",
    "HB": "horz_break",
    "rel_speed": "rel_speed",
    "spin_rate": "spin_rate",
    "extension": "extension",
}
PRIORITY_PHYSICS = ("IVB", "HB")  # primary; rel_speed/spin/extension are diagnostics

CONTEXT_COLS = ["balls_before", "strikes_before", "outs_before", "inning", "top_bottom"]
LEGAL_BALLS = (0, 1, 2, 3)
LEGAL_STRIKES = (0, 1, 2)
LEGAL_OUTS = (0, 1, 2)

# Sample-size diagnostic threshold. Chosen AFTER inspecting the n distribution
# (see report); a cell below this is flagged 'sparse' for stability interpretation
# only. It is never a performance/selection metric.
SPARSE_N = 1000


def load_trackman() -> pd.DataFrame:
    usecols = ["season", "pitch_type_group"] + CONTEXT_COLS + list(PHYSICS.values())
    df = pd.read_csv(TRACKMAN_PATH, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    df["pitch_type_group"] = df["pitch_type_group"].astype(str).str.lower().str.strip()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    for c in CONTEXT_COLS[:-1]:  # numeric context cols
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["top_bottom"] = df["top_bottom"].astype(str).str.strip()
    for v in PHYSICS.values():
        df[v] = pd.to_numeric(df[v], errors="coerce")
    return df


def inning_bucket(inning: pd.Series) -> pd.Series:
    b = pd.Series(pd.NA, index=inning.index, dtype="object")
    b[(inning >= 1) & (inning <= 3)] = "early_1_3"
    b[(inning >= 4) & (inning <= 6)] = "middle_4_6"
    b[inning >= 7] = "late_7plus"
    return b


def build_context_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the audit context keys. Only legal counts/outs are kept per context."""
    out = df.copy()
    legal_count = out["balls_before"].isin(LEGAL_BALLS) & out["strikes_before"].isin(LEGAL_STRIKES)
    out["count"] = np.where(
        legal_count,
        out["balls_before"].astype("Int64").astype(str) + "-" + out["strikes_before"].astype("Int64").astype(str),
        pd.NA,
    )
    out.loc[~legal_count, "count"] = pd.NA
    out["outs"] = np.where(out["outs_before"].isin(LEGAL_OUTS), out["outs_before"].astype("Int64").astype(str), pd.NA)
    out.loc[~out["outs_before"].isin(LEGAL_OUTS), "outs"] = pd.NA
    out["inning_bucket"] = inning_bucket(out["inning"])
    out["top_bottom_ctx"] = out["top_bottom"].where(out["top_bottom"].isin(["Top", "Bottom"]), pd.NA)
    return out


CONTEXT_TYPES = {
    "count": "count",
    "outs": "outs",
    "inning_bucket": "inning_bucket",
    "top_bottom": "top_bottom_ctx",
}


def cell_stats(values: pd.Series) -> dict[str, float]:
    n = int(len(values))
    obs = values.dropna()
    n_missing = n - int(len(obs))
    out = {
        "n": n,
        "missing_n": n_missing,
        "missing_rate": float(n_missing / n) if n else float("nan"),
        "mean": float("nan"),
        "std": float("nan"),
        "median": float("nan"),
        "p10": float("nan"),
        "p90": float("nan"),
    }
    if not obs.empty:
        out.update(
            {
                "mean": float(obs.mean()),
                "std": float(obs.std(ddof=0)),
                "median": float(obs.median()),
                "p10": float(obs.quantile(0.10)),
                "p90": float(obs.quantile(0.90)),
            }
        )
    return out


def pooled_std(cell_ns: list[int], cell_vars: list[float]) -> float:
    """Classic pooled within-cell SD across the context values of one group."""
    num = 0.0
    den = 0.0
    for n, var in zip(cell_ns, cell_vars):
        if n is None or not np.isfinite(var) or n <= 1:
            continue
        num += (n - 1) * var
        den += n - 1
    if den <= 0:
        return float("nan")
    return float(np.sqrt(num / den))


def compute_profiles(dfc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (profiles, summary). profiles = one row per
    (cutoff, context_type, context_value, pitch_group, variable)."""
    prof_rows: list[dict[str, Any]] = []
    summ_rows: list[dict[str, Any]] = []

    for cutoff in CUTOFFS:
        allowed = list(allowed_trackman_seasons(cutoff))
        hist = dfc[dfc["season"].isin(allowed)].copy()
        assert_trackman_temporal_safety(
            hist[["season"]].assign(season=hist["season"].astype(int)), cutoff
        )
        used = sorted(int(s) for s in hist["season"].dropna().unique())
        assert not any(s >= cutoff for s in used), f"cutoff {cutoff} leaked season>= {cutoff}"
        assert 2025 not in used, "2025 Trackman must never be used"
        tm_seasons_used = ",".join(str(s) for s in used)

        for group in CORE_GROUPS:
            g = hist[hist["pitch_type_group"] == group]
            # global (context-agnostic) reference per (cutoff, group, variable)
            global_mean = {name: float(g[col].dropna().mean()) for name, col in PHYSICS.items()}

            for ctx_type, ctx_col in CONTEXT_TYPES.items():
                for name, col in PHYSICS.items():
                    gm = global_mean[name]
                    grouped = g.groupby(ctx_col, dropna=True)[col]
                    cell_ns: list[int] = []
                    cell_vars: list[float] = []
                    means: list[float] = []
                    per_value_rows: list[dict[str, Any]] = []
                    for ctx_value, series in grouped:
                        st = cell_stats(series)
                        dev = st["mean"] - gm if np.isfinite(st["mean"]) else float("nan")
                        row = {
                            "cutoff_valid_year": cutoff,
                            "tm_seasons_used": tm_seasons_used,
                            "context_type": ctx_type,
                            "context_value": str(ctx_value),
                            "pitch_type_group": group,
                            "physical_variable": name,
                            **st,
                            "global_mean": gm,
                            "deviation": dev,
                            "abs_deviation": abs(dev) if np.isfinite(dev) else float("nan"),
                        }
                        per_value_rows.append(row)
                        if st["n"] > 0 and np.isfinite(st["mean"]):
                            cell_ns.append(st["n"])
                            cell_vars.append(st["std"] ** 2)
                            means.append(st["mean"])
                    prof_rows.extend(per_value_rows)

                    if means:
                        mean_min = float(np.min(means))
                        mean_max = float(np.max(means))
                        mrange = mean_max - mean_min
                        pstd = pooled_std(cell_ns, cell_vars)
                        eff = float(mrange / pstd) if (np.isfinite(pstd) and pstd > 0) else float("nan")
                        summ_rows.append(
                            {
                                "cutoff_valid_year": cutoff,
                                "context_type": ctx_type,
                                "pitch_type_group": group,
                                "physical_variable": name,
                                "n_context_values": len(means),
                                "context_mean_min": mean_min,
                                "context_mean_max": mean_max,
                                "mean_range": mrange,
                                "pooled_std": pstd,
                                "effect_ratio": eff,
                                "min_cell_n": int(np.min(cell_ns)),
                                "median_cell_n": float(np.median(cell_ns)),
                                "max_abs_deviation": float(np.max(np.abs(np.array(means) - gm))),
                                "global_mean": gm,
                            }
                        )
    return pd.DataFrame(prof_rows), pd.DataFrame(summ_rows)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_stability(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell across-cutoff means + per-(context_type,group,variable) rank stability."""
    keys = ["context_type", "context_value", "pitch_type_group", "physical_variable"]
    cell_rows: list[dict[str, Any]] = []
    for key_vals, sub in profiles.groupby(keys):
        rec = dict(zip(keys, key_vals))
        devs = {}
        for cutoff in CUTOFFS:
            r = sub[sub["cutoff_valid_year"] == cutoff]
            rec[f"mean_{cutoff}_cutoff"] = float(r["mean"].iloc[0]) if len(r) else float("nan")
            rec[f"dev_{cutoff}_cutoff"] = float(r["deviation"].iloc[0]) if len(r) else float("nan")
            rec[f"n_{cutoff}_cutoff"] = int(r["n"].iloc[0]) if len(r) else 0
            devs[cutoff] = rec[f"dev_{cutoff}_cutoff"]
        cutoff_means = [rec[f"mean_{c}_cutoff"] for c in CUTOFFS]
        finite = [m for m in cutoff_means if np.isfinite(m)]
        rec["max_minus_min"] = float(max(finite) - min(finite)) if len(finite) >= 2 else float("nan")
        dev_signs = [np.sign(devs[c]) for c in CUTOFFS if np.isfinite(devs[c])]
        rec["dev_sign_consistent"] = bool(len(set(dev_signs)) == 1 and len(dev_signs) == len(CUTOFFS))
        cell_rows.append(rec)
    cell_df = pd.DataFrame(cell_rows)

    # rank stability: correlation of context-value mean vectors across cutoff pairs
    rank_rows: list[dict[str, Any]] = []
    gkeys = ["context_type", "pitch_type_group", "physical_variable"]
    for key_vals, sub in cell_df.groupby(gkeys):
        rec = dict(zip(gkeys, key_vals))
        pairs = [(2022, 2023), (2023, 2024), (2022, 2024)]
        for a, b in pairs:
            m = sub[[f"mean_{a}_cutoff", f"mean_{b}_cutoff", f"n_{a}_cutoff", f"n_{b}_cutoff"]].copy()
            m = m[(m[f"n_{a}_cutoff"] >= SPARSE_N) & (m[f"n_{b}_cutoff"] >= SPARSE_N)]
            m = m.dropna(subset=[f"mean_{a}_cutoff", f"mean_{b}_cutoff"])
            rec[f"spearman_{a}_{b}"] = spearman(
                m[f"mean_{a}_cutoff"].to_numpy(), m[f"mean_{b}_cutoff"].to_numpy()
            )
            rec[f"n_common_{a}_{b}"] = int(len(m))
        rank_rows.append(rec)
    return cell_df, pd.DataFrame(rank_rows)


def compute_nonoverlap_profiles(dfc: pd.DataFrame) -> pd.DataFrame:
    """Same context profile computed on each INDEPENDENT single season.

    Expanding cutoffs (<2022, <2023, <2024) share 2019-2021 rows, so a high
    cross-cutoff Spearman is partly mechanical. This recomputes the profile on
    non-overlapping seasons so rank stability is not inflated by shared data.
    """
    rows: list[dict[str, Any]] = []
    for season in NONOVERLAP_SEASONS:
        sub = dfc[dfc["season"] == season]
        for group in CORE_GROUPS:
            g = sub[sub["pitch_type_group"] == group]
            global_mean = {name: float(g[col].dropna().mean()) for name, col in PHYSICS.items()}
            for ctx_type, ctx_col in CONTEXT_TYPES.items():
                for name, col in PHYSICS.items():
                    gm = global_mean[name]
                    for ctx_value, series in g.groupby(ctx_col, dropna=True)[col]:
                        st = cell_stats(series)
                        dev = st["mean"] - gm if np.isfinite(st["mean"]) else float("nan")
                        rows.append(
                            {
                                "season": int(season),
                                "context_type": ctx_type,
                                "context_value": str(ctx_value),
                                "pitch_type_group": group,
                                "physical_variable": name,
                                "n": st["n"],
                                "missing_rate": st["missing_rate"],
                                "mean": st["mean"],
                                "median": st["median"],
                                "std": st["std"],
                                "season_global_mean": gm,
                                "deviation_from_season_group_global": dev,
                            }
                        )
    return pd.DataFrame(rows)


def compute_nonoverlap_stability(nonoverlap: pd.DataFrame) -> pd.DataFrame:
    """Per-(context_type,group,variable): pairwise season Spearman (with and
    without sparse cells), deviation-sign consistency, season effect range."""
    gkeys = ["context_type", "pitch_type_group", "physical_variable"]
    season_pairs = [
        (a, b) for i, a in enumerate(NONOVERLAP_SEASONS) for b in NONOVERLAP_SEASONS[i + 1 :]
    ]
    out_rows: list[dict[str, Any]] = []
    for key_vals, sub in nonoverlap.groupby(gkeys):
        rec = dict(zip(gkeys, key_vals))
        wide_mean = sub.pivot_table(index="context_value", columns="season", values="mean")
        wide_n = sub.pivot_table(index="context_value", columns="season", values="n")
        wide_dev = sub.pivot_table(index="context_value", columns="season", values="deviation_from_season_group_global")

        def pair_spearman(sparse_ok: bool) -> tuple[list[float], list[int]]:
            vals: list[float] = []
            ns: list[int] = []
            for a, b in season_pairs:
                if a not in wide_mean.columns or b not in wide_mean.columns:
                    continue
                m = pd.DataFrame({"a": wide_mean[a], "b": wide_mean[b], "na": wide_n.get(a), "nb": wide_n.get(b)})
                m = m.dropna(subset=["a", "b"])
                if not sparse_ok:
                    m = m[(m["na"].fillna(0) >= SPARSE_N) & (m["nb"].fillna(0) >= SPARSE_N)]
                rho = spearman(m["a"].to_numpy(), m["b"].to_numpy())
                if np.isfinite(rho):
                    vals.append(rho)
                    ns.append(int(len(m)))
            return vals, ns

        all_v, all_n = pair_spearman(sparse_ok=True)
        ge_v, ge_n = pair_spearman(sparse_ok=False)
        rec["median_spearman_all"] = float(np.median(all_v)) if all_v else float("nan")
        rec["min_spearman_all"] = float(np.min(all_v)) if all_v else float("nan")
        rec["n_pairs_all"] = len(all_v)
        rec["median_spearman_ge_sparse"] = float(np.median(ge_v)) if ge_v else float("nan")
        rec["min_spearman_ge_sparse"] = float(np.min(ge_v)) if ge_v else float("nan")
        rec["n_pairs_ge_sparse"] = len(ge_v)
        rec["min_common_cells_ge_sparse"] = int(np.min(ge_n)) if ge_n else 0

        # deviation-sign consistency over non-sparse cells present in all seasons
        dev_ns = wide_dev.copy()
        keep = (wide_n.fillna(0) >= SPARSE_N).all(axis=1) & wide_dev.notna().all(axis=1)
        dev_ns = dev_ns[keep]
        if len(dev_ns):
            signs = np.sign(dev_ns.to_numpy())
            consistent = np.mean([len(set(row)) == 1 for row in signs])
            rec["sign_consistency_rate_ge_sparse"] = float(consistent)
            rec["n_cells_sign_checked"] = int(len(dev_ns))
        else:
            rec["sign_consistency_rate_ge_sparse"] = float("nan")
            rec["n_cells_sign_checked"] = 0

        # per-season effect range (max-min of context means within each season)
        season_ranges = []
        for s in NONOVERLAP_SEASONS:
            if s in wide_mean.columns:
                col = wide_mean[s].dropna()
                if len(col) >= 2:
                    season_ranges.append(float(col.max() - col.min()))
        rec["effect_range_min_season"] = float(np.min(season_ranges)) if season_ranges else float("nan")
        rec["effect_range_max_season"] = float(np.max(season_ranges)) if season_ranges else float("nan")
        rec["min_cell_n_any_season"] = int(np.nanmin(wide_n.to_numpy())) if wide_n.size else 0
        out_rows.append(rec)
    return pd.DataFrame(out_rows).sort_values(
        ["context_type", "pitch_type_group", "physical_variable"]
    ).reset_index(drop=True)


def make_plots(profiles: pd.DataFrame, stability_rank: pd.DataFrame, nonoverlap: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    count_order = [f"{b}-{s}" for b in LEGAL_BALLS for s in LEGAL_STRIKES]

    def count_line(variable: str, fname: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sub = profiles[
            (profiles["context_type"] == "count")
            & (profiles["pitch_type_group"] == "fastball")
            & (profiles["physical_variable"] == variable)
        ]
        for cutoff in CUTOFFS:
            s = sub[sub["cutoff_valid_year"] == cutoff].set_index("context_value")
            y = [s["mean"].get(c, np.nan) for c in count_order]
            ax.plot(count_order, y, marker="o", label=f"< {cutoff}")
        ax.set_xlabel("count (balls-strikes)")
        ax.set_ylabel(f"fastball {variable} mean")
        ax.set_title(title)
        ax.legend(title="TM cutoff")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / fname, dpi=110)
        plt.close(fig)
        saved.append(str((PLOT_DIR / fname).relative_to(ROOT)).replace("\\", "/"))

    count_line("IVB", "count_fastball_IVB_by_cutoff.png", "Count x fastball IVB by TM cutoff")
    count_line("HB", "count_fastball_HB_by_cutoff.png", "Count x fastball HB by TM cutoff")

    # strongest promising candidate (largest effect_ratio among priority physics on count)
    # plotted as breaking IVB by count (chosen dynamically below if different)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = profiles[
        (profiles["context_type"] == "count")
        & (profiles["pitch_type_group"] == "breaking")
        & (profiles["physical_variable"] == "IVB")
    ]
    for cutoff in CUTOFFS:
        s = sub[sub["cutoff_valid_year"] == cutoff].set_index("context_value")
        y = [s["mean"].get(c, np.nan) for c in count_order]
        ax.plot(count_order, y, marker="o", label=f"< {cutoff}")
    ax.set_xlabel("count (balls-strikes)")
    ax.set_ylabel("breaking IVB mean")
    ax.set_title("Count x breaking IVB by TM cutoff")
    ax.legend(title="TM cutoff")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "count_breaking_IVB_by_cutoff.png", dpi=110)
    plt.close(fig)
    saved.append(str((PLOT_DIR / "count_breaking_IVB_by_cutoff.png").relative_to(ROOT)).replace("\\", "/"))

    # temporal stability heatmap: spearman(2022,2024) for count context across group x variable
    fig, ax = plt.subplots(figsize=(7, 4.5))
    piv = (
        stability_rank[stability_rank["context_type"] == "count"]
        .pivot(index="pitch_type_group", columns="physical_variable", values="spearman_2022_2024")
        .reindex(index=list(CORE_GROUPS), columns=list(PHYSICS.keys()))
    )
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy(dtype=float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Count context rank stability: Spearman(<2022, <2024)")
    fig.colorbar(im, ax=ax, label="Spearman")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "count_rank_stability_heatmap.png", dpi=110)
    plt.close(fig)
    saved.append(str((PLOT_DIR / "count_rank_stability_heatmap.png").relative_to(ROOT)).replace("\\", "/"))

    # non-overlapping: breaking IVB by count, one line per independent season
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = nonoverlap[
        (nonoverlap["context_type"] == "count")
        & (nonoverlap["pitch_type_group"] == "breaking")
        & (nonoverlap["physical_variable"] == "IVB")
    ]
    for season in NONOVERLAP_SEASONS:
        s = sub[sub["season"] == season].set_index("context_value")
        y = [s["mean"].get(c, np.nan) for c in count_order]
        ax.plot(count_order, y, marker="o", label=str(season))
    ax.set_xlabel("count (balls-strikes)")
    ax.set_ylabel("breaking IVB mean (rel. units)")
    ax.set_title("Non-overlapping: breaking-group IVB profile by count, per season")
    ax.legend(title="season (independent)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "nonoverlap_breaking_ivb_by_count.png", dpi=110)
    plt.close(fig)
    saved.append(str((PLOT_DIR / "nonoverlap_breaking_ivb_by_count.png").relative_to(ROOT)).replace("\\", "/"))
    return saved


def verdict_flags(summary: pd.DataFrame, stability_rank: pd.DataFrame) -> pd.DataFrame:
    """Conservative per-(context_type,group,variable) verdict."""
    rows: list[dict[str, Any]] = []
    gkeys = ["context_type", "pitch_type_group", "physical_variable"]
    rank_map = stability_rank.set_index(gkeys)
    for key_vals, sub in summary.groupby(gkeys):
        rec = dict(zip(gkeys, key_vals))
        eff = sub["effect_ratio"].to_numpy(dtype=float)
        rec["effect_ratio_mean"] = float(np.nanmean(eff))
        rec["effect_ratio_min"] = float(np.nanmin(eff))
        rec["max_abs_deviation"] = float(sub["max_abs_deviation"].max())
        rec["min_cell_n"] = int(sub["min_cell_n"].min())
        rec["median_cell_n"] = float(sub["median_cell_n"].median())
        sp = rank_map.loc[key_vals] if key_vals in rank_map.index else None
        s22_24 = float(sp["spearman_2022_2024"]) if sp is not None else float("nan")
        s22_23 = float(sp["spearman_2022_2023"]) if sp is not None else float("nan")
        s23_24 = float(sp["spearman_2023_2024"]) if sp is not None else float("nan")
        rec["spearman_2022_2023"] = s22_23
        rec["spearman_2023_2024"] = s23_24
        rec["spearman_2022_2024"] = s22_24
        finite_sp = [v for v in (s22_23, s23_24, s22_24) if np.isfinite(v)]
        sp_min = float(np.min(finite_sp)) if finite_sp else float("nan")

        sparse = rec["min_cell_n"] < SPARSE_N
        strong_effect = rec["effect_ratio_min"] >= 0.10
        some_effect = rec["effect_ratio_mean"] >= 0.05
        stable = np.isfinite(sp_min) and sp_min >= 0.8
        unstable = np.isfinite(sp_min) and sp_min < 0.3

        if sparse:
            v = "SPARSE"
        elif strong_effect and stable:
            v = "PROMISING"
        elif some_effect and unstable:
            v = "UNSTABLE"
        elif some_effect and stable:
            v = "WEAK"
        elif not some_effect:
            v = "NOT_USEFUL"
        else:
            v = "WEAK"
        rec["verdict"] = v
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(
        ["effect_ratio_mean"], ascending=False
    ).reset_index(drop=True)


def audit_nan_inf(profiles: pd.DataFrame) -> dict[str, Any]:
    num = profiles.select_dtypes(include=[np.number])
    return {
        "profile_rows": int(len(profiles)),
        "duplicate_key_rows": int(
            profiles.duplicated(
                ["cutoff_valid_year", "context_type", "context_value", "pitch_type_group", "physical_variable"]
            ).sum()
        ),
        "inf_cells": int(np.isinf(num.to_numpy(dtype=float)).sum()),
        "mean_nan_rows": int(profiles["mean"].isna().sum()),
    }


def main() -> None:
    if not TRACKMAN_PATH.exists():
        raise FileNotFoundError(TRACKMAN_PATH)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_trackman()
    dfc = build_context_frame(df)

    profiles, summary = compute_profiles(dfc)
    stability_cell, stability_rank = compute_stability(profiles)
    verdicts = verdict_flags(summary, stability_rank)

    # non-overlapping single-season stability (independent seasons, not expanding)
    nonoverlap = compute_nonoverlap_profiles(dfc)
    nonoverlap_stability = compute_nonoverlap_stability(nonoverlap)

    integrity = audit_nan_inf(profiles)

    profiles.to_csv(ANALYSIS_DIR / "trackman_context_profiles_002.csv", index=False)
    stability_cell.to_csv(ANALYSIS_DIR / "trackman_context_stability_002.csv", index=False)
    # summary CSV = per-(context_type,group,variable) diagnostics incl. verdict + rank stability
    summary_out = summary.merge(
        stability_rank, on=["context_type", "pitch_type_group", "physical_variable"], how="left"
    )
    summary_out.to_csv(ANALYSIS_DIR / "trackman_context_summary_002.csv", index=False)
    verdicts.to_csv(ANALYSIS_DIR / "trackman_context_verdicts_002.csv", index=False)
    nonoverlap.to_csv(ANALYSIS_DIR / "trackman_context_nonoverlap_002.csv", index=False)
    nonoverlap_stability.to_csv(
        ANALYSIS_DIR / "trackman_context_nonoverlap_stability_002.csv", index=False
    )

    plots = make_plots(profiles, stability_rank, nonoverlap)

    # rows-by-cutoff for report
    rows_by_cutoff = {}
    for cutoff in CUTOFFS:
        allowed = list(allowed_trackman_seasons(cutoff))
        sub = dfc[dfc["season"].isin(allowed)]
        rows_by_cutoff[cutoff] = {
            "tm_seasons": allowed,
            "n_tm_rows": int(len(sub)),
            "n_core_rows": int(sub["pitch_type_group"].isin(CORE_GROUPS).sum()),
        }

    n_dist = profiles["n"].describe(percentiles=[0.1, 0.5, 0.9]).to_dict()
    sparse_share = float((profiles["n"] < SPARSE_N).mean())

    out = {
        "cutoffs": list(CUTOFFS),
        "rows_by_cutoff": rows_by_cutoff,
        "integrity": integrity,
        "n_distribution": {k: float(v) for k, v in n_dist.items()},
        "sparse_share": sparse_share,
        "sparse_n_threshold": SPARSE_N,
        "plots": plots,
        "top_candidates": verdicts.head(12).to_dict(orient="records"),
        "nonoverlap_seasons": list(NONOVERLAP_SEASONS),
        "nonoverlap_integrity": {
            "rows": int(len(nonoverlap)),
            "duplicate_keys": int(
                nonoverlap.duplicated(
                    ["season", "context_type", "context_value", "pitch_type_group", "physical_variable"]
                ).sum()
            ),
            "inf_cells": int(
                np.isinf(nonoverlap.select_dtypes(include=[np.number]).to_numpy(dtype=float)).sum()
            ),
        },
        "nonoverlap_top_candidate_stability": nonoverlap_stability[
            nonoverlap_stability.apply(
                lambda r: (r["context_type"], r["pitch_type_group"], r["physical_variable"]) in TOP_CANDIDATES,
                axis=1,
            )
        ].to_dict(orient="records"),
    }
    (PLOT_DIR).mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "trackman_audit_002_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
