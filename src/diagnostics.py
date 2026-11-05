from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import calibration_table, safe_group_scores
from src.utils import ensure_dir


def save_diagnostics(
    predictions: pd.DataFrame,
    output_dir: Path,
    target_col: str,
    metric: str,
    season_col: str,
    entity_col: str | None,
    n_bins: int,
    min_group_rows: int,
    metric_callable: str | None = None,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    paths: dict[str, Path] = {}
    distribution = predictions.groupby(["model", "fold"], sort=False)["prediction"].agg(
        ["count", "min", "max", "mean", "std"]
    ).reset_index()
    paths["probability_distribution"] = output_dir / "probability_distribution.csv"
    distribution.to_csv(paths["probability_distribution"], index=False)

    calibration_parts = []
    for (model, fold), part in predictions.groupby(["model", "fold"], sort=False):
        table = calibration_table(part[target_col], part["prediction"], n_bins=n_bins)
        table.insert(0, "fold", fold)
        table.insert(0, "model", model)
        calibration_parts.append(table)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    paths["calibration"] = output_dir / "calibration_bins.csv"
    calibration.to_csv(paths["calibration"], index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, part in predictions.groupby("model", sort=False):
        ax.hist(part["prediction"], bins=20, alpha=0.45, label=model)
    ax.set(xlabel="Predicted probability", ylabel="Rows", title="OOF probability distribution")
    ax.legend()
    fig.tight_layout()
    paths["probability_plot"] = output_dir / "probability_histogram.png"
    fig.savefig(paths["probability_plot"], dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for model, model_predictions in predictions.groupby("model", sort=False):
        aggregated = calibration_table(
            model_predictions[target_col], model_predictions["prediction"], n_bins=n_bins
        ).dropna()
        ax.plot(aggregated["mean_prediction"], aggregated["observed_rate"], marker="o", label=model)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean prediction", ylabel="Observed rate", title="OOF calibration")
    ax.legend()
    fig.tight_layout()
    paths["calibration_plot"] = output_dir / "calibration_curve.png"
    fig.savefig(paths["calibration_plot"], dpi=140)
    plt.close(fig)

    seasons = safe_group_scores(
        predictions, ["model", season_col], target_col, "prediction", metric,
        min_rows=min_group_rows, metric_callable=metric_callable,
    )
    paths["season_metrics"] = output_dir / "season_metrics.csv"
    seasons.to_csv(paths["season_metrics"], index=False)

    if entity_col:
        entities = safe_group_scores(
            predictions, ["model", entity_col], target_col, "prediction", metric,
            min_rows=min_group_rows, metric_callable=metric_callable,
        )
        paths["entity_metrics"] = output_dir / "entity_metrics.csv"
        entities.to_csv(paths["entity_metrics"], index=False)
        seen = safe_group_scores(
            predictions, ["model", "entity_status"], target_col, "prediction", metric,
            min_rows=min_group_rows, metric_callable=metric_callable,
        )
        paths["entity_status_metrics"] = output_dir / "entity_status_metrics.csv"
        seen.to_csv(paths["entity_status_metrics"], index=False)
    return paths
