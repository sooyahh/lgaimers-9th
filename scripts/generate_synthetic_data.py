from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "synthetic"


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    train_rows = []
    row_id = 0
    for season in range(2019, 2025):
        for pitch_no in range(24):
            x = rng.normal() + 0.08 * (season - 2019)
            pitcher = f"P{pitch_no % 5}" if pitch_no >= 3 else f"NEW_{season}"
            probability = 1 / (1 + np.exp(-x))
            train_rows.append(
                {
                    "row_id": row_id,
                    "season": season,
                    "event_time": f"{season}-{1 + pitch_no // 12:02d}-{1 + pitch_no % 12:02d}",
                    "pitcher_id": pitcher,
                    "numeric_feature": x,
                    "category_feature": "A" if pitch_no % 2 else "B",
                    "target": int(rng.random() < probability),
                }
            )
            row_id += 1
    train = pd.DataFrame(train_rows)
    test = pd.DataFrame(
        {
            "row_id": np.arange(9000, 9020),
            "season": 2025,
            "event_time": pd.date_range("2025-03-01", periods=20, freq="D").astype(str),
            "pitcher_id": [f"P{i % 5}" for i in range(20)],
            "numeric_feature": rng.normal(size=20),
            "category_feature": ["A", "B"] * 10,
        }
    )
    sample = pd.DataFrame({"row_id": test["row_id"], "success_probability": 0.5})
    return train, test, sample


def main() -> None:
    train, test, sample = build_frames()
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    sample.to_csv(OUT / "sample_submission.csv", index=False)
    print(f"synthetic_data={OUT}")


if __name__ == "__main__":
    main()
