from __future__ import annotations

import unittest

import numpy as np

from src.metrics import OFFICIAL_BSS_SCALE, brier_score, brier_skill_score, reference_brier_score


class BrierMetricTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        y = np.array([0, 1, 0, 1], dtype=float)
        p = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        self.assertAlmostEqual(brier_score(y, p), 0.0)
        self.assertAlmostEqual(brier_skill_score(y, p), float(OFFICIAL_BSS_SCALE))

    def test_all_half_predictions(self) -> None:
        y = np.array([0, 1, 0, 1], dtype=float)
        p = np.full(4, 0.5)
        # BS = mean((0.5-0)^2 + (0.5-1)^2 + ...) = 0.25
        self.assertAlmostEqual(brier_score(y, p), 0.25)
        r = 0.5
        expected = OFFICIAL_BSS_SCALE * (1.0 - 0.25 / (r * (1.0 - r)))
        self.assertAlmostEqual(brier_skill_score(y, p), expected)
        self.assertAlmostEqual(brier_skill_score(y, p), 0.0)

    def test_constant_train_mean_prediction(self) -> None:
        # Simulated train mean used as constant prediction on a validation fold.
        y = np.array([0, 0, 1, 1, 1], dtype=float)
        train_mean = 0.4
        p = np.full(len(y), train_mean)
        bs = brier_score(y, p)
        r = float(y.mean())  # 0.6
        hand = max(0.0, OFFICIAL_BSS_SCALE * (1.0 - bs / (r * (1.0 - r))))
        self.assertAlmostEqual(brier_skill_score(y, p), hand)
        # Constant prediction equal to fold r should score 0.
        p_climatology = np.full(len(y), r)
        self.assertAlmostEqual(brier_score(y, p_climatology), reference_brier_score(y))
        self.assertAlmostEqual(brier_skill_score(y, p_climatology), 0.0)

    def test_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            brier_score([0, 1], [0.1, 0.2, 0.3])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            brier_skill_score([0, 1], [0.1])

    def test_probability_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            brier_score([0, 1], [0.1, 1.1])
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            brier_skill_score([0, 1], [-0.01, 0.5])

    def test_non_binary_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary"):
            brier_score([0, 2], [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "binary"):
            brier_skill_score([0.5, 1], [0.1, 0.2])

    def test_single_class_target(self) -> None:
        y = np.ones(5)
        p = np.full(5, 0.9)
        self.assertAlmostEqual(brier_score(y, p), float(np.mean((0.9 - 1.0) ** 2)))
        self.assertAlmostEqual(reference_brier_score(y), 0.0)
        with self.assertRaisesRegex(ValueError, r"r\*\(1-r\)==0"):
            brier_skill_score(y, p)
        y0 = np.zeros(4)
        with self.assertRaisesRegex(ValueError, r"r\*\(1-r\)==0"):
            brier_skill_score(y0, np.full(4, 0.1))

    def test_official_formula_hand_calculation(self) -> None:
        # Hand calculation against the official formula.
        y = np.array([1, 0, 1, 0, 1], dtype=float)
        p = np.array([0.8, 0.2, 0.7, 0.4, 0.6], dtype=float)
        bs_hand = float(np.mean((p - y) ** 2))
        r = float(y.mean())  # 0.6
        denom = r * (1.0 - r)  # 0.24
        score_hand = max(0.0, 100000.0 * (1.0 - bs_hand / denom))
        self.assertAlmostEqual(brier_score(y, p), bs_hand)
        self.assertAlmostEqual(reference_brier_score(y), denom)
        self.assertAlmostEqual(brier_skill_score(y, p), score_hand)
        # Worse-than-climatology prediction must clip to 0 under official max(0, ...).
        bad = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
        raw = 100000.0 * (1.0 - brier_score(y, bad) / denom)
        self.assertLess(raw, 0.0)
        self.assertAlmostEqual(brier_skill_score(y, bad), 0.0)

    def test_evaluate_predictions_bundle(self) -> None:
        from src.evaluate import evaluate_predictions

        y = np.array([1, 0, 1, 0], dtype=float)
        p = np.array([0.9, 0.1, 0.8, 0.2], dtype=float)
        bundle = evaluate_predictions(y, p)
        self.assertEqual(
            set(bundle),
            {
                "n_samples",
                "target_rate",
                "brier_score",
                "raw_brier_skill_score",
                "official_brier_skill_score",
            },
        )
        self.assertEqual(bundle["n_samples"], 4)
        self.assertAlmostEqual(float(bundle["target_rate"]), 0.5)
        self.assertGreaterEqual(float(bundle["official_brier_skill_score"]), 0.0)
        self.assertAlmostEqual(
            float(bundle["official_brier_skill_score"]),
            max(0.0, float(bundle["raw_brier_skill_score"])),
        )


if __name__ == "__main__":
    unittest.main()
