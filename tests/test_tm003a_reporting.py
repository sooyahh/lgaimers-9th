"""TM-003A reporting/provenance helpers. Does not change feature math."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_tm003a_expected_ivb import (
    RUN_KIND_CANONICAL,
    RUN_KIND_NON_CANONICAL,
    aggregate_overall_fallback,
    assert_canonical_run_allowed,
    assert_frozen_tm003a_metadata,
    collect_cell_n_diagnostics,
    count_nonfinite_large_cells,
    mark_prior_tm003a_rows_superseded,
    parse_notes_json,
    resolve_run_kind,
)
from src.trackman_profiles import MIN_CELL_N
from src.utils import load_config, to_json_text


class Tm003AReportingTests(unittest.TestCase):
    def test_resolve_run_kind_ignores_git_dirty(self) -> None:
        kind, canonical = resolve_run_kind(canonical_requested=False)
        self.assertEqual(kind, RUN_KIND_NON_CANONICAL)
        self.assertFalse(canonical)
        kind, canonical = resolve_run_kind(canonical_requested=True)
        self.assertEqual(kind, RUN_KIND_CANONICAL)
        self.assertTrue(canonical)

    def test_git_dirty_false_is_not_sufficient_for_canonical(self) -> None:
        kind, canonical = resolve_run_kind(canonical_requested=False)
        self.assertFalse(canonical)
        self.assertEqual(kind, RUN_KIND_NON_CANONICAL)

    def test_overall_fallback_pools_constructible_and_valid_denominators(self) -> None:
        rows = [
            {
                "fold_id": "secondary_2022",
                "n_valid": 247472,
                "n_constructible": 247384,
                "fallback_any_row_rate": 0.011354816803026873,
                "fallback_weight_share": 0.0049023300014713975,
                "tm_feature_nan_rate": 0.00035559578457360834,
            },
            {
                "fold_id": "secondary_2023",
                "n_valid": 245525,
                "n_constructible": 245462,
                "fallback_any_row_rate": 0.011643350090849092,
                "fallback_weight_share": 0.005149180084021152,
                "tm_feature_nan_rate": 0.0002565930149679259,
            },
            {
                "fold_id": "primary_2024",
                "n_valid": 253507,
                "n_constructible": 253426,
                "fallback_any_row_rate": 0.012682203088870124,
                "fallback_weight_share": 0.0037703692529574705,
                "tm_feature_nan_rate": 0.00031951780424209193,
            },
        ]
        overall = aggregate_overall_fallback(rows)
        self.assertEqual(overall["n_valid"], 746504)
        self.assertEqual(overall["n_constructible"], 746272)
        self.assertAlmostEqual(overall["fallback_any_row_rate"], 0.01190049, places=8)
        self.assertAlmostEqual(overall["fallback_weight_share"], 0.00459912, places=8)
        self.assertAlmostEqual(overall["tm_feature_nan_rate"], 0.00031078, places=8)

    def test_cell_n_diagnostics_reuse_fold_extra_profile(self) -> None:
        folds = [
            {
                "fold_id": "secondary_2022",
                "fold_extra": {
                    "valid_season": 2022,
                    "cell_n_valid_profile": {"min": 227.0, "p10": 2552.0, "median": 16950.0, "n_cells": 36},
                },
            }
        ]
        out = collect_cell_n_diagnostics(folds)
        self.assertEqual(out["by_fold"]["secondary_2022"]["min"], 227.0)
        self.assertEqual(out["by_fold"]["secondary_2022"]["p10"], 2552.0)
        self.assertEqual(out["by_fold"]["secondary_2022"]["median"], 16950.0)
        self.assertIn("full cell row count", out["definition"])

    def test_nonfinite_large_cell_counter_does_not_change_values(self) -> None:
        cache = {
            2022: {
                "cell_n": {("0-0", "fastball"): 1000, ("0-1", "breaking"): 500},
                "cell_mean": {("0-0", "fastball"): 12.0, ("0-1", "breaking"): float("nan")},
            },
            2023: {
                "cell_n": {("3-0", "offspeed"): 1500},
                "cell_mean": {("3-0", "offspeed"): float("inf")},
            },
        }
        out = count_nonfinite_large_cells(cache)
        self.assertEqual(out["n_cells"], 1)
        self.assertEqual(out["min_n"], MIN_CELL_N)
        self.assertEqual(out["details"][0]["count"], "3-0")

    def test_frozen_metadata_matches_current_config(self) -> None:
        cfg = load_config("configs/tm003a.yaml")
        assert_frozen_tm003a_metadata(cfg)
        self.assertFalse(cfg["tm003a"]["min_n_is_tunable"])
        self.assertTrue(cfg["tm003a"]["canonical_requires_explicit_flag"])
        self.assertEqual(cfg["tm003a"]["min_n"], 1000)

    def test_frozen_metadata_rejects_min_n_drift(self) -> None:
        cfg = load_config("configs/tm003a.yaml")
        cfg["tm003a"]["min_n"] = 999
        with self.assertRaises(AssertionError):
            assert_frozen_tm003a_metadata(cfg)

    def test_canonical_refused_when_dirty_or_missing_commit(self) -> None:
        with self.assertRaises(AssertionError):
            assert_canonical_run_allowed({"git_dirty": "true", "git_commit": "abc"})
        with self.assertRaises(AssertionError):
            assert_canonical_run_allowed({"git_dirty": "false", "git_commit": ""})

    def test_prior_rows_not_superseded_unless_last_row_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "experiments.csv"
            fieldnames = ["experiment_id", "date", "notes", "status"]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "experiment_id": "TM-003A",
                        "date": "2026-08-18 00:45:47",
                        "notes": to_json_text({"run_kind": "pre_canonical", "canonical": False}),
                        "status": "success",
                    }
                )
                writer.writerow(
                    {
                        "experiment_id": "TM-003A",
                        "date": "2026-08-19 01:00:00",
                        "notes": to_json_text({"run_kind": "non_canonical", "canonical": False}),
                        "status": "success",
                    }
                )
            with self.assertRaises(AssertionError):
                mark_prior_tm003a_rows_superseded(path)
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            first_notes = parse_notes_json(rows[0]["notes"])
            self.assertFalse(first_notes.get("superseded", False))
            self.assertEqual(rows[0]["status"], "success")

    def test_prior_rows_superseded_only_after_canonical_row_exists(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "experiments.csv"
            fieldnames = ["experiment_id", "date", "notes", "status"]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "experiment_id": "TM-003A",
                        "date": "2026-08-18 00:45:47",
                        "notes": to_json_text({"run_kind": "pre_canonical", "canonical": False}),
                        "status": "success",
                    }
                )
                writer.writerow(
                    {
                        "experiment_id": "TM-003A",
                        "date": "2026-08-20 12:00:00",
                        "notes": to_json_text({"run_kind": "canonical", "canonical": True}),
                        "status": "success",
                    }
                )
            n = mark_prior_tm003a_rows_superseded(path)
            self.assertEqual(n, 1)
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            old_notes = parse_notes_json(rows[0]["notes"])
            new_notes = parse_notes_json(rows[1]["notes"])
            self.assertTrue(old_notes["superseded"])
            self.assertEqual(old_notes["run_kind"], "pre_canonical")
            self.assertEqual(rows[0]["status"], "success")
            self.assertTrue(new_notes["canonical"])
            self.assertNotIn("superseded", new_notes)

    def test_failed_canonical_does_not_call_supersede_via_noncanonical_last_row(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "experiments.csv"
            fieldnames = ["experiment_id", "date", "notes", "status"]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "experiment_id": "TM-003A",
                        "date": "2026-08-18 00:45:47",
                        "notes": to_json_text({"run_kind": "pre_canonical", "canonical": False}),
                        "status": "success",
                    }
                )
            n = mark_prior_tm003a_rows_superseded(path)
            self.assertEqual(n, 0)
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            notes = parse_notes_json(rows[0]["notes"])
            self.assertFalse(notes.get("superseded", False))


if __name__ == "__main__":
    unittest.main()
