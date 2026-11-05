from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils import (
    EXPERIMENT_COLUMNS,
    append_experiment_log,
    collect_run_provenance,
    update_experiment_log_fields,
)


OLD_HEADER = [
    "experiment_id",
    "date",
    "purpose",
    "data_version",
    "features",
    "model",
    "params",
    "cv_strategy",
    "metric",
    "metric_direction",
    "cv_score",
    "cv_std",
    "holdout_score",
    "public_lb_score",
    "cv_lb_gap",
    "seed",
    "notes",
    "next_action",
    "status",
    "error",
]


class ExperimentLogProvenanceTests(unittest.TestCase):
    def test_new_log_includes_provenance_columns(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "experiments.csv"
            append_experiment_log(
                path,
                {
                    "experiment_id": "EXP-TEST",
                    "purpose": "provenance_unit_test",
                    "status": "success",
                    "seed": 42,
                },
            )
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(list(reader.fieldnames), list(EXPERIMENT_COLUMNS))
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_id"], "EXP-TEST")
            self.assertEqual(rows[0]["seed"], "42")
            self.assertIn("author", rows[0])
            self.assertIn("git_branch", rows[0])
            self.assertIn("git_commit", rows[0])
            self.assertIn("git_dirty", rows[0])

    def test_old_schema_is_upgraded_without_inventing_historical_values(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "experiments.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=OLD_HEADER)
                writer.writeheader()
                writer.writerow(
                    {
                        "experiment_id": "EXP-000",
                        "purpose": "historical",
                        "status": "success",
                        "seed": "42",
                        "error": "",
                    }
                )
            append_experiment_log(
                path,
                {
                    "experiment_id": "EXP-007",
                    "purpose": "new_row",
                    "status": "success",
                    "author": "tester",
                },
            )
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(list(reader.fieldnames), list(EXPERIMENT_COLUMNS))
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            historical = rows[0]
            self.assertEqual(historical["experiment_id"], "EXP-000")
            self.assertEqual(historical["purpose"], "historical")
            self.assertEqual(historical["author"], "")
            self.assertEqual(historical["git_branch"], "")
            self.assertEqual(historical["git_commit"], "")
            self.assertEqual(historical["git_dirty"], "")
            new_row = rows[1]
            self.assertEqual(new_row["experiment_id"], "EXP-007")
            self.assertEqual(new_row["author"], "tester")

    def test_explicit_author_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "experiments.csv"
            append_experiment_log(
                path,
                {"experiment_id": "EXP-TEST", "author": "explicit-author", "status": "success"},
            )
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["author"], "explicit-author")

    def test_git_failure_does_not_raise(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "GIT_AUTHOR_NAME"}
        with patch.dict(os.environ, env, clear=True):
            with patch("src.utils.subprocess.run", side_effect=OSError("git missing")):
                provenance = collect_run_provenance()
                self.assertEqual(provenance["author"], "")
                self.assertEqual(provenance["git_branch"], "")
                self.assertEqual(provenance["git_commit"], "")
                self.assertEqual(provenance["git_dirty"], "")
                with tempfile.TemporaryDirectory() as raw_dir:
                    path = Path(raw_dir) / "experiments.csv"
                    append_experiment_log(
                        path,
                        {"experiment_id": "EXP-TEST", "status": "success"},
                    )
                    with path.open("r", newline="", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual(rows[0]["author"], "")
                    self.assertEqual(rows[0]["git_branch"], "")
                    self.assertEqual(rows[0]["git_commit"], "")
                    self.assertEqual(rows[0]["git_dirty"], "")
                    self.assertEqual(rows[0]["experiment_id"], "EXP-TEST")

    def test_explicit_git_dirty_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "experiments.csv"
            append_experiment_log(
                path,
                {"experiment_id": "EXP-TEST", "git_dirty": "false", "status": "success"},
            )
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["git_dirty"], "false")

    def test_git_dirty_false_when_porcelain_empty(self) -> None:
        def fake_run(args, **kwargs):
            result = Mock()
            result.returncode = 0
            result.stdout = "" if "status" in args else "value\n"
            result.stderr = ""
            return result

        with patch("src.utils.subprocess.run", side_effect=fake_run):
            provenance = collect_run_provenance()
        self.assertEqual(provenance["git_dirty"], "false")

    def test_git_dirty_true_when_tracked_or_untracked_changes(self) -> None:
        def fake_run(args, **kwargs):
            result = Mock()
            result.returncode = 0
            result.stdout = " M src/utils.py\n?? scratch.txt\n" if "status" in args else "value\n"
            result.stderr = ""
            return result

        with patch("src.utils.subprocess.run", side_effect=fake_run):
            provenance = collect_run_provenance()
        self.assertEqual(provenance["git_dirty"], "true")


class ExperimentLogUpdateTests(unittest.TestCase):
    def _write_two_rows(self, path: Path) -> None:
        append_experiment_log(
            path,
            {
                "experiment_id": "EXP-004",
                "purpose": "keep_me",
                "public_lb_score": "",
                "cv_score": "490.3",
                "notes": '{"decision": "results_only"}',
                "status": "success",
                "author": "historical",
            },
        )
        append_experiment_log(
            path,
            {
                "experiment_id": "EXP-005",
                "purpose": "also_keep",
                "public_lb_score": "",
                "status": "success",
                "author": "historical-005",
            },
        )

    def test_patches_only_requested_fields_on_matching_row(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "experiments.csv"
            self._write_two_rows(path)
            result = update_experiment_log_fields(
                path,
                "EXP-004",
                {"public_lb_score": "406.9505006093"},
                merge_notes={"public_lb_status": "recorded"},
            )
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["experiment_id"], "EXP-004")
            self.assertEqual(rows[0]["purpose"], "keep_me")
            self.assertEqual(rows[0]["cv_score"], "490.3")
            self.assertEqual(rows[0]["public_lb_score"], "406.9505006093")
            self.assertEqual(rows[0]["author"], "historical")
            notes = json.loads(rows[0]["notes"])
            self.assertEqual(notes["decision"], "results_only")
            self.assertEqual(notes["public_lb_status"], "recorded")
            self.assertEqual(rows[1]["experiment_id"], "EXP-005")
            self.assertEqual(rows[1]["public_lb_score"], "")
            self.assertEqual(rows[1]["author"], "historical-005")
            self.assertEqual(result["changed_fields"], ["notes", "public_lb_score"])

    def test_refuses_missing_id_and_experiment_id_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "experiments.csv"
            self._write_two_rows(path)
            with self.assertRaises(KeyError):
                update_experiment_log_fields(path, "EXP-MISSING", {"public_lb_score": "1"})
            with self.assertRaisesRegex(ValueError, "experiment_id"):
                update_experiment_log_fields(path, "EXP-004", {"experiment_id": "EXP-999"})


if __name__ == "__main__":
    unittest.main()
