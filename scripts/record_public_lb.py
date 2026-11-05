"""Record confirmed Public LB scores onto existing experiment log rows.

Does not append new rows. Does not rewrite unspecified historical fields.
Not a submission-management system.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils import resolve_path, update_experiment_log_fields

DEFAULT_LOG = "outputs/experiments.csv"

CONFIRMED_PUBLIC_LB = (
    {
        "experiment_id": "EXP-004",
        "public_lb_score": "406.9505006093",
        "submission_artifact": "outputs/submissions/EXP-004_submit.zip",
        "score_source": "user_confirmed_dacon_public_lb",
        "score_status": "recorded",
        "dacon_submit_timestamp": "unknown",
    },
    {
        "experiment_id": "EXP-005",
        "public_lb_score": "877.6038710153",
        "submission_artifact": "outputs/submissions/EXP-005_submit.zip",
        "score_source": "user_confirmed_dacon_public_lb",
        "score_status": "recorded",
        "dacon_submit_timestamp": "unknown",
    },
)


def file_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    text = data.decode("utf-8")
    return {
        "path": str(path),
        "exists": True,
        "nbytes": len(data),
        "n_lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def artifact_mtime_iso(path: Path) -> str:
    if not path.exists():
        return "missing"
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def _gap(public_lb: str, other: str) -> str:
    try:
        return str(float(public_lb) - float(other))
    except (TypeError, ValueError):
        return ""


def _last_row(path: Path, experiment_id: str) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        matches = [row for row in csv.DictReader(f) if row.get("experiment_id") == experiment_id]
    if not matches:
        raise KeyError(f"No experiment_id={experiment_id!r} row in {path}")
    return dict(matches[-1])


def record_confirmed_scores(log_path: str | Path = DEFAULT_LOG) -> dict[str, Any]:
    path = resolve_path(log_path)
    before = file_stats(path)
    recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updates = []
    for item in CONFIRMED_PUBLIC_LB:
        artifact = resolve_path(item["submission_artifact"])
        previous = _last_row(path, item["experiment_id"])
        public_lb = item["public_lb_score"]
        cv_lb_gap = _gap(public_lb, str(previous.get("cv_score", "")))
        merge_notes = {
            "public_lb_recorded_at": recorded_at,
            "public_lb_score_source": item["score_source"],
            "public_lb_status": item["score_status"],
            "public_lb_submission_artifact": item["submission_artifact"],
            "public_lb_artifact_mtime_utc": artifact_mtime_iso(artifact),
            "public_lb_dacon_submit_timestamp": item["dacon_submit_timestamp"],
            "cv_lb_gap_definition": "public_lb_score - cv_score",
        }
        fields = {"public_lb_score": public_lb}
        if cv_lb_gap:
            fields["cv_lb_gap"] = cv_lb_gap
        result = update_experiment_log_fields(
            path,
            item["experiment_id"],
            fields=fields,
            merge_notes=merge_notes,
        )
        updates.append(
            {
                "experiment_id": item["experiment_id"],
                "previous_public_lb_score": previous.get("public_lb_score", ""),
                "new_public_lb_score": public_lb,
                "previous_cv_lb_gap": previous.get("cv_lb_gap", ""),
                "new_cv_lb_gap": fields.get("cv_lb_gap", previous.get("cv_lb_gap", "")),
                "changed_fields": result["changed_fields"],
                "submission_artifact": item["submission_artifact"],
                "artifact_mtime_utc": artifact_mtime_iso(artifact),
                "score_source": item["score_source"],
                "score_status": item["score_status"],
            }
        )
    after = file_stats(path)
    return {"before": before, "after": after, "updates": updates, "recorded_at": recorded_at}


def main() -> None:
    parser = argparse.ArgumentParser(description="Record confirmed Public LB scores.")
    parser.add_argument("--log", default=DEFAULT_LOG)
    args = parser.parse_args()
    payload = record_confirmed_scores(args.log)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
