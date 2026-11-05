from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


EXPERIMENT_COLUMNS = [
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
    "author",
    "git_branch",
    "git_commit",
    "git_dirty",
]

PROVENANCE_COLUMNS = ("author", "git_branch", "git_commit", "git_dirty")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    config["_config_path"] = str(path)
    return config


def resolve_path(path_value: str | Path, root: Path | None = None) -> Path:
    root = root or project_root()
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    return path


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs(config: dict[str, Any]) -> None:
    output = config.get("output", {})
    for key in ["models_dir", "predictions_dir", "submissions_dir"]:
        if key in output:
            ensure_dir(resolve_path(output[key]))
    if "experiments_path" in output:
        ensure_dir(resolve_path(output["experiments_path"]).parent)


def make_experiment_id(prefix: str = "exp") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}"


def assert_not_placeholder(value: Any, name: str) -> None:
    if value is None:
        raise ValueError(f"Missing required config value: {name}")
    if isinstance(value, str) and (not value.strip() or value.startswith("TODO_")):
        raise ValueError(
            f"Config value '{name}' is still a placeholder: {value}. "
            "Update configs/baseline.yaml after data release."
        )


def _git_text(*args: str) -> str:
    """Return stripped git stdout, or empty string if git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def _git_dirty_flag() -> str:
    """Return 'true'/'false' from git status, or empty if git is unavailable.

    Empty porcelain is clean. Tracked or untracked changes are dirty.
    Ignored files are not considered. Never raises.
    """
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return "true" if (completed.stdout or "").strip() else "false"


def collect_run_provenance() -> dict[str, str]:
    """Best-effort author / branch / commit / dirty flag. Never raises."""
    author = os.environ.get("GIT_AUTHOR_NAME", "").strip() or _git_text("config", "user.name")
    return {
        "author": author,
        "git_branch": _git_text("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git_text("rev-parse", "HEAD"),
        "git_dirty": _git_dirty_flag(),
    }


def _apply_run_provenance(row: dict[str, Any]) -> dict[str, Any]:
    filled = dict(row)
    provenance = collect_run_provenance()
    for col in PROVENANCE_COLUMNS:
        current = filled.get(col, "")
        if current is None or str(current).strip() == "":
            filled[col] = provenance.get(col, "")
    return filled


def ensure_experiment_log_schema(path: str | Path) -> None:
    """Add missing provenance columns to an existing log without inventing values."""
    path = resolve_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [col for col in EXPERIMENT_COLUMNS if col not in existing_fields]
    if not missing:
        return
    extra = [col for col in existing_fields if col not in EXPERIMENT_COLUMNS]
    fieldnames = list(EXPERIMENT_COLUMNS) + extra
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            padded = {col: row.get(col, "") if row.get(col, "") is not None else "" for col in fieldnames}
            writer.writerow(padded)


def append_experiment_log(path: str | Path, row: dict[str, Any]) -> None:
    path = resolve_path(path)
    ensure_dir(path.parent)
    ensure_experiment_log_schema(path)
    normalized = {col: row.get(col, "") for col in EXPERIMENT_COLUMNS}
    provenance_row = _apply_run_provenance(normalized)
    normalized = {col: provenance_row.get(col, "") for col in EXPERIMENT_COLUMNS}
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalized)


def update_experiment_log_fields(
    path: str | Path,
    experiment_id: str,
    fields: dict[str, Any],
    *,
    which: str = "last",
    merge_notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Patch specified fields on an existing experiment_id row. Does not append.

    Other rows and unspecified columns are left as read. Historical provenance
    columns are not auto-filled. ``experiment_id`` cannot be changed.

    ``which`` selects ``last`` (default) or ``first`` matching row.
    ``merge_notes`` is merged into the selected row's notes JSON when parseable.
    """
    path = resolve_path(path)
    if which not in {"first", "last"}:
        raise ValueError("which must be 'first' or 'last'")
    if "experiment_id" in fields:
        raise ValueError("Refusing to change experiment_id.")
    unknown = [key for key in fields if key not in EXPERIMENT_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown experiment log fields: {unknown}")
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Experiment log not found: {path}")

    ensure_experiment_log_schema(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"Experiment log has no header: {path}")

    match_idx = [i for i, row in enumerate(rows) if row.get("experiment_id") == experiment_id]
    if not match_idx:
        raise KeyError(f"No experiment_id={experiment_id!r} row in {path}")
    target = match_idx[0] if which == "first" else match_idx[-1]
    previous = dict(rows[target])
    updated = dict(previous)
    for key, value in fields.items():
        updated[key] = "" if value is None else value
    if merge_notes:
        raw_notes = updated.get("notes", "")
        try:
            notes_obj = json.loads(raw_notes) if str(raw_notes).strip() else {}
        except json.JSONDecodeError:
            notes_obj = {"_unparsed_notes": raw_notes}
        if not isinstance(notes_obj, dict):
            notes_obj = {"_non_object_notes": notes_obj}
        notes_obj.update(merge_notes)
        updated["notes"] = json.dumps(notes_obj, ensure_ascii=False)
    rows[target] = {col: updated.get(col, "") if updated.get(col, "") is not None else "" for col in fieldnames}

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "path": str(path),
        "experiment_id": experiment_id,
        "which": which,
        "n_matches": len(match_idx),
        "row_index": target,
        "previous": previous,
        "updated": dict(rows[target]),
        "changed_fields": sorted(
            key
            for key in set(fields) | ({"notes"} if merge_notes else set())
            if str(previous.get(key, "")) != str(rows[target].get(key, ""))
        ),
    }


def to_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def save_json(value: Any, path: str | Path) -> Path:
    path = resolve_path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    return path


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = resolve_path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path
