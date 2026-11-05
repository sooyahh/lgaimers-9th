"""Read-only Git preflight for LG Aimers.

check/commit inspect the current repo only (no fetch).
push/pr run `git fetch origin` first so origin/main is not stale; fetch
failure is BLOCKED. Fetch updates remote-tracking refs only.

Never checkout, add, commit, push, rebase, reset, merge, cherry-pick, or amend.
Exit 0 when the selected mode is safe; non-zero when blocked.

    python -m scripts.git_preflight --mode check
    python -m scripts.git_preflight --mode commit
    python -m scripts.git_preflight --mode push
    python -m scripts.git_preflight --mode pr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

EXIT_SAFE = 0
EXIT_BLOCKED = 1

VERDICT_CHECK = "CHECK_PASS"
VERDICT_COMMIT = "SAFE_TO_COMMIT"
VERDICT_PUSH = "SAFE_TO_PUSH"
VERDICT_PR = "SAFE_TO_PR"
VERDICT_BLOCKED = "BLOCKED"
MODE_VERDICTS = {
    "check": VERDICT_CHECK,
    "commit": VERDICT_COMMIT,
    "push": VERDICT_PUSH,
    "pr": VERDICT_PR,
}
WRITE_MODES = frozenset({"commit", "push", "pr"})
PROTECTED_BRANCHES = frozenset({"main", "master"})
UNEXPECTED_PREFIXES = (
    "data/raw/",
    "data/processed/",
    "outputs/models/",
    "outputs/predictions/",
    "outputs/submissions/",
    "reports/gpt_audit/oof/",
)
UNEXPECTED_NAMES = frozenset({".env", "reports/gpt_audit.zip"})
UNEXPECTED_SUFFIXES = (".joblib",)
KEEP_NAMES = frozenset(
    {
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "outputs/models/.gitkeep",
        "outputs/predictions/.gitkeep",
        "outputs/submissions/.gitkeep",
        "outputs/experiments.csv",
    }
)
IN_PROGRESS_CODES = (
    "merge_in_progress",
    "cherry_pick_in_progress",
    "revert_in_progress",
    "rebase_in_progress",
    "sequencer_in_progress",
    "bisect_in_progress",
)


@dataclass
class GitSnapshot:
    branch: str = ""
    head: str = ""
    git_dir: str = ""
    origin_main: str = ""
    ahead_main: int = 0
    behind_main: int = 0
    commits_since_main: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    staged_files: list[str] = field(default_factory=list)
    unstaged_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)
    diff_check_cached: list[str] = field(default_factory=list)
    diff_check_unstaged: list[str] = field(default_factory=list)
    diff_check_committed: list[str] = field(default_factory=list)
    diff_check_errors: list[str] = field(default_factory=list)
    upstream: str = ""
    ahead_upstream: int = 0
    behind_upstream: int = 0
    remote_same_exists: bool = False
    remote_same_ahead: int = 0
    remote_same_behind: int = 0
    in_progress: str = ""
    git_error: str = ""


@dataclass
class PreflightResult:
    mode: str
    verdict: str
    snapshot: GitSnapshot
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return EXIT_SAFE if self.verdict != VERDICT_BLOCKED else EXIT_BLOCKED

    def add_block(self, code: str, detail: str = "") -> None:
        if code not in self.codes:
            self.codes.append(code)
        self.reasons.append(f"{code}: {detail}" if detail else code)


def git_path(path: str) -> str:
    """Keep Git's repository-relative path bytes decoded as-is."""
    return path


def is_unexpected_path(path: str) -> bool:
    rel = git_path(path)
    if rel in KEEP_NAMES:
        return False
    name = rel.rsplit("/", 1)[-1]
    if rel in UNEXPECTED_NAMES or name in UNEXPECTED_NAMES:
        return True
    if rel.endswith(UNEXPECTED_SUFFIXES):
        return True
    return any(rel.startswith(prefix) for prefix in UNEXPECTED_PREFIXES)


def unexpected_in(paths: Sequence[str]) -> list[str]:
    return [git_path(p) for p in paths if is_unexpected_path(p)]


def parse_porcelain_z(blob: str) -> tuple[list[str], list[str], list[str]]:
    """Parse `git status --porcelain=v1 -z` without stripping or quoting."""
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    tokens = blob.split("\0")
    i = 0
    while i < len(tokens):
        item = tokens[i]
        if item == "":
            i += 1
            continue
        if len(item) < 3 or item[2] != " ":
            i += 1
            continue
        xy = item[:2]
        path = item[3:]
        x, y = xy[0], xy[1]
        rename_or_copy = x in "RC" or y in "RC"
        dest = path
        if rename_or_copy:
            i += 1
            orig = tokens[i] if i < len(tokens) else ""
            # Porcelain -z: first path is destination, second is source.
            if orig and dest:
                path = dest
            i += 1
        else:
            i += 1
        if x == "?" and y == "?":
            untracked.append(path)
            continue
        if x not in {" ", "?"}:
            staged.append(path)
        if y not in {" ", "?"}:
            unstaged.append(path)
    return staged, unstaged, untracked


def parse_porcelain(text: str) -> tuple[list[str], list[str], list[str]]:
    if "\0" in text:
        return parse_porcelain_z(text)
    return parse_porcelain_z(text.replace("\n", "\0"))


def fetch_origin(cwd: Path | None = None) -> None:
    """Update remote-tracking refs only. Does not change HEAD or the working tree."""
    code, out, err = run_git(["fetch", "origin"], cwd=cwd)
    if code != 0:
        detail = (err or out).strip() or f"git fetch origin failed ({code})"
        raise RuntimeError(detail)


def run_git(args: Sequence[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _git_ok(args: Sequence[str], cwd: Path | None = None) -> str:
    code, out, err = run_git(args, cwd=cwd)
    if code != 0:
        detail = (err or out).strip() or f"git {' '.join(args)} failed ({code})"
        raise RuntimeError(detail)
    return out.rstrip("\n")


def _count_left_right(rev_a: str, rev_b: str, cwd: Path | None = None) -> tuple[int, int]:
    text = _git_ok(["rev-list", "--left-right", "--count", f"{rev_a}...{rev_b}"], cwd=cwd)
    left, right = text.split()
    return int(left), int(right)


def _diff_check(extra: Sequence[str], cwd: Path | None = None) -> list[str]:
    code, out, err = run_git(["diff", "--check", *extra], cwd=cwd)
    text = "\n".join(part for part in (out, err) if part)
    if code == 0:
        return []
    if code in (1, 2):
        return [
            line
            for line in text.splitlines()
            if line and not line.startswith("warning:")
        ]
    raise RuntimeError(text.strip() or f"git diff --check failed ({code})")


def detect_in_progress(git_dir: Path) -> str:
    if (git_dir / "MERGE_HEAD").exists():
        return "merge_in_progress"
    if (git_dir / "CHERRY_PICK_HEAD").exists():
        return "cherry_pick_in_progress"
    if (git_dir / "REVERT_HEAD").exists():
        return "revert_in_progress"
    if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
        return "rebase_in_progress"
    if (git_dir / "sequencer").is_dir():
        return "sequencer_in_progress"
    if (git_dir / "BISECT_LOG").exists():
        return "bisect_in_progress"
    return ""


def collect_snapshot(cwd: Path | None = None) -> GitSnapshot:
    snap = GitSnapshot()
    try:
        snap.git_dir = _git_ok(["rev-parse", "--absolute-git-dir"], cwd=cwd)
        snap.in_progress = detect_in_progress(Path(snap.git_dir))
        snap.branch = _git_ok(["branch", "--show-current"], cwd=cwd)
        snap.head = _git_ok(["rev-parse", "HEAD"], cwd=cwd)
        porcelain = _git_ok(["status", "--porcelain=v1", "-z", "-uall"], cwd=cwd)
        snap.staged_files, snap.unstaged_files, snap.untracked_files = parse_porcelain_z(porcelain)
        snap.diff_check_cached = _diff_check(["--cached"], cwd=cwd)
        snap.diff_check_unstaged = _diff_check([], cwd=cwd)
        main_code, main_out, main_err = run_git(["rev-parse", "--verify", "origin/main"], cwd=cwd)
        if main_code == 0 and main_out.strip():
            snap.origin_main = main_out.strip()
            snap.behind_main, snap.ahead_main = _count_left_right("origin/main", "HEAD", cwd=cwd)
            log = _git_ok(["log", "--oneline", "origin/main..HEAD"], cwd=cwd)
            snap.commits_since_main = [line for line in log.splitlines() if line]
            changed = _git_ok(["diff", "--name-only", "-z", "origin/main...HEAD"], cwd=cwd)
            snap.changed_files = [p for p in changed.split("\0") if p]
            if snap.commits_since_main:
                snap.diff_check_committed = _diff_check(["origin/main...HEAD"], cwd=cwd)
        up_code, up_out, _up_err = run_git(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            cwd=cwd,
        )
        if up_code == 0 and up_out.strip():
            snap.upstream = up_out.strip()
            snap.behind_upstream, snap.ahead_upstream = _count_left_right(
                "@{upstream}", "HEAD", cwd=cwd
            )
        if snap.branch:
            remote_ref = f"origin/{snap.branch}"
            same_code, same_out, _same_err = run_git(["rev-parse", "--verify", remote_ref], cwd=cwd)
            if same_code == 0 and same_out.strip():
                snap.remote_same_exists = True
                snap.remote_same_ahead, snap.remote_same_behind = _count_left_right(
                    remote_ref, "HEAD", cwd=cwd
                )
        snap.diff_check_errors = (
            snap.diff_check_cached + snap.diff_check_unstaged + snap.diff_check_committed
        )
    except RuntimeError as exc:
        snap.git_error = str(exc)
    return snap


def expected_origin_upstream(branch: str) -> str:
    return f"origin/{branch}"


def invalid_push_upstream(branch: str, upstream: str) -> str | None:
    """Return a block reason if upstream is set and is not origin/<branch>."""
    if not upstream:
        return None
    expected = expected_origin_upstream(branch)
    if upstream == expected:
        return None
    return f"upstream is {upstream}, expected {expected} or none (first push)"


def force_push_needed(snap: GitSnapshot) -> bool:
    return bool(snap.upstream) and snap.ahead_upstream > 0 and snap.behind_upstream > 0


def explicit_head_refspec(branch: str) -> str:
    if not branch or branch in PROTECTED_BRANCHES:
        raise RuntimeError(f"refusing explicit push destination for branch {branch!r}")
    return f"HEAD:refs/heads/{branch}"


def is_wrong_upstream_only_block(result: PreflightResult) -> bool:
    """True only when BLOCKED solely because upstream is not origin/<branch>."""
    snap = result.snapshot
    if result.verdict != VERDICT_BLOCKED:
        return False
    if not snap.branch or snap.branch in PROTECTED_BRANCHES:
        return False
    if "invalid_push_upstream" not in result.codes:
        return False
    other = [code for code in result.codes if code != "invalid_push_upstream"]
    return not other


def staged_set(paths: Sequence[str]) -> set[str]:
    return {git_path(p) for p in paths}


def staged_matches_allowlist(allow_files: Sequence[str], staged_files: Sequence[str]) -> bool:
    """True when the current staged set is exactly the confirmed commit allowlist."""
    return staged_set(allow_files) == staged_set(staged_files)


def _blocking_diff_checks(mode: str, snap: GitSnapshot) -> list[str]:
    cached = list(snap.diff_check_cached)
    unstaged = list(snap.diff_check_unstaged)
    committed = list(snap.diff_check_committed)
    if snap.diff_check_errors and not cached and not unstaged and not committed:
        cached = unstaged = committed = list(snap.diff_check_errors)
    if mode == "commit":
        return cached
    if mode in {"push", "pr"}:
        return committed
    return cached + unstaged + committed


def evaluate(
    mode: str,
    snap: GitSnapshot,
    allow_files: Sequence[str] | None = None,
) -> PreflightResult:
    result = PreflightResult(mode=mode, verdict=VERDICT_BLOCKED, snapshot=snap)
    allow = {git_path(p) for p in allow_files} if allow_files else None

    if snap.git_error:
        result.add_block("git_command_failed", f"git command failed: {snap.git_error}")
        return result

    if snap.in_progress:
        if mode in WRITE_MODES:
            result.add_block(snap.in_progress)
        else:
            result.warnings.append(snap.in_progress)

    if not snap.branch:
        result.add_block("detached_head", "detached HEAD is not a writable feature branch")
    if snap.branch in PROTECTED_BRANCHES:
        result.add_block(
            "protected_branch",
            f"protected branch {snap.branch}: commit/push/PR on main is forbidden",
        )

    blocking_diff = _blocking_diff_checks(mode, snap)
    if blocking_diff:
        result.add_block("diff_check_failed", "git diff --check failed")
        result.reasons.extend(blocking_diff[:8])

    if mode == "commit" and snap.diff_check_unstaged:
        result.warnings.append("unstaged diff-check findings are not blocking commit")
    if mode in {"push", "pr"} and (snap.diff_check_cached or snap.diff_check_unstaged):
        result.warnings.append("local unstaged/cached diff-check findings are not blocking push/pr")

    staged_unexpected = unexpected_in(snap.staged_files)
    changed_unexpected = unexpected_in(snap.changed_files)
    if staged_unexpected:
        result.add_block("unexpected_staged_path", "unexpected staged path: " + ", ".join(staged_unexpected))
    if mode in {"push", "pr"} and changed_unexpected:
        result.add_block("unexpected_committed_path", ", ".join(changed_unexpected))

    if mode == "commit":
        if allow is None:
            result.add_block(
                "commit_scope_unconfirmed",
                "commit scope not confirmed; pass --allow for each intended file",
            )
        else:
            staged = staged_set(snap.staged_files)
            extra = sorted(staged - allow)
            missing = sorted(allow - staged)
            if extra:
                result.add_block(
                    "allowlist_mismatch",
                    "staged/changed files outside allowed scope: " + ", ".join(extra),
                )
            if missing:
                result.add_block(
                    "allowlist_mismatch",
                    "allowlist files are not staged: " + ", ".join(missing),
                )
    elif allow is not None:
        extra = [git_path(p) for p in snap.changed_files if git_path(p) not in allow]
        if extra:
            result.add_block("allowlist_mismatch", ", ".join(extra))

    if mode in {"push", "pr"}:
        if not snap.origin_main:
            result.add_block("origin_main_missing")
        upstream_err = invalid_push_upstream(snap.branch, snap.upstream)
        if upstream_err:
            result.add_block("invalid_push_upstream", upstream_err)
        if snap.remote_same_exists:
            if snap.remote_same_ahead > 0 and snap.remote_same_behind > 0:
                result.add_block("remote_same_branch_diverged")
            elif snap.remote_same_ahead > 0:
                result.add_block("remote_same_branch_ahead")
            elif snap.remote_same_behind == 0:
                result.warnings.append("remote_same_branch_noop")

    if force_push_needed(snap):
        result.add_block(
            "force_push_required",
            f"diverged from {snap.upstream} (ahead {snap.ahead_upstream}, behind {snap.behind_upstream}); force push is forbidden",
        )

    if mode == "commit" and not snap.staged_files:
        result.add_block("nothing_staged", "nothing staged to commit")
    if mode in {"push", "pr"} and snap.origin_main and not snap.commits_since_main:
        result.add_block("no_unique_commits", "no commits on this branch vs origin/main")
    if mode in {"push", "pr"} and snap.upstream and snap.behind_upstream > 0 and snap.ahead_upstream == 0:
        result.add_block("behind_upstream", snap.upstream)

    if snap.behind_main > 0:
        result.warnings.append(f"behind origin/main by {snap.behind_main}")
    if snap.untracked_files:
        result.warnings.append(f"untracked files present ({len(snap.untracked_files)})")
    if mode in {"push", "pr"} and (snap.staged_files or snap.unstaged_files):
        result.warnings.append("working tree is not clean")

    if result.reasons:
        result.verdict = VERDICT_BLOCKED
        return result

    result.verdict = MODE_VERDICTS[mode]
    return result


def format_report(result: PreflightResult) -> str:
    snap = result.snapshot
    lines = [
        "Git preflight (read-only)",
        f"mode: {result.mode}",
        f"branch: {snap.branch or '(empty)'}",
        f"head: {snap.head}",
        f"origin/main: {snap.origin_main or '(none)'}",
        f"ahead/behind origin/main: {snap.ahead_main} {snap.behind_main}",
        f"upstream: {snap.upstream or '(none)'}",
        f"ahead/behind upstream: {snap.ahead_upstream} {snap.behind_upstream}",
        f"remote_same_branch: {('exists' if snap.remote_same_exists else 'absent')} ahead/behind {snap.remote_same_ahead} {snap.remote_same_behind}",
        f"in_progress: {snap.in_progress or '(none)'}",
        f"force_push_needed: {str(force_push_needed(snap)).lower()}",
        f"on_protected_branch: {str(snap.branch in PROTECTED_BRANCHES).lower()}",
        "commits origin/main..HEAD:",
    ]
    if snap.commits_since_main:
        lines.extend(f"  {item}" for item in snap.commits_since_main)
    else:
        lines.append("  (none)")
    lines.append("changed files origin/main...HEAD:")
    if snap.changed_files:
        lines.extend(f"  {item}" for item in snap.changed_files)
    else:
        lines.append("  (none)")
    lines.append("staged:")
    lines.extend(f"  {item}" for item in snap.staged_files) if snap.staged_files else lines.append("  (none)")
    lines.append("unstaged:")
    lines.extend(f"  {item}" for item in snap.unstaged_files) if snap.unstaged_files else lines.append("  (none)")
    lines.append("untracked:")
    lines.extend(f"  {item}" for item in snap.untracked_files) if snap.untracked_files else lines.append("  (none)")
    lines.append("diff-check cached: " + ("FAIL" if snap.diff_check_cached else "PASS"))
    lines.extend(f"  {item}" for item in snap.diff_check_cached)
    lines.append("diff-check unstaged: " + ("FAIL" if snap.diff_check_unstaged else "PASS"))
    lines.extend(f"  {item}" for item in snap.diff_check_unstaged)
    lines.append("diff-check committed: " + ("FAIL" if snap.diff_check_committed else "PASS"))
    lines.extend(f"  {item}" for item in snap.diff_check_committed)
    if result.codes:
        lines.append("CODES: " + ",".join(result.codes))
    if result.warnings:
        lines.append("WARNINGS:")
        lines.extend(f"- {item}" for item in result.warnings)
    if result.reasons:
        lines.append("BLOCKED:")
        lines.extend(f"- {item}" for item in result.reasons)
    lines.append(f"VERDICT={result.verdict}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Git preflight for LG Aimers.")
    parser.add_argument(
        "--mode",
        choices=("check", "commit", "push", "pr"),
        default="check",
        help="Which write action this preflight is gating.",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Exact allowlist path for commit; may be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode in {"push", "pr"}:
        try:
            fetch_origin()
        except RuntimeError as exc:
            snap = GitSnapshot(git_error=f"git fetch origin failed: {exc}")
            result = evaluate(args.mode, snap, allow_files=args.allow or None)
            print(format_report(result))
            return result.exit_code
    snap = collect_snapshot()
    result = evaluate(args.mode, snap, allow_files=args.allow or None)
    print(format_report(result))
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
