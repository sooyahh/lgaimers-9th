"""Unit tests for read-only Git preflight verdicts. Does not touch remotes."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.git_preflight import (
    EXIT_BLOCKED,
    EXIT_SAFE,
    VERDICT_BLOCKED,
    VERDICT_CHECK,
    VERDICT_COMMIT,
    VERDICT_PR,
    VERDICT_PUSH,
    GitSnapshot,
    _diff_check,
    collect_snapshot,
    detect_in_progress,
    evaluate,
    explicit_head_refspec,
    fetch_origin,
    format_report,
    is_unexpected_path,
    is_wrong_upstream_only_block,
    main,
    parse_porcelain,
    parse_porcelain_z,
    staged_matches_allowlist,
)


def _snap(**kwargs: object) -> GitSnapshot:
    data = dict(
        branch="exp/fe-002-prev1-shrinkage",
        head="abc123",
        origin_main="def456",
        ahead_main=1,
        behind_main=0,
        commits_since_main=["abc123 experiment: reject FE-002"],
        changed_files=["scripts/run_fe002_prev1_shrinkage.py"],
        staged_files=[],
        unstaged_files=[],
        untracked_files=[],
        diff_check_errors=[],
        upstream="origin/exp/fe-002-prev1-shrinkage",
        ahead_upstream=0,
        behind_upstream=0,
        git_error="",
    )
    data.update(kwargs)
    return GitSnapshot(**data)  # type: ignore[arg-type]


class GitPreflightVerdictTests(unittest.TestCase):
    def test_main_branch_blocked(self) -> None:
        result = evaluate("check", _snap(branch="main", ahead_main=0, commits_since_main=[], changed_files=[]))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertEqual(result.exit_code, EXIT_BLOCKED)
        self.assertTrue(any("protected branch" in reason for reason in result.reasons))
        self.assertIn("VERDICT=BLOCKED", format_report(result))

    def test_clean_feature_branch(self) -> None:
        result = evaluate("check", _snap())
        self.assertEqual(result.verdict, VERDICT_CHECK)
        self.assertEqual(result.exit_code, EXIT_SAFE)
        self.assertFalse(result.reasons)
        self.assertIn("VERDICT=CHECK_PASS", format_report(result))

    def test_check_mode_success_verdict(self) -> None:
        result = evaluate("check", _snap())
        self.assertEqual(result.verdict, VERDICT_CHECK)
        self.assertIn("VERDICT=CHECK_PASS", format_report(result))

    def test_commit_mode_success_verdict(self) -> None:
        result = evaluate(
            "commit",
            _snap(staged_files=["scripts/git_preflight.py"]),
            allow_files=("scripts/git_preflight.py",),
        )
        self.assertEqual(result.verdict, VERDICT_COMMIT)
        self.assertIn("VERDICT=SAFE_TO_COMMIT", format_report(result))

    def test_push_mode_success_verdict(self) -> None:
        result = evaluate("push", _snap())
        self.assertEqual(result.verdict, VERDICT_PUSH)
        self.assertEqual(result.exit_code, EXIT_SAFE)
        self.assertIn("VERDICT=SAFE_TO_PUSH", format_report(result))

    def test_pr_mode_success_verdict(self) -> None:
        result = evaluate("pr", _snap())
        self.assertEqual(result.verdict, VERDICT_PR)
        self.assertEqual(result.exit_code, EXIT_SAFE)
        self.assertIn("VERDICT=SAFE_TO_PR", format_report(result))

    def test_fetch_failure_blocks_push_and_pr(self) -> None:
        from io import StringIO
        from unittest.mock import patch

        for mode in ("push", "pr"):
            with self.subTest(mode=mode):
                stdout = StringIO()
                with patch("scripts.git_preflight.fetch_origin", side_effect=RuntimeError("unable to access origin")):
                    with patch("sys.stdout", stdout):
                        code = main(["--mode", mode])
                self.assertEqual(code, EXIT_BLOCKED)
                text = stdout.getvalue()
                self.assertIn("VERDICT=BLOCKED", text)
                self.assertIn("git fetch origin failed", text)

    def test_diff_check_failure(self) -> None:
        result = evaluate(
            "check",
            _snap(diff_check_errors=["configs/tm003a.yaml:149: new blank line at EOF."]),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertEqual(result.exit_code, EXIT_BLOCKED)
        self.assertTrue(any("diff --check" in reason for reason in result.reasons))

    def test_unexpected_staged_scope(self) -> None:
        result = evaluate(
            "commit",
            _snap(staged_files=["data/raw/train.csv", "scripts/git_preflight.py"]),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("unexpected staged path" in reason for reason in result.reasons))
        self.assertTrue(any("data/raw/train.csv" in reason for reason in result.reasons))

    def test_command_failure_handling(self) -> None:
        result = evaluate("push", _snap(git_error="fatal: not a git repository"))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertEqual(result.exit_code, EXIT_BLOCKED)
        self.assertTrue(any("git command failed" in reason for reason in result.reasons))

    def test_commit_mode_requires_staged_files(self) -> None:
        result = evaluate("commit", _snap(staged_files=[]))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("nothing staged" in reason for reason in result.reasons))

    def test_commit_mode_allows_expected_staged_file(self) -> None:
        result = evaluate(
            "commit",
            _snap(
                staged_files=["scripts/git_preflight.py"],
                unstaged_files=[],
                untracked_files=[],
            ),
            allow_files=("scripts/git_preflight.py",),
        )
        self.assertEqual(result.verdict, VERDICT_COMMIT)
        self.assertEqual(result.exit_code, EXIT_SAFE)

    def test_allowlist_rejects_extra_staged_file(self) -> None:
        result = evaluate(
            "commit",
            _snap(staged_files=["scripts/git_preflight.py", "README.md"]),
            allow_files=("scripts/git_preflight.py",),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("outside allowed scope" in reason for reason in result.reasons))

    def test_force_push_needed_is_blocked(self) -> None:
        result = evaluate("push", _snap(ahead_upstream=1, behind_upstream=1))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("force push is forbidden" in reason for reason in result.reasons))

    def test_collect_snapshot_command_failure(self) -> None:
        with patch("scripts.git_preflight.run_git", return_value=(128, "", "fatal: not a git repository")):
            snap = collect_snapshot()
        self.assertTrue(snap.git_error)
        result = evaluate("check", snap)
        self.assertEqual(result.verdict, VERDICT_BLOCKED)

    def test_parse_porcelain_and_unexpected_helpers(self) -> None:
        staged, unstaged, untracked = parse_porcelain(
            "A  scripts/git_preflight.py\n M tests/test_git_preflight.py\n?? .env\n"
        )
        self.assertEqual(staged, ["scripts/git_preflight.py"])
        self.assertEqual(unstaged, ["tests/test_git_preflight.py"])
        self.assertEqual(untracked, [".env"])
        self.assertTrue(is_unexpected_path("data/raw/train.csv"))
        self.assertTrue(is_unexpected_path(".env"))
        self.assertFalse(is_unexpected_path("outputs/experiments.csv"))
        self.assertFalse(is_unexpected_path("scripts/git_preflight.py"))

    def test_push_blocks_upstream_origin_main(self) -> None:
        result = evaluate("push", _snap(upstream="origin/main"))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("origin/main" in reason for reason in result.reasons))

    def test_push_blocks_upstream_other_origin_branch(self) -> None:
        result = evaluate("push", _snap(upstream="origin/exp/other-branch"))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("origin/exp/other-branch" in reason for reason in result.reasons))

    def test_push_blocks_upstream_other_remote(self) -> None:
        result = evaluate("push", _snap(upstream="fork/exp/fe-002-prev1-shrinkage"))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("fork/" in reason for reason in result.reasons))

    def test_push_allows_missing_upstream_on_feature_branch(self) -> None:
        result = evaluate("push", _snap(upstream="", ahead_upstream=0, behind_upstream=0))
        self.assertEqual(result.verdict, VERDICT_PUSH)
        self.assertEqual(result.exit_code, EXIT_SAFE)

    def test_commit_blocks_unrelated_readme_outside_allowlist(self) -> None:
        result = evaluate(
            "commit",
            _snap(staged_files=["scripts/git_preflight.py", "README.md"]),
            allow_files=("scripts/git_preflight.py",),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("README.md" in reason for reason in result.reasons))

    def test_commit_safe_when_allowlist_matches_staged_exactly(self) -> None:
        result = evaluate(
            "commit",
            _snap(staged_files=["scripts/git_preflight.py", "tests/test_git_preflight.py"]),
            allow_files=("scripts/git_preflight.py", "tests/test_git_preflight.py"),
        )
        self.assertEqual(result.verdict, VERDICT_COMMIT)

    def test_commit_blocks_allowlist_file_not_staged(self) -> None:
        result = evaluate(
            "commit",
            _snap(staged_files=["scripts/git_preflight.py"]),
            allow_files=("scripts/git_preflight.py", "README.md"),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("not staged" in reason for reason in result.reasons))

    def test_diff_check_rc0_is_pass(self) -> None:
        with patch("scripts.git_preflight.run_git", return_value=(0, "", "")):
            self.assertEqual(_diff_check([]), [])

    def test_diff_check_rc1_is_finding(self) -> None:
        with patch(
            "scripts.git_preflight.run_git",
            return_value=(1, "file.py:1: trailing whitespace.\n", ""),
        ):
            self.assertEqual(_diff_check([]), ["file.py:1: trailing whitespace."])

    def test_diff_check_rc2_is_finding(self) -> None:
        with patch(
            "scripts.git_preflight.run_git",
            return_value=(2, "file.py:1: trailing whitespace.\n", ""),
        ):
            self.assertEqual(_diff_check([]), ["file.py:1: trailing whitespace."])

    def test_diff_check_other_rc_is_git_failure(self) -> None:
        with patch("scripts.git_preflight.run_git", return_value=(128, "", "fatal: bad revision")):
            with self.assertRaises(RuntimeError):
                _diff_check([])

    def test_wrong_upstream_only_block_true_for_origin_main(self) -> None:
        result = evaluate("push", _snap(upstream="origin/main"))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(is_wrong_upstream_only_block(result))

    def test_wrong_upstream_recovery_denied_when_other_blocks(self) -> None:
        result = evaluate(
            "push",
            _snap(upstream="origin/main", diff_check_errors=["file.py:1: trailing whitespace."]),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertFalse(is_wrong_upstream_only_block(result))

    def test_commit_without_allowlist_is_blocked(self) -> None:
        result = evaluate("commit", _snap(staged_files=["README.md"]))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertTrue(any("commit scope not confirmed" in reason for reason in result.reasons))

    def test_commit_command_does_not_change_staging_after_preflight(self) -> None:
        text = (Path(__file__).resolve().parents[1] / ".cursor/commands/git-commit.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("--mode commit", text)
        self.assertIn("--allow", text)
        self.assertIn("Do not run `git add` in this command", text)
        self.assertIn("staged set must stay unchanged", text)
        self.assertNotRegex(text, r"(?im)^\d+\.\s+Use `git add")
        self.assertNotRegex(text, r"(?im)^\d+\.\s+Run `git add")

    def test_push_command_uses_explicit_refspec(self) -> None:
        text = (Path(__file__).resolve().parents[1] / ".cursor/commands/git-push.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("HEAD:refs/heads/<current_branch>", text)
        self.assertIn("wrong-upstream-only recovery", text)
        self.assertIn("invalid_push_upstream", text)
        self.assertNotIn("or `git push` if upstream exists", text)

    def test_pr_command_does_not_jump_to_create_on_safe(self) -> None:
        text = (Path(__file__).resolve().parents[1] / ".cursor/commands/git-pr.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("If `VERDICT=SAFE_TO_PR`, go to the PR-create step.", text)
        self.assertIn("do **not** create a PR yet", text)
        self.assertIn("Case B — 최초 push", text)
        self.assertIn("HEAD:refs/heads/B", text)
        self.assertIn("wrong-upstream-only recovery", text)

    def test_preflight_module_does_not_invoke_push(self) -> None:
        src = (Path(__file__).resolve().parents[1] / "scripts/git_preflight.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("def recover_wrong_upstream", src)
        self.assertNotIn('["push"', src)
        self.assertNotIn("['push'", src)

    def test_parse_porcelain_z_hangul_space_rename_delete(self) -> None:
        blob = (
            "A  한글 파일.txt\0"
            "A  spaced name.txt\0"
            "R  renamed dest.txt\0renamed src.txt\0"
            "D  gone.txt\0"
            "?? untracked tab\there.txt\0"
        )
        staged, unstaged, untracked = parse_porcelain_z(blob)
        self.assertEqual(
            staged,
            ["한글 파일.txt", "spaced name.txt", "renamed dest.txt", "gone.txt"],
        )
        self.assertEqual(unstaged, [])
        self.assertEqual(untracked, ["untracked tab\there.txt"])

    def test_parse_porcelain_z_copy_quote_backslash(self) -> None:
        blob = 'C  copy dest.txt\0copy src.txt\0A  quote"name.txt\0A  path\\slash.txt\0'
        staged, unstaged, untracked = parse_porcelain_z(blob)
        self.assertEqual(
            staged,
            ["copy dest.txt", 'quote"name.txt', "path\\slash.txt"],
        )
        self.assertEqual(unstaged, [])
        self.assertEqual(untracked, [])

    def test_detect_in_progress_markers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            git_dir = Path(raw)
            self.assertEqual(detect_in_progress(git_dir), "")
            (git_dir / "MERGE_HEAD").write_text("abc\n", encoding="utf-8")
            self.assertEqual(detect_in_progress(git_dir), "merge_in_progress")
            (git_dir / "MERGE_HEAD").unlink()
            (git_dir / "CHERRY_PICK_HEAD").write_text("abc\n", encoding="utf-8")
            self.assertEqual(detect_in_progress(git_dir), "cherry_pick_in_progress")
            (git_dir / "CHERRY_PICK_HEAD").unlink()
            (git_dir / "REVERT_HEAD").write_text("abc\n", encoding="utf-8")
            self.assertEqual(detect_in_progress(git_dir), "revert_in_progress")
            (git_dir / "REVERT_HEAD").unlink()
            (git_dir / "rebase-merge").mkdir()
            self.assertEqual(detect_in_progress(git_dir), "rebase_in_progress")
            (git_dir / "rebase-merge").rmdir()
            (git_dir / "sequencer").mkdir()
            self.assertEqual(detect_in_progress(git_dir), "sequencer_in_progress")
            (git_dir / "sequencer").rmdir()
            (git_dir / "BISECT_LOG").write_text("git bisect start\n", encoding="utf-8")
            self.assertEqual(detect_in_progress(git_dir), "bisect_in_progress")

    def test_write_modes_block_in_progress_check_warns(self) -> None:
        snap = _snap(in_progress="merge_in_progress", staged_files=["scripts/git_preflight.py"])
        check = evaluate("check", snap)
        self.assertEqual(check.verdict, VERDICT_CHECK)
        self.assertIn("merge_in_progress", check.warnings)
        commit = evaluate("commit", snap, allow_files=("scripts/git_preflight.py",))
        self.assertEqual(commit.verdict, VERDICT_BLOCKED)
        self.assertIn("merge_in_progress", commit.codes)
        push = evaluate("push", snap)
        self.assertEqual(push.verdict, VERDICT_BLOCKED)
        self.assertIn("merge_in_progress", push.codes)

    def test_commit_ignores_unstaged_diff_check(self) -> None:
        result = evaluate(
            "commit",
            _snap(
                staged_files=["scripts/git_preflight.py"],
                diff_check_unstaged=["other.py:1: trailing whitespace."],
            ),
            allow_files=("scripts/git_preflight.py",),
        )
        self.assertEqual(result.verdict, VERDICT_COMMIT)

    def test_commit_blocks_cached_diff_check(self) -> None:
        result = evaluate(
            "commit",
            _snap(
                staged_files=["scripts/git_preflight.py"],
                diff_check_cached=["scripts/git_preflight.py:1: trailing whitespace."],
            ),
            allow_files=("scripts/git_preflight.py",),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertIn("diff_check_failed", result.codes)

    def test_push_ignores_local_unstaged_diff_check(self) -> None:
        result = evaluate(
            "push",
            _snap(diff_check_unstaged=["local.py:1: trailing whitespace."], diff_check_cached=["x"]),
        )
        self.assertEqual(result.verdict, VERDICT_PUSH)

    def test_push_blocks_committed_diff_check(self) -> None:
        result = evaluate(
            "push",
            _snap(diff_check_committed=["committed.py:1: trailing whitespace."]),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertIn("diff_check_failed", result.codes)

    def test_push_blocks_same_name_remote_diverged(self) -> None:
        result = evaluate(
            "push",
            _snap(upstream="", remote_same_exists=True, remote_same_ahead=1, remote_same_behind=1),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertIn("remote_same_branch_diverged", result.codes)

    def test_push_blocks_same_name_remote_ahead(self) -> None:
        result = evaluate(
            "push",
            _snap(upstream="", remote_same_exists=True, remote_same_ahead=1, remote_same_behind=0),
        )
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertIn("remote_same_branch_ahead", result.codes)

    def test_check_does_not_require_origin_main(self) -> None:
        result = evaluate("check", _snap(origin_main="", commits_since_main=[], changed_files=[]))
        self.assertEqual(result.verdict, VERDICT_CHECK)

    def test_push_requires_origin_main(self) -> None:
        result = evaluate("push", _snap(origin_main=""))
        self.assertEqual(result.verdict, VERDICT_BLOCKED)
        self.assertIn("origin_main_missing", result.codes)

    def test_staged_matches_allowlist_helper(self) -> None:
        self.assertTrue(staged_matches_allowlist(["a.py"], ["a.py"]))
        self.assertFalse(staged_matches_allowlist(["a.py"], ["a.py", "b.py"]))


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=preflight-test",
            "-c",
            "user.email=preflight@example.com",
            *args,
        ],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def recover_wrong_upstream_explicit(cwd: Path):
    """Command-equivalent helper: explicit refspec only. Not part of preflight."""
    snap = collect_snapshot(cwd=cwd)
    refspec = explicit_head_refspec(snap.branch)
    _git(cwd, "push", "-u", "origin", refspec)
    fetch_origin(cwd=cwd)
    return evaluate("push", collect_snapshot(cwd=cwd))


def git_pr_prepare_remote(cwd: Path) -> list[str]:
    """Command-equivalent /git-pr through remote publish, before PR create.

    Never opens a PR. Distinguishes first-push (SAFE, no upstream) from
    wrong-upstream-only recovery (BLOCKED).
    """
    steps: list[str] = []
    snap = collect_snapshot(cwd=cwd)
    result = evaluate("pr", snap)
    steps.append("preflight")
    expected = f"origin/{snap.branch}"

    if result.verdict == VERDICT_BLOCKED:
        if not is_wrong_upstream_only_block(result):
            raise RuntimeError("pr preflight blocked; recovery not allowed")
        steps.append("wrong_upstream_recovery")
        recover_wrong_upstream_explicit(cwd)
        again = evaluate("pr", collect_snapshot(cwd=cwd))
        if again.verdict != VERDICT_PR:
            raise RuntimeError("pr preflight still blocked after recovery")
        steps.append("pr_create_ready")
        return steps

    if result.verdict != VERDICT_PR:
        raise RuntimeError(f"unexpected pr verdict {result.verdict}")
    steps.append("safe_to_pr")

    if snap.upstream == expected:
        steps.append("normal_upstream")
        if snap.ahead_upstream > 0:
            refspec = explicit_head_refspec(snap.branch)
            _git(cwd, "push", "-u", "origin", refspec)
            steps.append("ahead_push")
            again = evaluate("pr", collect_snapshot(cwd=cwd))
            if again.verdict != VERDICT_PR:
                raise RuntimeError("pr preflight blocked after ahead push")
        steps.append("pr_create_ready")
        return steps

    if not snap.upstream and not snap.remote_same_exists:
        steps.append("first_push_required")
        refspec = explicit_head_refspec(snap.branch)
        _git(cwd, "push", "-u", "origin", refspec)
        fetch_origin(cwd=cwd)
        steps.append("first_push")
        after = collect_snapshot(cwd=cwd)
        again = evaluate("pr", after)
        if after.upstream != expected:
            raise RuntimeError("upstream was not origin/<branch> after first push")
        if again.verdict != VERDICT_PR:
            raise RuntimeError("pr preflight not SAFE after first push")
        steps.append("pr_create_ready")
        return steps

    raise RuntimeError("SAFE_TO_PR with unexpected upstream/remote state")


def _repo_fingerprint(cwd: Path) -> dict[str, str]:
    head = _git(cwd, "rev-parse", "HEAD").stdout.strip()
    up = _git(cwd, "rev-parse", "--abbrev-ref", "@{upstream}", check=False)
    upstream = up.stdout.strip() if up.returncode == 0 else ""
    index = _git(cwd, "write-tree").stdout.strip()
    remote_heads = _git(cwd, "ls-remote", "--heads", "origin").stdout
    return {
        "head": head,
        "upstream": upstream,
        "index": index,
        "remote_heads": remote_heads,
    }


class GitPreflightIntegrationTests(unittest.TestCase):
    def _make_cloned_repo(self, tmp: Path) -> tuple[Path, Path]:
        bare = tmp / "origin.git"
        work = tmp / "work"
        _git(tmp, "init", "--bare", "-b", "main", str(bare))
        _git(tmp, "clone", str(bare), str(work))
        (work / "README.md").write_text("init\n", encoding="utf-8", newline="\n")
        _git(work, "add", "README.md")
        _git(work, "commit", "-m", "init")
        _git(work, "push", "-u", "origin", "HEAD:refs/heads/main")
        return bare, work

    def test_integration_wrong_upstream_origin_main_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _bare, work = self._make_cloned_repo(tmp)
            _git(work, "switch", "-c", "feat/demo")
            (work / "feat.txt").write_text("x\n", encoding="utf-8", newline="\n")
            _git(work, "add", "feat.txt")
            _git(work, "commit", "-m", "feat")
            _git(work, "branch", "--set-upstream-to=origin/main")
            snap = collect_snapshot(cwd=work)
            result = evaluate("push", snap)
            self.assertEqual(snap.branch, "feat/demo")
            self.assertEqual(snap.upstream, "origin/main")
            self.assertEqual(result.verdict, VERDICT_BLOCKED)
            self.assertTrue(any("origin/main" in reason for reason in result.reasons))

    def test_integration_fetch_then_missing_upstream_push_allowed(self) -> None:
        from scripts.git_preflight import fetch_origin

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _bare, work = self._make_cloned_repo(tmp)
            _git(work, "switch", "-c", "feat/demo")
            (work / "feat.txt").write_text("x\n", encoding="utf-8", newline="\n")
            _git(work, "add", "feat.txt")
            _git(work, "commit", "-m", "feat")
            fetch_origin(cwd=work)
            snap = collect_snapshot(cwd=work)
            result = evaluate("push", snap)
            self.assertEqual(snap.branch, "feat/demo")
            self.assertFalse(snap.upstream)
            self.assertFalse(snap.git_error)
            self.assertEqual(result.verdict, VERDICT_PUSH)

    def test_integration_wrong_upstream_recovery_and_main_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _bare, work = self._make_cloned_repo(tmp)
            _git(work, "switch", "-c", "feat/demo")
            (work / "feat.txt").write_text("x\n", encoding="utf-8", newline="\n")
            _git(work, "add", "feat.txt")
            _git(work, "commit", "-m", "feat")
            _git(work, "branch", "--set-upstream-to=origin/main")
            before_main = _git(work, "rev-parse", "origin/main").stdout.strip()
            blocked = evaluate("push", collect_snapshot(cwd=work))
            self.assertEqual(blocked.verdict, VERDICT_BLOCKED)
            self.assertTrue(is_wrong_upstream_only_block(blocked))
            recovered = recover_wrong_upstream_explicit(cwd=work)
            after_main = _git(work, "rev-parse", "origin/main").stdout.strip()
            self.assertEqual(before_main, after_main)
            self.assertEqual(recovered.verdict, VERDICT_PUSH)
            self.assertEqual(recovered.snapshot.upstream, "origin/feat/demo")
            self.assertEqual(
                _git(work, "rev-parse", "--abbrev-ref", "@{upstream}").stdout.strip(),
                "origin/feat/demo",
            )
            feature_sha = _git(work, "rev-parse", "HEAD").stdout.strip()
            remote_feature = _git(work, "rev-parse", "origin/feat/demo").stdout.strip()
            self.assertEqual(feature_sha, remote_feature)
            self.assertNotEqual(after_main, remote_feature)


REPO_ROOT = Path(__file__).resolve().parents[1]


class GitPreflightStateIntegrationTests(unittest.TestCase):
    def _make_cloned_repo(self, tmp: Path) -> tuple[Path, Path]:
        bare = tmp / "origin.git"
        work = tmp / "work"
        _git(tmp, "init", "--bare", "-b", "main", str(bare))
        _git(tmp, "clone", str(bare), str(work))
        _git(work, "config", "core.autocrlf", "false")
        (work / "README.md").write_text("init\n", encoding="utf-8", newline="\n")
        _git(work, "add", "README.md")
        _git(work, "commit", "-m", "init")
        _git(work, "push", "-u", "origin", "HEAD:refs/heads/main")
        return bare, work

    def _feature_with_commit(self, work: Path, name: str = "feat/demo") -> None:
        _git(work, "switch", "-c", name, "--no-track")
        (work / "feat.txt").write_text("x\n", encoding="utf-8", newline="\n")
        _git(work, "add", "feat.txt")
        _git(work, "commit", "-m", "feat")

    def _assert_write_modes_blocked(self, work: Path, code: str) -> None:
        snap = collect_snapshot(cwd=work)
        self.assertEqual(snap.in_progress, code)
        check = evaluate("check", snap)
        reported = check.warnings + check.codes + check.reasons
        self.assertTrue(any(code in item for item in reported), reported)
        for mode in ("commit", "push", "pr"):
            result = evaluate(
                mode,
                snap,
                allow_files=tuple(snap.staged_files) if snap.staged_files else ("feat.txt",),
            )
            self.assertEqual(result.verdict, VERDICT_BLOCKED, msg=mode)
            self.assertIn(code, result.codes)

    def test_integration_merge_in_progress_blocks_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = self._make_cloned_repo(Path(raw))[1]
            self._feature_with_commit(work)
            _git(work, "switch", "main")
            _git(work, "switch", "-c", "other")
            (work / "other.txt").write_text("y\n", encoding="utf-8", newline="\n")
            _git(work, "add", "other.txt")
            _git(work, "commit", "-m", "other")
            _git(work, "switch", "feat/demo")
            _git(work, "merge", "--no-commit", "--no-ff", "other")
            self._assert_write_modes_blocked(work, "merge_in_progress")

    def test_integration_cherry_pick_in_progress_blocks_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = self._make_cloned_repo(Path(raw))[1]
            self._feature_with_commit(work)
            _git(work, "switch", "main")
            _git(work, "switch", "-c", "other", "--no-track")
            (work / "feat.txt").write_text("y\n", encoding="utf-8", newline="\n")
            _git(work, "add", "feat.txt")
            _git(work, "commit", "-m", "other")
            _git(work, "switch", "feat/demo")
            cherry = _git(work, "cherry-pick", "other", check=False)
            self.assertNotEqual(cherry.returncode, 0)
            self._assert_write_modes_blocked(work, "cherry_pick_in_progress")

    def test_integration_revert_in_progress_blocks_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = self._make_cloned_repo(Path(raw))[1]
            self._feature_with_commit(work)
            _git(work, "revert", "--no-commit", "HEAD")
            self._assert_write_modes_blocked(work, "revert_in_progress")

    def test_integration_rebase_in_progress_blocks_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = self._make_cloned_repo(Path(raw))[1]
            self._feature_with_commit(work)
            _git(work, "switch", "main")
            (work / "conflict.txt").write_text("main\n", encoding="utf-8", newline="\n")
            _git(work, "add", "conflict.txt")
            _git(work, "commit", "-m", "main-conflict")
            _git(work, "switch", "feat/demo")
            (work / "conflict.txt").write_text("feat\n", encoding="utf-8", newline="\n")
            _git(work, "add", "conflict.txt")
            _git(work, "commit", "-m", "feat-conflict")
            rebase = _git(work, "rebase", "main", check=False)
            self.assertNotEqual(rebase.returncode, 0)
            self._assert_write_modes_blocked(work, "rebase_in_progress")

    def test_integration_bisect_in_progress_blocks_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = self._make_cloned_repo(Path(raw))[1]
            self._feature_with_commit(work)
            _git(work, "bisect", "start")
            self._assert_write_modes_blocked(work, "bisect_in_progress")

    def test_integration_preflight_cli_is_read_only_except_fetch(self) -> None:
        import os
        import sys

        with tempfile.TemporaryDirectory() as raw:
            work = self._make_cloned_repo(Path(raw))[1]
            self._feature_with_commit(work)
            before = _repo_fingerprint(work)
            self.assertNotIn("refs/heads/feat/demo", before["remote_heads"])
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT)
            proc = subprocess.run(
                [sys.executable, "-m", "scripts.git_preflight", "--mode", "push"],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            after = _repo_fingerprint(work)
            self.assertIn("VERDICT=SAFE_TO_PUSH", proc.stdout)
            self.assertEqual(before["head"], after["head"])
            self.assertEqual(before["upstream"], after["upstream"])
            self.assertEqual(before["index"], after["index"])
            self.assertEqual(before["remote_heads"], after["remote_heads"])
            self.assertNotIn("refs/heads/feat/demo", after["remote_heads"])

    def test_integration_same_name_remote_divergence_blocks_push(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _bare, work = self._make_cloned_repo(tmp)
            clone2 = tmp / "clone2"
            _git(tmp, "clone", str(_bare), str(clone2))
            _git(clone2, "config", "core.autocrlf", "false")
            _git(clone2, "switch", "-c", "feat/demo")
            (clone2 / "remote.txt").write_text("remote\n", encoding="utf-8", newline="\n")
            _git(clone2, "add", "remote.txt")
            _git(clone2, "commit", "-m", "remote-feat")
            _git(clone2, "push", "-u", "origin", "HEAD:refs/heads/feat/demo")
            _git(work, "switch", "-c", "feat/demo", "--no-track")
            (work / "local.txt").write_text("local\n", encoding="utf-8", newline="\n")
            _git(work, "add", "local.txt")
            _git(work, "commit", "-m", "local-feat")
            fetch_origin(cwd=work)
            snap = collect_snapshot(cwd=work)
            self.assertFalse(snap.upstream)
            self.assertTrue(snap.remote_same_exists)
            for mode in ("push", "pr"):
                result = evaluate(mode, snap)
                self.assertEqual(result.verdict, VERDICT_BLOCKED, msg=mode)
                self.assertIn("remote_same_branch_diverged", result.codes)

    def test_integration_porcelain_z_special_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = self._make_cloned_repo(Path(raw))[1]
            _git(work, "switch", "-c", "feat/paths", "--no-track")
            (work / "한글 파일.txt").write_text("h\n", encoding="utf-8", newline="\n")
            (work / "spaced name.txt").write_text("s\n", encoding="utf-8", newline="\n")
            (work / "old name.txt").write_text("old\n", encoding="utf-8", newline="\n")
            (work / "gone.txt").write_text("gone\n", encoding="utf-8", newline="\n")
            _git(work, "add", "한글 파일.txt", "spaced name.txt", "old name.txt", "gone.txt")
            _git(work, "commit", "-m", "add paths")
            _git(work, "rm", "gone.txt")
            _git(work, "mv", "old name.txt", "renamed dest.txt")
            (work / "한글 파일.txt").write_text("h2\n", encoding="utf-8", newline="\n")
            _git(work, "add", "한글 파일.txt")
            (work / "another file.txt").write_text("a\n", encoding="utf-8", newline="\n")
            _git(work, "add", "another file.txt")
            snap = collect_snapshot(cwd=work)
            self.assertIn("한글 파일.txt", snap.staged_files)
            self.assertIn("gone.txt", snap.staged_files)
            self.assertIn("renamed dest.txt", snap.staged_files)
            self.assertIn("another file.txt", snap.staged_files)
            self.assertNotIn("old name.txt", snap.staged_files)

    def test_integration_commit_index_mutation_after_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = self._make_cloned_repo(Path(raw))[1]
            _git(work, "switch", "-c", "feat/commit", "--no-track")
            first = work / "keep.py"
            first.write_text("ok\n", encoding="utf-8", newline="\n")
            extra = work / "extra.py"
            extra.write_text("no\n", encoding="utf-8", newline="\n")
            _git(work, "add", "keep.py")
            snap = collect_snapshot(cwd=work)
            allow = tuple(snap.staged_files)
            result = evaluate("commit", snap, allow_files=allow)
            self.assertEqual(result.verdict, VERDICT_COMMIT)
            self.assertTrue(staged_matches_allowlist(allow, snap.staged_files))
            _git(work, "add", "extra.py")
            later = collect_snapshot(cwd=work)
            self.assertFalse(staged_matches_allowlist(allow, later.staged_files))
            blocked = evaluate("commit", later, allow_files=allow)
            self.assertEqual(blocked.verdict, VERDICT_BLOCKED)
            self.assertIn("allowlist_mismatch", blocked.codes)

    def test_integration_mode_diff_check_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = self._make_cloned_repo(Path(raw))[1]
            _git(work, "switch", "-c", "feat/ws", "--no-track")
            clean = work / "clean.py"
            dirty = work / "dirty.py"
            clean.write_text("x = 1\n", encoding="utf-8", newline="\n")
            dirty.write_text("x = 1\n", encoding="utf-8", newline="\n")
            _git(work, "add", "clean.py", "dirty.py")
            _git(work, "commit", "-m", "clean")
            dirty.write_text("x = 1 \n", encoding="utf-8", newline="\n")
            snap = collect_snapshot(cwd=work)
            self.assertTrue(snap.diff_check_unstaged)
            self.assertFalse(snap.diff_check_cached)
            commit = evaluate(
                "commit",
                snap,
                allow_files=tuple(snap.staged_files) if snap.staged_files else ("clean.py",),
            )
            # nothing staged; commit blocked for that reason, not unstaged whitespace
            if snap.staged_files:
                self.assertEqual(commit.verdict, VERDICT_COMMIT)
            else:
                self.assertEqual(commit.verdict, VERDICT_BLOCKED)
                self.assertIn("nothing_staged", commit.codes)
                self.assertNotIn("diff_check_failed", commit.codes)
            push = evaluate("push", snap)
            self.assertEqual(push.verdict, VERDICT_PUSH)

            _git(work, "add", "dirty.py")
            cached = collect_snapshot(cwd=work)
            self.assertTrue(cached.diff_check_cached)
            commit_cached = evaluate("commit", cached, allow_files=("dirty.py",))
            self.assertEqual(commit_cached.verdict, VERDICT_BLOCKED)
            self.assertIn("diff_check_failed", commit_cached.codes)
            push_cached = evaluate("push", cached)
            self.assertEqual(push_cached.verdict, VERDICT_PUSH)

            _git(work, "commit", "-m", "whitespace")
            committed = collect_snapshot(cwd=work)
            self.assertTrue(committed.diff_check_committed)
            push_committed = evaluate("push", committed)
            pr_committed = evaluate("pr", committed)
            self.assertEqual(push_committed.verdict, VERDICT_BLOCKED)
            self.assertEqual(pr_committed.verdict, VERDICT_BLOCKED)
            self.assertIn("diff_check_failed", push_committed.codes)

    def test_integration_pr_safe_first_publish_pushes_before_pr_create(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _bare, work = self._make_cloned_repo(tmp)
            _git(work, "switch", "-c", "feat/demo", "--no-track")
            (work / "feat.txt").write_text("x\n", encoding="utf-8", newline="\n")
            _git(work, "add", "feat.txt")
            _git(work, "commit", "-m", "feat")
            before = collect_snapshot(cwd=work)
            before_main = _git(work, "rev-parse", "origin/main").stdout.strip()
            self.assertFalse(before.upstream)
            self.assertFalse(before.remote_same_exists)
            remote_before = _git(work, "ls-remote", "--heads", "origin").stdout
            self.assertNotIn("refs/heads/feat/demo", remote_before)
            preflight = evaluate("pr", before)
            self.assertEqual(preflight.verdict, VERDICT_PR)
            self.assertFalse(is_wrong_upstream_only_block(preflight))
            steps = git_pr_prepare_remote(work)
            self.assertIn("safe_to_pr", steps)
            self.assertIn("first_push_required", steps)
            self.assertIn("first_push", steps)
            self.assertIn("pr_create_ready", steps)
            self.assertNotIn("wrong_upstream_recovery", steps)
            self.assertLess(steps.index("first_push"), steps.index("pr_create_ready"))
            self.assertLess(steps.index("first_push_required"), steps.index("first_push"))
            after_main = _git(work, "rev-parse", "origin/main").stdout.strip()
            self.assertEqual(before_main, after_main)
            self.assertEqual(
                _git(work, "rev-parse", "--abbrev-ref", "@{upstream}").stdout.strip(),
                "origin/feat/demo",
            )
            head = _git(work, "rev-parse", "HEAD").stdout.strip()
            remote = _git(work, "rev-parse", "origin/feat/demo").stdout.strip()
            self.assertEqual(head, remote)
            self.assertNotEqual(after_main, remote)
            final = evaluate("pr", collect_snapshot(cwd=work))
            self.assertEqual(final.verdict, VERDICT_PR)
            self.assertEqual(final.snapshot.upstream, "origin/feat/demo")


if __name__ == "__main__":
    unittest.main()
