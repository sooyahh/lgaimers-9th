# git-check

Read-only Git status audit. Do not modify the repository.

Follow `.cursor/rules/git-workflow.mdc`.

## Do

1. Run `python -m scripts.git_preflight --mode check`.
2. Report the full preflight output, especially `VERDICT=`, `CODES:`, and `in_progress:`.
3. `VERDICT=BLOCKED` means stop write operations. Do not commit, push, or open a PR.
4. In-progress merge/cherry-pick/revert/rebase/sequencer/bisect is reported here; write commands must not proceed.

## Do not

* commit, push, fetch, merge, rebase, reset, amend, cherry-pick, or `git add`
* force push
* invent test results
* change ADOPT / HOLD / REJECT
* rewrite Git history
