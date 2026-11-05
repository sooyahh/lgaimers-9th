# git-commit

Commit currently staged files only. Do not push.

Follow `.cursor/rules/git-workflow.mdc`.

`/git-commit` does not stage files. If the index is wrong, stop and ask the user. Do not run `git add` in this command.

`VERDICT=BLOCKED` → stop immediately. There is no recovery path in `/git-commit`.

## Do

1. Read the current staged list: `git diff --cached --name-only`.
2. If nothing is staged, stop. Do not `git add`.
3. Confirm that exact list with the user (or the user's named files). That list is the commit scope / allowlist. If scope cannot be confirmed, stop. Do not guess.
4. Run preflight with that exact allowlist:

```text
python -m scripts.git_preflight --mode commit --allow <file1> --allow <file2> ...
```

5. If `VERDICT=BLOCKED`, stop and report. Do not commit. Do not change the index.
6. Do not run `git add`, `git add -u`, `git add .`, or `git add -A`.
7. Immediately before `git commit`, re-read staged paths (`git diff --cached --name-only`). If that set is not exactly the allowlist from step 3, the staged set must stay unchanged: stop and restart from step 1. Do not `git add`.
8. Commit only when the staged set still equals the allowlist. Include experiment ID when committing experiment results.
9. After commit, run `git status` and report hash. Do not push.

## Do not

* run `git add` in this command
* commit on `main` or `master` (never, including with user approval)
* push
* force push, amend, rebase, reset, merge, cherry-pick
* include unrelated, secret, raw-data, or large artifact files
* invent test results or change ADOPT / HOLD / REJECT
