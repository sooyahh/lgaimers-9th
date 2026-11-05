# git-pr

Push the current feature branch if needed, then open a PR against `main`. Never merge.

Follow `.cursor/rules/git-workflow.mdc`.

Current branch is `B` (`git branch --show-current`). If `B` is `main` or `master`, stop.

## Default

```text
VERDICT=BLOCKED → 즉시 중단
```

유일한 예외는 아래 **wrong-upstream-only recovery**다.

`VERDICT=SAFE_TO_PR`이어도 PR-create step으로 바로 가지 않는다. 먼저 remote push 필요 여부를 확인한다.

## Do

1. Run `python -m scripts.git_preflight --mode pr`.
2. If `VERDICT=BLOCKED`, inspect `CODES:` and follow case C below. Do not create a PR.
3. If `VERDICT=SAFE_TO_PR`, do **not** create a PR yet. Continue with remote push checks (cases A or B).
4. Identify `B`, `@{upstream}` if any, and whether `origin/B` exists.

### Case A — 정상 upstream

`upstream == origin/B`

1. Confirm push state (`ahead`/`behind` vs `origin/B`).
2. If local is ahead, push with explicit refspec only:

```text
git push -u origin HEAD:refs/heads/B
```

Do not use bare `git push`. Do not force push.
3. If already in sync with `origin/B`, skip that push.
4. Confirm `@{upstream}` is `origin/B`.
5. Re-run `python -m scripts.git_preflight --mode pr` if a push was performed.
6. Continue only if `VERDICT=SAFE_TO_PR`.
7. Then go to the PR-create step.

### Case B — 최초 push (upstream 없음 + same-name remote 없음)

This is the normal first-publish path. It is **not** wrong-upstream recovery.

1. Confirm there is no upstream and `origin/B` does not exist.
2. Push with explicit refspec only:

```text
git push -u origin HEAD:refs/heads/B
```

Do not use bare `git push`. Do not force push. Do not push `main`.
3. Confirm the push succeeded.
4. Confirm `git rev-parse --abbrev-ref @{upstream}` equals `origin/B`.
5. Confirm `origin/B` exists and matches local HEAD.
6. Re-run `python -m scripts.git_preflight --mode pr`.
7. Continue only if `VERDICT=SAFE_TO_PR`.
8. Then go to the PR-create step.

If `origin/B` already exists while upstream is missing, do not treat this as case B. Stop unless preflight already classified it (divergence/ahead). Do not force-push.

### Case C — invalid upstream (wrong-upstream-only recovery)

Use this path **only** when all of these are true:

* `B` is not `main` / `master`
* preflight is `VERDICT=BLOCKED`
* `CODES` contains `invalid_push_upstream`
* no other blocking code
* destination will be exactly `origin/B`
* no bare `git push`
* no force push
* explicit refspec only

Then:

1. Identify the wrong upstream.
2. `git push -u origin HEAD:refs/heads/B` (never bare `git push`, never `main`).
3. Confirm `@{upstream}` is `origin/B`.
4. Re-run `python -m scripts.git_preflight --mode pr`.
5. Continue only if `VERDICT=SAFE_TO_PR`.
6. Then go to the PR-create step.

Otherwise stop. Do not recover.

### PR-create step (only after A, B, or C finished)

5. Re-check commit range, changed files, and committed-range diff-check.
6. State tests actually run vs not run. Do not claim unrun tests.
7. Create a PR targeting `main` with purpose, files, validation, tests, and unverified items.
8. Return the PR URL. Leave it OPEN.

## Do not

* merge the PR
* push or commit on `main`
* `git push` with no refspec
* go to PR-create on `SAFE_TO_PR` before the remote branch exists
* recover from BLOCKED states other than wrong-upstream-only
* force push, amend, rebase, reset, cherry-pick
* invent ADOPT / HOLD / REJECT; quote recorded experiment results only
* rewrite Git history
