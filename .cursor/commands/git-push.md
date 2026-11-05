# git-push

Push the current feature branch to origin. Do not create a PR.

Follow `.cursor/rules/git-workflow.mdc`.

Current branch is `B` (`git branch --show-current`). If `B` is `main` or `master`, stop. Never push `main`.

## Default

```text
VERDICT=BLOCKED → 즉시 중단
```

유일한 예외는 아래 **wrong-upstream-only recovery**다.

## Do

1. Run `python -m scripts.git_preflight --mode push`.
2. If `VERDICT=SAFE_TO_PUSH`, go to the explicit refspec push step.
3. If `VERDICT=BLOCKED`, inspect `CODES:`.

### Wrong-upstream-only recovery (유일한 BLOCKED 예외)

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

1. Record that upstream is wrong (print `B` and current `@{upstream}`).
2. Push with an explicit destination only:

```text
git push -u origin HEAD:refs/heads/B
```

Do not use bare `git push`. Do not push `HEAD:refs/heads/main`. Do not update any other branch.
3. Confirm `git rev-parse --abbrev-ref @{upstream}` equals `origin/B`.
4. Re-run `python -m scripts.git_preflight --mode push`.
5. Continue only if `VERDICT=SAFE_TO_PUSH`. Otherwise stop.

If BLOCKED for any other reason, stop. Do not recover.

4. Re-check `origin/main..HEAD` commits and `origin/main...HEAD` files.
5. If upstream is already `origin/B` or there is no upstream (first push) and same-name remote is absent or fast-forwardable, push with:

```text
git push -u origin HEAD:refs/heads/<current_branch>
```

6. Report the remote branch. Do not open a PR. Do not merge.

## Do not

* push `main`
* `git push` with no refspec
* `git push --force` / `--force-with-lease`
* recover from BLOCKED states other than wrong-upstream-only
* create or merge a PR
* rebase, reset, amend, cherry-pick
* push a different branch than the current checkout
* invent test results or change ADOPT / HOLD / REJECT
