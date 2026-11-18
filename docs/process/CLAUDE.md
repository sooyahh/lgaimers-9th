# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DACON LG Aimers 9기 Phase 2: per-pitch binary probability prediction of `control_success`, scored by
Brier Skill Score. This is an **experiment repository**, not an application. Almost every change is
"run one experiment, record its numbers, compare it to a locked baseline." Reproducibility rules
(fixed validation, fixed seed, append-only experiment log, no artifact overwrites) are the point of
the codebase, not incidental process. Project docs (`README.md`, `CONTRIBUTING.md`, `reports/`,
`docs/GPT_HANDOFF.md`) are in Korean.

## Note on overlap with README.md

The install/command overlap with `README.md` is a deliberate choice, not drift to be cleaned up:
`README.md` is the quick-start ("how do I begin"), while this file is the detailed working reference
that carries the failure modes and traps alongside each command. Keep both; do not consolidate.

## Commands

Run everything from the repository root.

```powershell
pip install -r requirements.txt

# Tests (no competition data required — they use synthetic frames)
python -m unittest discover -s tests -v
python -m unittest tests.test_tm003a_temporal_safety -v                  # single module
python -m unittest tests.test_metrics.BrierMetricTests.test_single_class_target -v  # single test

# Generic pipeline (needs data/raw/*.csv)
python -m src.train --config configs/baseline.yaml
python -m src.predict --config configs/baseline.yaml --experiment-id EXPERIMENT_ID

# Experiment runners — always `python -m scripts.<module>`, not `python scripts/....py`
python -m scripts.run_exp004_catboost_weighted
python -m scripts.run_tm003a_expected_ivb --smoke   # newer runners have a --smoke fast path

# Synthetic smoke data (the one script invoked by path, it does not import src)
python scripts/generate_synthetic_data.py
```

`python scripts/run_exp000_target_mean.py` fails with `ModuleNotFoundError: No module named 'src'`;
some newer runners patch `sys.path` themselves, but `-m` is the rule.

`python -m src.train` **appends rows to the shared `outputs/experiments.csv`** and mints a throwaway
`temporal_YYYYMMDD_...` id. Use unittest, not `src.train`, to verify a working setup.

## Data placement

Competition CSVs are local-only (`.gitignore`s `data/raw/*`):

```
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
data/raw/trackman_history.csv   # auxiliary Trackman table used by TM-* experiments
```

## Architecture

### Two training paths — know which one you are in

1. **`src/train.py`** — the generic, config-driven pipeline (`sklearn` `Pipeline` =
   `FeatureBuilder` + model from `src/models.py`). Reads `cv.strategy` (`season_holdout` or
   `expanding_window`), dumps joblib models, OOF CSVs, and diagnostics. Random KFold is
   intentionally rejected in code. This path produced only the early smoke runs.
2. **`scripts/tree_exp_common.py::run_tree_experiment`** — the path every official `EXP-*` /
   `FE-*` / `TM-*` result actually came from. It hardcodes the three locked folds
   (`secondary_2022`, `secondary_2023`, `primary_2024`), does fold-local median imputation plus
   native CatBoost/LightGBM categorical handling (`prepare_native_frames`, no one-hot), and writes
   `outputs/predictions/<EXPERIMENT_ID>/{summary.json,fold_metrics.csv,oof_holdout_predictions.csv,report.md}`.

A new model experiment is normally a thin `scripts/run_<id>_<topic>.py` that supplies a `fit_model`
callable to `run_tree_experiment` — not a new pipeline.

### Locked validation and metric

Defined in `configs/baseline.yaml` `cv.*` and `reports/validation_policy.md`:

- Primary: train `2019–2023` → valid `2024`
- Secondary (expanding): `2019–2021→2022`, `2019–2022→2023`, `2019–2023→2024`
- Seed: `runtime.seed = 42`; `order_cols` is empty (within-season chronology is **not** established,
  and no game-level leakage-free split is claimed)

`src/metrics.py` implements the official formula. It exposes both `raw_brier_skill_score` and
`official_brier_skill_score` (`max(0, raw)`). **Compare models on raw BSS / Brier** — the official
score floor-clips at 0 and hides differences between bad models. When `r*(1-r)==0` the metric raises
rather than inventing a score; never paper over that.

Changing validation, metric, or seed invalidates comparison with existing EXP rows. Don't do it as a
side effect of another change.

### Fold-safe engineered features

`src/trackman_profiles.py` builds Trackman-derived features and is where the leakage rules live:
`allowed_trackman_seasons`, `prior_history_seasons`, `assert_trackman_temporal_safety`,
`assert_row_level_trackman_safety` (no season ≥ the validation year, never 2025, no player-ID join,
no current-pitch physics). New features enter the model through `run_tree_experiment`'s
`fold_extra_features` / `extra_numeric_cols` hooks, which assert the injected columns never touch the
shared `train_df`, keep the index aligned, and refuse to run on protected experiment ids
(`EXP-001A`–`EXP-004`). Preserve those assertions; they are the leakage test, not boilerplate.

Each feature experiment gets its own `configs/<id>.yaml` that is a byte-for-byte copy of the baseline
except `project.purpose/notes/next_action` — the runner re-asserts that `cv`, `task`, `runtime`,
CatBoost params, `exclude_cols`, and `categorical_cols` still match `baseline.yaml`.

### Experiment log

`outputs/experiments.csv` is the tracked, shared, append-only registry; schema is
`EXPERIMENT_COLUMNS` in `src/utils.py`. `append_experiment_log()` auto-fills provenance
(`author`, `git_branch`, `git_commit`, `git_dirty`) and back-fills missing columns in older rows.
`experiment_id` is a single team-wide namespace: `EXP-000`–`EXP-006`, `FE-001A`, `FE-002A`,
`TM-002A`, `TM-003A` are taken; the next official model id is **`EXP-007`**. Audits use `VAL-*` /
`AUDIT-*` prefixes. Never reuse an id, and never overwrite another experiment's directory under
`outputs/{models,predictions,submissions}/` — runners fingerprint protected paths to detect it.

## Working rules (from CONTRIBUTING.md and .cursor/rules/git-workflow.mdc)

- Never work on or push to `main`. Branch as `exp/`, `feat/`, `fix/`, `audit/`, `docs/`, `chore/`.
- One experiment changes exactly **one** core hypothesis.
- Record a result only if it was actually run. Mark anything unexecuted as `not run` / `not verified`,
  including tests. Failed experiments still get a row in `experiments.csv` with `status`/`error`.
- Report fold-level numbers and variability (`cv_std`, per-fold raw BSS), not just a mean. Never
  adopt or reject on Public LB alone; leave `public_lb_score` blank when unknown.
- Stage explicit paths (`git add <file>`), not `git add .` — the working tree routinely holds local
  artifacts, raw data, and other experiments' outputs.
- Ask before anything irreversible: PR merge, force push, `git reset --hard`, `git clean -fd`,
  history rewrite, branch/tag deletion, discarding unrelated changes.
- Prefer running official comparison experiments from a clean commit; `git_dirty=true` rows are
  treated as reproducibility warnings.
- Finish by reporting: branch, changes, commands actually run, test results (PASS/FAIL/NOT RUN),
  commit/push state, uncommitted changes, unverified items, next action.

## Where the current state lives

| Path | Contents |
|---|---|
| `outputs/experiments.csv` | official experiment registry |
| `docs/GPT_HANDOFF.md` | EXP-000–006 numbers and decision status |
| `reports/validation_policy.md` | validation lock |
| `reports/feature_safety_audit.md` | which features are considered safe (EXP-005/006 drop `pitcher_id`/`batter_id`) |
| `reports/<id>_*.md` | per-experiment writeups |
| `outputs/analysis/`, `outputs/eda/` | audit CSV/JSON/plots (tracked summaries only) |
