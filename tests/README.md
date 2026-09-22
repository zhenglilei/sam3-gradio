# Regression Suites

Run these commands from the repository root using the deployment's Python
environment and existing dependency paths. No real model is loaded by the
default tests; paid VLM checks remain opt-in.

| Command | Scope |
| --- | --- |
| `python scripts/run_tests.py` | Daily WebUI/backend regression |
| `python scripts/run_tests.py --suite offline` | Offline evaluation tools and artifact contracts |
| `python scripts/run_tests.py --suite components` | Custom-component Python contracts |
| `python scripts/run_tests.py --suite all` | All of the above, including component directories |
| `python scripts/run_tests.py --suite all --list` | Inspect included files without importing tests |

`python -m unittest discover -s tests -p 'test_*.py'` remains a daily WebUI
entrypoint, not a repository-wide test run. `offline_eval/` intentionally has
no `__init__.py`, so this discovery command does not recurse into the offline
suite. The full runner explicitly includes it. Keep fixture-only/backup files
out of the `test_*.py` naming convention.

The runner uses separate processes for component directories because some of
them contain the same module names (for example `test_backend_preprocess`). It
returns a failure status if any group fails, and fails rather than passing an
empty requested group. Existing `PYTHONPATH` dependency entries are retained.

## When to Run

- During a small change, run its focused module and the affected public contracts.
- Before committing, run the current branch's WebUI suite. Changes to evaluation
  tools also require `offline`; component source changes require `components`.
- Before release or after a shared change is merged into product branches, run
  `all` on each affected product branch. Identical test files do not prove that
  different production implementations behave identically.
- Do not run every historical backup branch after every local edit.
- Real-model/GPU acceptance and browser workflows are separate from these mocked
  backend and source-contract tests. Passing this runner does not replace them.

## Test Boundaries

Business, geometry, owner isolation, and concurrency assertions must not be
removed merely to reduce the test count. Configuration snapshots do not replace
real browser interaction, and regenerating a snapshot requires semantic review.

Pure validation/PVS logic fixtures can stub `fsync` within the individual test,
while still exercising real file serialization and reads. Do not stub it in the
dedicated durable round-trip, failed-commit, or cross-process locking tests.
Production durability settings must never be changed to make tests run faster.
