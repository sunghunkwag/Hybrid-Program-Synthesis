# Refactor Plan

## Overview
This plan follows the required commit sequence while keeping changes bounded and testable. Each commit includes risk, tests, and rollback guidance.

## Commit Plan

### Commit 0 — Add missing inspect_checkpoint utility (preflight)
- **Goal**: Create `inspect_checkpoint.py` to satisfy required preflight compile checks.
- **Risk**: Low (new tooling script only).
- **Tests**: `python -m py_compile inspect_checkpoint.py`.
- **Rollback**: `git revert <commit>` to remove the file.

### Commit 1 — Add minimal tracing (no behavior change)
- **Goal**: Add `TRACE_EXEC=1` environment flag to log key entrypoint hits without altering logic.
- **Risk**: Low (log-only).
- **Tests**: `python -m py_compile Systemtest.py`.
- **Rollback**: `git revert <commit>`.

### Commit 2 — Remove duplicated WatchdogExecutor (single source of truth)
- **Goal**: Remove any local duplicate `WatchdogExecutor` implementation and import from `watchdog_executor.py`.
- **Risk**: Low (module-level swap).
- **Tests**: `python -m py_compile Systemtest.py watchdog_executor.py`.
- **Rollback**: `git revert <commit>`.

### Commit 3 — Ban raw exec in main process for generated code
- **Goal**: Ensure generated code executes only via `WatchdogExecutor` (child process). Replace direct `exec()` usage used for generated code and centralize helpers in `systemtest/execution.py`.
- **Risk**: Medium (execution path changes).
- **Tests**: pytest coverage for execution path + `python -m py_compile Systemtest.py`.
- **Rollback**: `git revert <commit>`.

### Commit 4 — Registry filename normalization
- **Goal**: Default to `rsi_primitive_registry.json` with fallback to `rsi_library_registry.json` (no auto-load of .bak).
- **Risk**: Low (compatibility fallback).
- **Tests**: Unit test for fallback load + `python -m py_compile library_manager.py`.
- **Rollback**: `git revert <commit>`.

### Commit 5 — Quarantine HRM v2
- **Goal**: Move unreachable HRM v2 entrypoint to `systemtest/legacy/hrm_life_v2.py` and gate it behind a subcommand if needed.
- **Risk**: Low (legacy only).
- **Tests**: `python -m py_compile Systemtest.py systemtest/legacy/hrm_life_v2.py`.
- **Rollback**: `git revert <commit>`.

### Commit 6 — Validator dedup
- **Goal**: Deduplicate validators into `systemtest/validators.py` and update call sites.
- **Risk**: Medium (behavior alignment).
- **Tests**: Golden-sample tests + pytest.
- **Rollback**: `git revert <commit>`.

### Commit 7 — Bounded RSI-style improvement harness
- **Goal**: Add `rsi_watchdog_loop.py` with explicit caps and persistence artifacts. Ensure all generated code executes through `WatchdogExecutor`.
- **Risk**: Medium (new runtime harness).
- **Tests**: pytest (caps + timeout + persistence), short harness run.
- **Rollback**: `git revert <commit>`.

## Rollback Plan
If any commit introduces regressions, revert in reverse order, starting from the last applied commit:
```
# Example: revert last 2 commits
git revert HEAD~1..HEAD
```

## Test Plan Summary
- `python -m py_compile` for all touched modules each commit.
- `pytest -q` after validator and harness changes.
- `TRACE_EXEC=1 python rsi_watchdog_loop.py --rounds 3 --trials 10` as a short bounded run.
