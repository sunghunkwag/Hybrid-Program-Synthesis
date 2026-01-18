# Forensics Report

## A) Dependency + connectivity map

### Systemtest.py
- **Imports**: `neuro_genetic_synthesizer`, `meta_heuristic`, `watchdog_executor`, and `systemtest` helpers along with standard libs and optional numpy/torch. (Systemtest.py:30-86)
- **Imported by**: `systemtest/legacy/hrm_life_v2.py` imports `HRMSystem` from it. (systemtest/legacy/hrm_life_v2.py:1-18)
- **Runtime role**: Core runtime entrypoint with multiple embedded subsystems and multiple `main()` definitions. (Systemtest.py:1854-1857, 2819-2835, 9632-9637, 12335-12355, 16654-16680, 16804-16805)

### watchdog_executor.py
- **Imports**: multiprocessing + stdlib only. (watchdog_executor.py:12-16)
- **Used by**: `Systemtest.py` imports `WatchdogExecutor`. (Systemtest.py:40-50)
- **Runtime role**: Core execution-safety helper for isolated exec. (watchdog_executor.py:1-120)

### safe_interpreter.py
- **Imports**: stdlib only. (safe_interpreter.py:9-11)
- **Used by**: `library_manager.py` and optional `neuro_genetic_synthesizer.py` helpers. (library_manager.py:20, 293-296, 371-373)
- **Runtime role**: DSL-only execution (no exec/eval). (safe_interpreter.py:1-55)

### library_manager.py
- **Imports**: `safe_interpreter` DSL types and utilities. (library_manager.py:20)
- **Used by**: `neuro_genetic_synthesizer.py` internal library. (neuro_genetic_synthesizer.py:367-378, 964-966)
- **Runtime role**: Primitive registry/persistence and weighted sampling. (library_manager.py:107-444)

### systemtest/validators.py
- **Imports**: stdlib only. (systemtest/validators.py:1-197)
- **Used by**: `Systemtest.py` imports validation helpers. (Systemtest.py:42-50)
- **Runtime role**: Validation policies for generated code. (systemtest/validators.py:1-197)

### systemtest/execution.py
- **Imports**: stdlib only + WatchdogExecutor. (systemtest/execution.py:1-31)
- **Used by**: `Systemtest.py` imports Watchdog-backed call helpers. (Systemtest.py:51-55)
- **Runtime role**: Watchdog-backed execution wrappers for generated code. (systemtest/execution.py:1-31)

### neuro_genetic_synthesizer.py
- **Imports**: stdlib + optional numpy. (neuro_genetic_synthesizer.py:5-18)
- **Used by**: `Systemtest.py` imports `NeuroGeneticSynthesizer`, `SafeInterpreter`, `LibraryManager`. (Systemtest.py:70-78)
- **Runtime role**: RSI synthesizer with its own `SafeInterpreter` (AST-based, no exec/eval). (neuro_genetic_synthesizer.py:36-169)

### meta_heuristic.py
- **Imports**: stdlib only. (meta_heuristic.py:2-7)
- **Used by**: `Systemtest.py` and `neuro_genetic_synthesizer.py` use MetaHeuristic. (Systemtest.py:77-78; meta_heuristic.py:243-454)
- **Runtime role**: Meta-heuristic adjustment and failure analysis. (meta_heuristic.py:10-240, 243-454)

### verify_rsi_impact.py
- **Imports**: stdlib only + local import of `NeuroGeneticSynthesizer`. (verify_rsi_impact.py:8-17, 160-166)
- **Runtime role**: Tooling/benchmark harness for A/B meta-heuristic tests. (verify_rsi_impact.py:206-408)

### analyze_reuse.py
- **Imports**: stdlib only. (analyze_reuse.py:1-2)
- **Runtime role**: Tooling script (top-level code runs immediately). (analyze_reuse.py:4-26)

### systemtest/legacy/hrm_life_v2.py
- **Imports**: `HRMSystem` from `Systemtest.py`. (systemtest/legacy/hrm_life_v2.py:1-18)
- **Runtime role**: Legacy entrypoint (not referenced elsewhere). (systemtest/legacy/hrm_life_v2.py:7-18)

### rsi_watchdog_loop.py
- **Imports**: `watchdog_executor` + `neuro_genetic_synthesizer`. (rsi_watchdog_loop.py:1-13)
- **Runtime role**: Bounded RSI-style harness with explicit caps and Watchdog enforcement. (rsi_watchdog_loop.py:14-200)

### requirements.txt
- **Runtime role**: Dependency list: numpy, torch, tqdm, colorama, psutil. (requirements.txt:1-5)

### rsi_meta_weights.json / rsi_primitive_registry.json
- **Runtime role**: Persistence artifacts for weights and primitives (loaded by `meta_heuristic.py` and `neuro_genetic_synthesizer.py`). (meta_heuristic.py:254-295; neuro_genetic_synthesizer.py:31-38)

## B) Entrypoints & call graph

### Entrypoints
- `Systemtest.py` defines multiple `main()` functions, but the last definition wins. (Systemtest.py:1854-1857, 2819-2835, 9632-9637, 12335-12355, 16654-16680)
- `Systemtest.py` uses `if __name__ == "__main__": main()` at the bottom. (Systemtest.py:16804-16805)
- `watchdog_executor.py`, `safe_interpreter.py`, `library_manager.py`, `neuro_genetic_synthesizer.py`, and `verify_rsi_impact.py` all have their own `__main__` blocks. (watchdog_executor.py:220-273; safe_interpreter.py:322-338; library_manager.py:482-505; neuro_genetic_synthesizer.py:1412-1426; verify_rsi_impact.py:406-408)

### Which `main()` wins?
- Python resolves the *last* `main()` definition in `Systemtest.py`. The final `main()` is at `Systemtest.py:16654-16680`, so it overrides the earlier `main()` definitions. (Systemtest.py:1854-1857, 2819-2835, 9632-9637, 12335-12355, 16654-16680)

### Default execution path
- `python Systemtest.py --help` → `__main__` calls the last `main()`, which creates an argparse parser and invokes `parser.parse_args()`. The `--help` flag is handled by argparse and exits after printing help (implicit behavior), but the call path is `__main__` → `main()` → argparse config. (Systemtest.py:16654-16680, 16804-16805)
- Representative runtime command: `python Systemtest.py orchestrator-smoke`
  - `__main__` → `main()` parses args → dispatch to `orchestrator_main()`. (Systemtest.py:16490-16519, 16654-16680, 16804-16805)

## C) Zombie code verification

### 1) RuntimeGuard / EXEC BANNED legacy paths
- `RuntimeGuard` always raises `RuntimeError` and every `legacy_*` execution wrapper immediately calls it, so any runtime path into these functions is guaranteed to throw. (Systemtest.py:4959-4990)
- These functions are still referenced in live helper paths, which means existing code will crash if it tries to use them. Examples:
  - `sample_batch` calls `legacy_run` when `t.target_code` is set. (Systemtest.py:5482-5486)
  - `legacy_run_mse` calls `legacy_run`. (Systemtest.py:5948-5954)
  - `algo_runner` calls `legacy_run_algo`. (Systemtest.py:5988-6005)
  - `FunctionLibrary.get_helpers` calls `legacy_evaluate_expr`. (Systemtest.py:6688-6696)
  - `_collect_outputs` falls back to `legacy_run` in non-algo modes. (Systemtest.py:8130-8134)

### 2) Orphan HRM v2
- `run_hrm_life_v2` exists only in `systemtest/legacy/hrm_life_v2.py` and is not referenced elsewhere. (systemtest/legacy/hrm_life_v2.py:7-18)

### 3) Validator duplication
- Validators now live in `systemtest/validators.py` with a single base class and policy tables. (systemtest/validators.py:1-197)
- `Systemtest.py` imports `validate_code`, `validate_program`, `validate_algo_program`, and `validate_expr` from the module. (Systemtest.py:40-49)

### 4) Architecture conflation
- `Systemtest.py` includes multiple embedded subsystems and CLI layers, including an invention controller path (`cmd_invention`) that is only reachable through the older CLI definition that is overwritten by the final `main()`. (Systemtest.py:4735-4743, 9632-9637, 16654-16680)
- The default runtime path for `Systemtest.py` is the orchestrator smoke test, not the invention controller. (Systemtest.py:16490-16519, 16654-16680)

## D) Execution-safety strategy conflict

- **DSL-only safety**: `safe_interpreter.py` explicitly forbids exec/eval and uses a DSL interpreter. (safe_interpreter.py:1-55)
- **Watchdog safety**: `watchdog_executor.py` explicitly runs code via `exec()` in a child process. (watchdog_executor.py:19-103)
- **Systemtest execution path**: Generated candidate evaluation runs through WatchdogExecutor wrappers, not raw exec. (Systemtest.py:3890-3949; systemtest/execution.py:1-31)

**Current strategy**: Use WatchdogExecutor for any generated/general Python code (Strategy B), keep SafeInterpreter for strict DSL-only tasks (Strategy A). (Systemtest.py:3890-3949; rsi_watchdog_loop.py:61-123; safe_interpreter.py:1-55; watchdog_executor.py:19-103)

## E) dead_code_candidates.json

Created at repo root as `dead_code_candidates.json` with evidence-backed candidates and reasons.
