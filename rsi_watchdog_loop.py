"""Bounded RSI-style improvement harness using WatchdogExecutor."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, Iterable, List, Tuple

from watchdog_executor import WatchdogExecutor
from neuro_genetic_synthesizer import NeuroGeneticSynthesizer

MAX_ROUNDS = 20
MAX_TRIALS = 50
WATCHDOG_TIMEOUT_SEC = 2.0
MAX_FILE_BYTES = 5_000_000


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    if os.path.getsize(path) > MAX_FILE_BYTES:
        return {"error": "file_too_large"}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _timestamp_dir(root: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, stamp)
    os.makedirs(path, exist_ok=True)
    return path


def generate_tasks(seed: int, count: int) -> List[Dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = seed % 9973
    tasks: List[Dict[str, Any]] = []
    task_types = ["identity", "double", "square", "increment"]
    for i in range(count):
        t = task_types[(rng + i) % len(task_types)]
        if t == "identity":
            io_pairs = [{"input": n, "output": n} for n in range(5)]
        elif t == "double":
            io_pairs = [{"input": n, "output": n * 2} for n in range(5)]
        elif t == "square":
            io_pairs = [{"input": n, "output": n * n} for n in range(5)]
        else:
            io_pairs = [{"input": n, "output": n + 1} for n in range(5)]
        tasks.append({"id": f"{t}_{seed}_{i}", "type": t, "io_pairs": io_pairs})
    return tasks


def _watchdog_eval_expr(expr: str, inp: Any, timeout: float) -> Tuple[bool, Any, str]:
    code = (
        "from neuro_genetic_synthesizer import SafeInterpreter, LibraryManager\n"
        "lib = LibraryManager()\n"
        "interp = SafeInterpreter(lib.runtime_primitives)\n"
        f"result = interp.run({expr!r}, {{'n': {inp!r}}})\n"
    )
    watchdog = WatchdogExecutor(timeout=timeout)
    result = watchdog.run_safe(code)
    if result.get("killed"):
        return False, None, "timeout"
    if not result.get("success"):
        return False, None, result.get("error", "exec_failed")
    return True, result.get("result"), ""


def watchdog_run_code(code: str, timeout: float) -> Dict[str, Any]:
    watchdog = WatchdogExecutor(timeout=timeout)
    return watchdog.run_safe(code)


def evaluate_candidate(expr: str, io_pairs: Iterable[Dict[str, Any]], timeout: float) -> Tuple[bool, str]:
    for io in io_pairs:
        ok, value, err = _watchdog_eval_expr(expr, io["input"], timeout)
        if not ok:
            return False, err
        if value != io["output"]:
            return False, "mismatch"
    return True, ""


def run_condition(
    synth: NeuroGeneticSynthesizer,
    tasks: List[Dict[str, Any]],
    synth_timeout: float,
    watchdog_timeout: float,
) -> Dict[str, Any]:
    successes = 0
    runtimes: List[float] = []
    complexities: List[int] = []
    for task in tasks:
        start = time.time()
        results = synth.synthesize(task["io_pairs"], timeout=synth_timeout)
        elapsed = time.time() - start
        runtimes.append(elapsed)
        solved = False
        if results:
            for code, _expr, _size, _score in results:
                ok, _err = evaluate_candidate(code, task["io_pairs"], watchdog_timeout)
                if ok:
                    solved = True
                    complexities.append(len(code))
                    break
        if solved:
            successes += 1
    success_rate = successes / max(1, len(tasks))
    median_runtime = statistics.median(runtimes) if runtimes else 0.0
    median_complexity = statistics.median(complexities) if complexities else 0.0
    return {
        "success_rate": success_rate,
        "median_runtime_sec": median_runtime,
        "median_complexity": median_complexity,
    }


def run_loop(rounds: int, trials: int, output_root: str, synth_timeout: float, watchdog_timeout: float) -> str:
    if rounds < 1 or rounds > MAX_ROUNDS:
        raise ValueError(f"rounds must be between 1 and {MAX_ROUNDS}")
    if trials < 1 or trials > MAX_TRIALS:
        raise ValueError(f"trials must be between 1 and {MAX_TRIALS}")

    run_dir = _timestamp_dir(output_root)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    weights_before = _read_json("rsi_meta_weights.json")
    registry_before = _read_json("rsi_primitive_registry.json")

    with open(metrics_path, "w", encoding="utf-8") as handle:
        for round_idx in range(rounds):
            tasks = generate_tasks(seed=round_idx, count=trials)
            control = NeuroGeneticSynthesizer(use_meta_heuristic=False)
            treatment = NeuroGeneticSynthesizer(use_meta_heuristic=True)

            control_metrics = run_condition(control, tasks, synth_timeout, watchdog_timeout)
            treatment_metrics = run_condition(treatment, tasks, synth_timeout, watchdog_timeout)

            record = {
                "round": round_idx,
                "control": control_metrics,
                "treatment": treatment_metrics,
            }
            handle.write(json.dumps(record) + "\n")

    summary_path = os.path.join(run_dir, "summary.md")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    control_rates = [r["control"]["success_rate"] for r in records]
    treatment_rates = [r["treatment"]["success_rate"] for r in records]
    summary = {
        "rounds": rounds,
        "trials": trials,
        "control_avg_success": statistics.mean(control_rates) if control_rates else 0.0,
        "treatment_avg_success": statistics.mean(treatment_rates) if treatment_rates else 0.0,
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# RSI Watchdog Loop Summary\n\n")
        for key, value in summary.items():
            handle.write(f"- {key}: {value}\n")

    weights_after = _read_json("rsi_meta_weights.json")
    registry_after = _read_json("rsi_primitive_registry.json")

    _write_json(os.path.join(run_dir, "meta_weights_before_after.json"), {
        "before": weights_before,
        "after": weights_after,
    })
    _write_json(os.path.join(run_dir, "registry_before_after.json"), {
        "before": registry_before,
        "after": registry_after,
    })

    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded RSI-style Watchdog loop")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--trials", type=int, default=MAX_TRIALS)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--synth-timeout", type=float, default=2.0)
    parser.add_argument("--watchdog-timeout", type=float, default=WATCHDOG_TIMEOUT_SEC)
    args = parser.parse_args()

    run_dir = run_loop(args.rounds, args.trials, args.output_root, args.synth_timeout, args.watchdog_timeout)
    print(f"Run artifacts saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
