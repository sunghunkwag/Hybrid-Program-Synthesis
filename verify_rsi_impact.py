
import json
import os
import random
import time
import sys
from collections import defaultdict
import contextlib
import io

# Ensure local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuro_genetic_synthesizer import NeuroGeneticSynthesizer
from meta_heuristic import MetaHeuristic

RESULTS_FILE = "rsi_verification_results.json"
WEIGHTS_FILE = "rsi_meta_weights.json"

def reset_environment():
    """Reset persistent state for fair/clean experiments."""
    if os.path.exists(WEIGHTS_FILE):
        try:
            os.remove(WEIGHTS_FILE)
            print("[Setup] Deleted existing meta-weights (Fresh Start)")
        except Exception as e:
            print(f"[Setup] Failed to delete weights: {e}")

def generate_tasks(num_tasks=30):
    """Generate a mix of 3 task types."""
    tasks = []
    
    # 1. Matrix Sum (2D List -> Int)
    for _ in range(num_tasks // 3):
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        matrix = [[random.randint(1, 10) for _ in range(cols)] for _ in range(rows)]
        # Calc sum
        total = sum(sum(row) for row in matrix)
        tasks.append({'inputs': [matrix], 'output': total, 'type': 'matrix_sum'})

    # 2. List Reverse (List -> List)
    for _ in range(num_tasks // 3):
        length = random.randint(3, 6)
        lst = [random.randint(1, 20) for _ in range(length)]
        rev = lst[::-1]
        tasks.append({'inputs': [lst], 'output': rev, 'type': 'list_reverse'})
        
    # 3. Sum To N (Int -> Int, Recursion preferrable)
    for _ in range(num_tasks // 3):
        n = random.randint(3, 8)
        # Sum 1..n
        res = sum(range(1, n+1))
        tasks.append({'inputs': [n], 'output': res, 'type': 'sum_to_n'})
        
    random.shuffle(tasks)
    return tasks

def capture_weights():
    """Capture current weights from disk or default."""
    mh = MetaHeuristic(no_io=False) # Load from disk
    # Filter interesting ones
    w = mh.weights.copy()
    return w

def run_group(label, use_meta, tasks):
    print(f"\n--- Running {label} Group (Meta={use_meta}) ---")
    synth = NeuroGeneticSynthesizer(use_meta_heuristic=use_meta)
    
    success_count = 0
    failures = defaultdict(int)
    ops_distribution = defaultdict(int)
    
    start_time = time.time()
    
    for i, task in enumerate(tasks):
        # Format for synthesizer: list of dicts
        io_pairs = [{'input': task['inputs'][0], 'output': task['output']}]
        
        # Verbose progress
        io_pairs = [{'input': task['inputs'][0], 'output': task['output']}]
        
        print(f"  Trial {i+1}/{len(tasks)} ({task['type']})...")
        
        try:
            # We allow stdout to show "Learning" logs
            solutions = synth.synthesize(io_pairs, timeout=0.8) 
            
            valid = [s for s in solutions if s[3] >= 0.95]
            if valid:
                success_count += 1
                code = valid[0][0]
                # Analysis op usage
                if '(' in code:
                    main_op = code.split('(')[0]
                    ops_distribution[main_op] += 1
            else:
                failures['NoSolution'] += 1
                
        except Exception as e:
            failures['Exception'] += 1
            
    duration = time.time() - start_time
    
    return {
        'success_rate': success_count / len(tasks),
        'failures': dict(failures),
        'ops_dist': dict(ops_distribution),
        'duration': duration
    }

def verify_organic_rsi():
    # 1. Reset
    reset_environment()
    
    # 2. Generate Tasks (Same seed for fairness if needed, but we want organic)
    # Actually, we should use same tasks for Control and Treatment?
    # Yes, for A/B test.
    random.seed(999)
    tasks = generate_tasks(num_tasks=45) # 15 of each for speed/debug
    random.seed(None)
    
    # 3. Capture Initial Weights
    w_initial = capture_weights()
    
    # 4. Run Control
    # Control doesn't touch disk weights (no_io=True inside synth)
    res_control = run_group("Control", False, tasks)
    
    # 5. Run Treatment
    # Treatment updates disk weights.
    # We expect failures initially, then updates.
    res_treatment = run_group("Treatment", True, tasks)
    
    # 6. Capture Final Weights
    w_final = capture_weights()
    
    # 7. Analyze Deltas
    deltas = {}
    for k, v in w_final.items():
        v_init = w_initial.get(k, MetaHeuristic.DEFAULT_WEIGHTS.get(k, 1.0))
        if abs(v - v_init) > 0.001:
            deltas[k] = {'before': v_init, 'after': v}
            
    # 8. Report
    print("\n=== ORGANIC RSI RESULTS ===")
    print(f"{'Metric':<15} | {'Control':<10} | {'Treatment':<10}")
    print("-" * 40)
    print(f"{'Success':<15} | {res_control['success_rate']:.1%}    | {res_treatment['success_rate']:.1%}")
    print(f"{'Duration':<15} | {res_control['duration']:.1f}s      | {res_treatment['duration']:.1f}s")
    
    print("\n[Weight Changes via Learning]")
    if deltas:
        for k, info in deltas.items():
            print(f"  {k}: {info['before']:.4f} -> {info['after']:.4f}")
    else:
        print("  (No significant weight changes observed - System might be too robust or learning rate too low)")
        
    print("\n[Op Distribution Shift]")
    # Compare top ops
    c_ops = res_control['ops_dist']
    t_ops = res_treatment['ops_dist']
    all_ops = set(c_ops.keys()) | set(t_ops.keys())
    for op in sorted(list(all_ops)):
        if c_ops.get(op,0) > 0 or t_ops.get(op,0) > 0:
             print(f"  {op:<15}: Control={c_ops.get(op,0)} | Treatment={t_ops.get(op,0)}")

    # Save JSON
    final = {
        'control': res_control,
        'treatment': res_treatment,
        'weight_deltas': deltas,
        'initial_weights': w_initial,
        'final_weights': w_final
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    verify_organic_rsi()
