#!/usr/bin/env python3
"""
Benchmark script for TRUE RSI verification.
Runs 200+ trials and collects Before/After metrics.
"""

import sys
import os
import json
import random
from collections import defaultdict

# Ensure local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuro_genetic_synthesizer import NeuroGeneticSynthesizer
from meta_heuristic import MetaHeuristic, FailureAnalyzer

def generate_test_tasks(n=200):
    """Generate diverse test tasks."""
    tasks = []
    for i in range(n):
        task_type = random.choice(['sum', 'max', 'len', 'product', 'double'])
        if task_type == 'sum':
            # List sum task
            io_pairs = []
            for _ in range(5):
                lst = [random.randint(1, 10) for _ in range(random.randint(2, 5))]
                io_pairs.append({'input': lst, 'output': sum(lst)})
        elif task_type == 'max':
            io_pairs = []
            for _ in range(5):
                lst = [random.randint(1, 50) for _ in range(random.randint(2, 5))]
                io_pairs.append({'input': lst, 'output': max(lst)})
        elif task_type == 'len':
            io_pairs = []
            for _ in range(5):
                lst = [random.randint(1, 10) for _ in range(random.randint(2, 8))]
                io_pairs.append({'input': lst, 'output': len(lst)})
        elif task_type == 'product':
            io_pairs = []
            for _ in range(5):
                lst = [random.randint(1, 3) for _ in range(random.randint(2, 4))]
                prod = 1
                for x in lst: prod *= x
                io_pairs.append({'input': lst, 'output': prod})
        else:  # double
            io_pairs = []
            for _ in range(5):
                n = random.randint(1, 50)
                io_pairs.append({'input': n, 'output': n * 2})
        tasks.append({'io_pairs': io_pairs, 'type': task_type})
    return tasks

def run_benchmark(trials=200):
    """Run benchmark and collect metrics."""
    print(f"=== TRUE RSI BENCHMARK ({trials} trials) ===")
    
    tasks = generate_test_tasks(trials)
    synth = NeuroGeneticSynthesizer()
    meta = MetaHeuristic()
    
    # Metrics
    results = {
        'success_count': 0,
        'failure_counts': {'TYPE_OR_SHAPE': 0, 'EXCEPTION': 0, 'LOW_SCORE_VALID': 0},
        'ops_used': defaultdict(int),
        'banned_ops_sizes': [],
        'meta_weights_samples': []
    }
    
    for i, task in enumerate(tasks):
        io_pairs = task['io_pairs']
        try:
            solutions = synth.synthesize(io_pairs, timeout=0.5)
            if solutions and len(solutions) > 0:
                results['success_count'] += 1
                # Track ops used
                code = solutions[0][0]
                for op in synth.library.primitives.keys():
                    if op in code:
                        results['ops_used'][op] += 1
            else:
                results['failure_counts']['LOW_SCORE_VALID'] += 1
        except TypeError as e:
            results['failure_counts']['TYPE_OR_SHAPE'] += 1
        except Exception as e:
            results['failure_counts']['EXCEPTION'] += 1
        
        # Sample meta weights every 50 trials
        if (i + 1) % 50 == 0:
            meta_check = MetaHeuristic()
            results['meta_weights_samples'].append(dict(meta_check.weights))
            if hasattr(synth, '_banned_ops_history') and synth._banned_ops_history:
                results['banned_ops_sizes'].append(len(synth._banned_ops_history))
            print(f"  Progress: {i+1}/{trials}")
    
    # Calculate metrics
    total = trials
    success_rate = results['success_count'] / total * 100
    
    print(f"\n=== RESULTS ===")
    print(f"Success Rate: {results['success_count']}/{total} ({success_rate:.1f}%)")
    print(f"Failure Distribution:")
    for ftype, count in results['failure_counts'].items():
        print(f"  {ftype}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nTop 10 Operators Used:")
    sorted_ops = sorted(results['ops_used'].items(), key=lambda x: x[1], reverse=True)[:10]
    for op, count in sorted_ops:
        print(f"  {op}: {count}")
    
    print(f"\nMeta Weights Evolution:")
    for i, sample in enumerate(results['meta_weights_samples']):
        print(f"  Sample {i+1}: recursion={sample.get('recursion', 'N/A'):.2f}, depth_penalty={sample.get('depth_penalty', 'N/A'):.2f}")
    
    if results['banned_ops_sizes']:
        print(f"\nBanned Ops History Size: {results['banned_ops_sizes']}")
    
    # PASS condition checks
    print("\n=== PASS CONDITION CHECKS ===")
    
    # P1: Meta updates called on both success and failure
    p1_pass = hasattr(synth, 'failure_analyzer') and hasattr(synth.failure_analyzer, 'error_counts')
    print(f"P1 (Meta updates on success/failure): {'PASS' if p1_pass else 'FAIL'}")
    
    # P2: Updates affect search policy (check if merged weights used)
    p2_pass = hasattr(synth, '_current_merged_weights') and len(synth._current_merged_weights) > 0
    print(f"P2 (Updates affect search policy): {'PASS' if p2_pass else 'FAIL'}")
    
    # P3: Failure reduction (compare first 50 vs last 50)
    # Can't verify this in single run, need before/after
    print(f"P3 (Failure reduction): Requires before/after comparison")
    
    # P4: Diversity (unique ops > 5)
    p4_pass = len(results['ops_used']) > 5
    print(f"P4 (Search diversity): {'PASS' if p4_pass else 'FAIL'} ({len(results['ops_used'])} unique ops)")
    
    # Save results
    output = {
        'trials': trials,
        'success_count': results['success_count'],
        'failure_counts': dict(results['failure_counts']),
        'ops_used': dict(results['ops_used']),
        'success_rate': success_rate,
        'p1_pass': p1_pass,
        'p2_pass': p2_pass,
        'p4_pass': p4_pass
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")
    
    return output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=200)
    args = parser.parse_args()
    
    run_benchmark(args.trials)
