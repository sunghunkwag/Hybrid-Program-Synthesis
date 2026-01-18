
import json
import re
from collections import defaultdict

def analyze_rsi_impact(registry_path="rsi_primitive_registry.json"):
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except FileNotFoundError:
        print("Error: Registry file not found.")
        return

    print(f"Total Concepts: {len(registry)}")
    
    dependencies = defaultdict(set)
    reuse_counts = defaultdict(int)
    max_depth = 0
    
    # 1. Build Dependency Graph
    for name, data in registry.items():
        code = data.get('code', '')
        # Find all concept_XX usage in code
        used_concepts = re.findall(r'concept_(\d+)', code)
        
        current_id = int(name.split('_')[1])
        
        for dep_id_str in used_concepts:
            dep_name = f"concept_{dep_id_str}"
            dependencies[name].add(dep_name)
            reuse_counts[dep_name] += 1
            
            # Sanity Check: No forward dependency (DAG violation)
            if int(dep_id_str) >= current_id:
                print(f"[WARN] Forward dependency detection: {name} uses {dep_name}")

    # 2. Calculate Depth (Chain Length)
    depths = {}
    def get_depth(node):
        if node in depths: return depths[node]
        if not dependencies[node]:
            depths[node] = 1
            return 1
        
        # Max depth of children + 1
        d = 1 + max(get_depth(child) for child in dependencies[node])
        depths[node] = d
        return d

    for name in registry:
        get_depth(name)

    avg_depth = sum(depths.values()) / len(depths) if depths else 0
    max_depth = max(depths.values()) if depths else 0
    
    # 3. Analyze meaningless bloat (Heuristic)
    # Tautology check: reverse(reverse(x)), sub(x, x), etc.
    bloat_patterns = [
        (r'reverse\(reverse\(', 'Double Reverse'),
        (r'sub\((\w+), \1\)', 'Self Subtraction (Zero)'),
        (r'div\((\w+), \1\)', 'Self Division (One)'),
        (r'not_op\(not_op\(', 'Double Negation')
    ]
    
    bloat_count = 0
    for name, data in registry.items():
        code = data.get('code', '')
        for pattern, desc in bloat_patterns:
            if re.search(pattern, code):
                bloat_count += 1
                # print(f"[BLOAT] {name}: {desc} -> {code}")
                break

    # 4. Report
    print(f"\n=== RSI IMPACT ANALYSIS ===")
    print(f"1. Reuse Ratio: {len([c for c in reuse_counts if reuse_counts[c] > 0])}/{len(registry)} concepts are reused.")
    print(f"2. Max Innovation Depth: {max_depth} layers (e.g. A uses B which uses C...)")
    print(f"3. Bloat Detection: {bloat_count} concepts seem potentially redundant ({bloat_count/len(registry)*100:.1f}%)")
    
    print("\n=== TOP 5 MOST REUSED PRIMITIVES ===")
    sorted_usage = sorted(reuse_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, count in sorted_usage:
        print(f"{name}: Used by {count} higher-level concepts. Code: {registry[name]['code']}")

    print("\n=== LATEST INNOVATIONS (Highest Depth) ===")
    sorted_depth = sorted(depths.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, depth in sorted_depth:
        print(f"{name} (Depth {depth}): {registry[name]['code']}")

if __name__ == "__main__":
    analyze_rsi_impact()
