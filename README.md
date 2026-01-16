# Hybrid Program Synthesis with RSI

A hybrid program synthesis engine implementing Recursive Self-Improvement (RSI) with process isolation and hierarchical abstraction.

## Core Architecture

The system is built on four safety pillars:

### 1. Watchdog Sandbox (Process Isolation)
Executes all generated code in a separate process/sandbox (The "Watchdog").
- **Crash Protection**: Total isolation ensures the main system never crashes, even if the AI generates infinite loops or segmentation faults.
- **Strict Monitoring**: Enforces hard timeouts (default 2.0s) and resource limits.
- **Unrestricted Learning**: Allows full python `exec()` within the box, enabling the AI to learn real coding without risking the host.

### RSI Features

### 2. Genuine RSI (Type A: Meta-Heuristic)
Unlike "Fake RSI" that relies on hardcoded prompts or LLM calls, this system implements **True Recursive Self-Improvement**:
*   **Meta-Heuristic Search:** The system maintains a persistent set of feature weights (`rsi_meta_weights.json`) that evolves via reinforcement learning. It learns *how to search* effectively.
*   **Semantic De-Bloating:** To prevent "Complexity Bloat" (a common cheat in genetic programming), the `LibraryManager` rigorously validates new concepts. Tautologies like `reverse(reverse(n))` are rejected. Only semantically novel concepts are registered.

### 3. Safe Interpreter
AST-based interpreter for DSL execution.
- Prevents usage of `eval` or `exec` within the synthesis loop.
- Only allows whitelisted atomic operations.

### 4. Hierarchical Library (DAG)
Manages learned concepts in a Directed Acyclic Graph.
- **Semantic Hashing**: Prevents duplicate functionality even if code differs.
- **Level Constraints**: Higher-level primitives can only utilize lower-level components.
- **Utility Scoring**: Automatically prunes unused or inefficient concepts.

### 5. RSI Transfer
Automated mechanism to transfer verified improvements to the core system.
- **Sandbox Verification**: Modifications are tested in the Watchdog sandbox first.
- **Atomic Updates**: Source code is updated only after dual verification passes.

## Usage

Run the main life loop:

```bash
python Systemtest.py hrm-life
```

## Files

- `Systemtest.py`: Main entry point containing the inlined Safe RSI engine.
- `watchdog_executor.py`: Standalone reference implementation of the process isolation module.
- `rsi_primitive_registry.json`: Persisted library of learned primitives (DAG structure) and their weights.

## License

MIT
