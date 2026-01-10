# SystemTest: Structural Synthesis and Discovery

## Overview
This repository contains an autonomous algorithm discovery system implementing Recursive Self-Improvement (RSI). The system uses a hybrid Neuro-Genetic architecture to synthesize programs, discover reusable concepts, and expand its own grammar dynamically.

## Core Components
*   **Systemtest.py**: Main orchestrator handling the life-cycle loop, problem generation, and H-Module (Discovery) / L-Module (Execution) coordination.
*   **neuro_genetic_synthesizer.py**: Implements a pure-Python Neural Network (`SimpleNN`) with manual Backpropagation (Stochastic Gradient Descent). It guides the genetic search without external dependencies like PyTorch.

## Capabilities
1.  **Genuine Learning**: Uses mathematical gradient descent to update neural weights based on successful program evaluations.
2.  **No Heuristics**: All hardcoded strategies and template-based searchers have been removed. Solutions are found solely through stochastic search and neural guidance.
3.  **Recursive Self-Improvement**: Functions discovered by the system are archived and dynamically loaded into the synthesis grammar, allowing the system to reuse its own inventions for future tasks.

## Usage
Run the infinite life-cycle loop:

```bash
python Systemtest.py hrm-life
```

This will start the autonomous loop:
1.  L-Module: Wakes up and attempts to solve generated tasks using the current neural model.
2.  H-Module: Dreams on accumulated experiences to discover new subroutines.
3.  Feedback: Updates neural weights and expands the primitive set with new concepts.

## Requirements
*   Python 3.8+
*   Standard library only (No `numpy`, `torch`, or other heavy dependencies required).
