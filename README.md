# SystemTest: Structural Synthesis and Discovery

## Overview

This repository contains an autonomous algorithm discovery system implementing Recursive Self-Improvement (RSI). The system uses a hybrid Neuro-Genetic architecture to synthesize programs, discover reusable concepts, and expand its own grammar dynamically.

## System Architecture

```mermaid
graph TD
    subgraph "Decision & Logic (Python)"
        Orchestrator["Systemtest.py<br/>(Orchestrator)"]
        Brain["Neuro-Genetic Synthesizer<br/>(H-Module)"]
        Purpose["SelfPurposeEngine<br/>(Autonomous Goals)"]
    end

    subgraph "Speed Layer (Rust)"
        RustVM["rs_machine<br/>(Virtual Machine)"]
    end

    subgraph "Long-Term Memory"
        Disk[("Concept Library")]
    end

    Purpose -->|1. Define Goal| Orchestrator
    Orchestrator -->|2. Evolve Solution| Brain
    Brain -->|3. Evaluate Genomes| RustVM
    RustVM -->|4. Return Fitness| Brain
    Brain -->|5. New Concept| Disk
    Disk -.->|Reuse| Brain
```

## Core Components

* **Systemtest.py**: Main orchestrator handling the life-cycle loop, problem generation, and H-Module (Discovery) / L-Module (Execution) coordination.
* **rs_machine (Rust)**: High-performance Virtual Machine implementation using PyO3. Accelerates program evaluation by orders of magnitude compared to the Python fallback.
* **SelfPurposeEngine**: Autonomous goal definition system that detects emergent patterns in the environment to formulate internal objectives.
* **ConceptTransferEngine**: Mechanism for generalizing learned concepts to new, unseen domains (Human-Level Generalization).

## New Capabilities: High Performance & Autonomy

* **Rust Acceleration**: The core execution loop is rewriting in Rust. The system automatically detects and loads the optimized `rs_machine` binary if installed, falling back to Python transparently if not.
* **Autonomous Goal Discovery**: The system no longer relies solely on external tasks but can formulate its own "purpose" based on environmental novelty and pattern consistency.

## Installation & Build

To unlock High-Performance Mode (Rust VM), you must compile the extension:

1. **Install Rust**: Ensure you have `cargo` and the Rust toolchain installed.
2. **Install Build Tool**: `pip install maturin`
3. **Compile Extension**:
   ```bash
   cd rs_machine
   maturin develop --release
   cd ..
   ```
   *(Note: On Windows, use a Developer Command Prompt if you encounter linker errors)*

## Usage

Once built, simply run the infinite life-cycle loop. The system will automatically detect the Rust engine:

```bash
python Systemtest.py hrm-life
```

## Requirements

* Python 3.8+
* Rust Toolchain & Cargo (for compiling `rs_machine`)
* `maturin` (for building the Python-Rust bridge)
* Standard library only for fallback mode.

