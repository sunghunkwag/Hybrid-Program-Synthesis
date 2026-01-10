"""
NEURO-GENETIC SYNTHESIZER
Combines Evolutionary Search (Genetic Algorithm) with Neural Guidance.
NO TRANSFORMERS. Uses simple probability distributions from the Neural Guide.
"""
import random
import time
import math
import collections
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable


# ==============================================================================
# RUST MACHINE OPTIMIZATION
# ==============================================================================
try:
    # Trick to prioritize the installed 'rs_machine' package over the local folder
    import sys
    import os
    _original_path = list(sys.path)
    _cwd = os.getcwd()
    # Temporarily remove current dir
    sys.path = [p for p in sys.path if p != _cwd and p != '']
    
    import rs_machine
    
    # Restore path
    sys.path = _original_path
    
    if hasattr(rs_machine, "VirtualMachine"):
        HAS_RUST_VM = True
        print("[NeuroGen] [OK] Rust Virtual Machine loaded for acceleration.")
    else:
        HAS_RUST_VM = False
except ImportError:
    HAS_RUST_VM = False
    print("[NeuroGen] [INFO] Rust VM not found. Running in slow Python mode.")

class RustCompiler:
    """JIT Compiler from BSExpr (Tree) to rs_machine.Instruction (Linear)."""
    def __init__(self):
        self.code = []
        
    def compile(self, expr) -> Optional[List[Any]]:
        self.code = []
        try:
            self._compile_recursive(expr, target_reg=0)
            # Add HALT to stop execution explicitly, though VM halts on instruction end.
            # But adding HALT is safer for some loop constructs if we had them.
            # rs_machine Instruction signature: (op, a, b, c) - all ints
            return [rs_machine.Instruction(op, int(a), int(b), int(c)) for op, a, b, c in self.code]
        except Exception:
            return None
            
    def _compile_recursive(self, expr, target_reg):
        if target_reg > 7:
            raise ValueError("Register spill (depth > 8)")
            
        if isinstance(expr, BSVal):
            # SET val, 0, target_reg
            self.code.append(("SET", int(expr.val), 0, target_reg))
            
        elif isinstance(expr, BSVar):
            # Assume 'n' input is at memory[0].
            # We need a register to hold the address 0.
            # Use target_reg to hold 0, then LOAD from it.
            self.code.append(("SET", 0, 0, target_reg))    # reg = 0 (pointer)
            self.code.append(("LOAD", target_reg, 0, target_reg)) # reg = memory[reg + 0]
            
        elif isinstance(expr, BSApp):
            fn = expr.func
            
            # Binary Operators
            if fn in ['add', 'sub', 'mul', 'div']:
                # Compile LHS to target_reg
                self._compile_recursive(expr.args[0], target_reg)
                # Compile RHS to target_reg + 1
                self._compile_recursive(expr.args[1], target_reg + 1)
                
                ops = {'add': 'ADD', 'sub': 'SUB', 'mul': 'MUL', 'div': 'DIV'}
                # OP target_reg, target_reg+1, target_reg
                self.code.append((ops[fn], target_reg, target_reg + 1, target_reg))
                
            elif fn == 'mod':
                # Special case: rs_machine might not support MOD directly if not in primitives.
                # Systemtest.py instructions: MOV, SET, SWAP, ADD, SUB, MUL, DIV, INC, DEC, LOAD, STORE, LDI, STI, JMP...
                # No MOD instruction in generic rs_machine?
                # Let's check Systemtest.py's _step function.
                # op == "DIV": r[c] = r[a] / r[b]
                # No MOD. So we fail compilation for mod.
                raise ValueError("MOD instruction not supported in rs_machine")
                
            elif fn == 'if_gt':
                # if_gt(a, b, c, d) -> if a > b then c else d
                # 1. Compile A -> target
                self._compile_recursive(expr.args[0], target_reg)
                # 2. Compile B -> target + 1
                self._compile_recursive(expr.args[1], target_reg + 1)
                
                # 3. Compile D (Else) first (to verify length)
                # We need to compile to detailed lists to measure jump offsets.
                # This is tricky in one pass.
                # Strategy: Compile C and D into temp buffers.
                
                c_compiler = RustCompiler()
                c_compiler._compile_recursive(expr.args[2], target_reg) # Result to target
                c_code = c_compiler.code
                
                d_compiler = RustCompiler()
                d_compiler._compile_recursive(expr.args[3], target_reg) # Result to target
                d_code = d_compiler.code
                
                # JGT target, target+1, <skip_d_and_jump>
                # But rs_machine JGT: if r[a] > r[b] pc += c
                # Layout:
                # [A]
                # [B]
                # JGT target, target+1, len(d_code) + 2 (jump over D and the jump-over-C)
                # [D code]
                # JMP len(c_code) + 1, 0, 0
                # [C code]
                
                # Note: rs_machine JMP offset is relative to current PC?
                # Systemtest.py: st.pc += int(a). JMP 1 means next instruction?
                # No, st.pc += 1 happens automatically if no jump.
                # JMP: st.pc += int(a); jump=True.
                # So JMP 1 skips 0 instructions? 
                # If current is PC. JMP 1 sets PC = PC + 1. Next loop PC increments? 
                # Systemtest.py loop:
                #   if jump: (no increment).
                # So JMP 1 -> PC becomes PC+1. Next iter fetches PC+1. Effectively standard flow.
                # To skip N instructions, we need JMP N+1.
                # Example: JMP 2 -> skips 1 instruction.
                
                skip_d_offset = len(d_code) + 2 # Skip D + Skip JMP
                self.code.append(("JGT", target_reg, target_reg + 1, skip_d_offset))
                
                # Else Block (D)
                self.code.extend(d_code)
                
                # Jump over C
                skip_c_offset = len(c_code) + 1
                self.code.append(("JMP", skip_c_offset, 0, 0))
                
                # Then Block (C)
                self.code.extend(c_code)
                
            else:
                raise ValueError(f"Unknown op: {fn}")

# ==============================================================================
# AST Nodes

# ==============================================================================

# ==============================================================================
# PURE PYTHON NEURAL NETWORK (No External Dependencies)
# ==============================================================================
class SimpleNN:
    """
    A lightweight Multi-Layer Perceptron implementation in pure Python.
    Used as a fallback when PyTorch is not available.
    Structure: Input -> Hidden (ReLU) -> Output (Softmax)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: random.Random):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights (Xavier-like initialization)
        scale = math.sqrt(2.0 / (input_dim + hidden_dim))
        self.W1 = [[rng.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        
        scale2 = math.sqrt(2.0 / (hidden_dim + output_dim))
        self.W2 = [[rng.gauss(0, scale2) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * output_dim
        
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass causing neural activation."""
        if len(inputs) != self.input_dim:
            if len(inputs) < self.input_dim:
                inputs = inputs + [0.0] * (self.input_dim - len(inputs))
            else:
                inputs = inputs[:self.input_dim]
        
        self.last_input = inputs
        
        # Layer 1: Linear + ReLU
        self.last_hidden_pre = []
        self.last_hidden = []
        for j in range(self.hidden_dim):
            acc = self.b1[j]
            for i in range(self.input_dim):
                acc += inputs[i] * self.W1[i][j]
            self.last_hidden_pre.append(acc)
            self.last_hidden.append(max(0.0, acc)) # ReLU
            
        # Layer 2: Linear
        self.last_output_pre = []
        for j in range(self.output_dim):
            acc = self.b2[j]
            for i in range(self.hidden_dim):
                acc += self.last_hidden[i] * self.W2[i][j]
            self.last_output_pre.append(acc)
            
        # Softmax
        max_val = max(self.last_output_pre)
        exp_vals = [math.exp(v - max_val) for v in self.last_output_pre]
        sum_exp = sum(exp_vals)
        self.last_output = [v / sum_exp for v in exp_vals]
        
        return self.last_output

    def train(self, target_idx: int):
        """REAL Backpropagation (No PyTorch).
        Loss = CrossEntropy = -log(prob[target])
        Gradient of Loss w.r.t logits (z2) = p - y
        """
        if self.last_output is None: return
        
        # 1. Output Gradient
        d_z2 = list(self.last_output)
        d_z2[target_idx] -= 1.0
        
        # 2. Backprop to W2 (grad = d_z2 * h), b2 (grad = d_z2)
        # Note: W2 is [hidden][output] in previous code based on loop: W2[r][c] where r=hidden, c=output
        # BUT wait, the file showed W1[input][hidden]. So W2 is [hidden][output].
        d_W2 = [[0.0] * self.output_dim for _ in range(self.hidden_dim)]
        d_b2 = [0.0] * self.output_dim
        d_h = [0.0] * self.hidden_dim
        
        for i in range(self.output_dim):
            d_b2[i] = d_z2[i]
            for j in range(self.hidden_dim):
                # Gradient for W2[j][i]
                d_W2[j][i] = d_z2[i] * self.last_hidden[j]
                # Gradient for h[j]
                d_h[j] += d_z2[i] * self.W2[j][i]
                
        # 3. Hidden Gradient (ReLU)
        d_z1 = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            d_z1[i] = d_h[i] * (1.0 if self.last_hidden_pre[i] > 0 else 0.0)
            
        # 4. Backprop to W1 (grad = d_z1 * x), b1
        d_W1 = [[0.0] * self.hidden_dim for _ in range(self.input_dim)]
        d_b1 = [0.0] * self.hidden_dim
        
        for i in range(self.hidden_dim):
            d_b1[i] = d_z1[i]
            for j in range(self.input_dim):
                 d_W1[j][i] = d_z1[i] * self.last_input[j]
                 
        # 5. Optimization (SGD)
        lr = getattr(self, 'lr', 0.01)
        for i in range(self.output_dim):
            self.b2[i] -= lr * d_b2[i]
            for j in range(self.hidden_dim):
                self.W2[j][i] -= lr * d_W2[j][i]
                
        for i in range(self.hidden_dim):
            self.b1[i] -= lr * d_b1[i]
            for j in range(self.input_dim):
                self.W1[j][i] -= lr * d_W1[j][i]

    def mutate(self, rng: random.Random, rate: float = 0.01):
        """Neuro-Evolution: Small random weight perturbations."""
        for i in range(self.input_dim):
             for j in range(self.hidden_dim):
                 if rng.random() < rate:
                     self.W1[i][j] += rng.gauss(0, 0.01)
        for i in range(self.hidden_dim):
             for j in range(self.output_dim):
                 if rng.random() < rate:
                     self.W2[i][j] += rng.gauss(0, 0.01)
@dataclass(frozen=True)
class Expr:
    pass

@dataclass(frozen=True)
class BSVar(Expr):
    name: str = 'n'
    def __repr__(self): return self.name

@dataclass(frozen=True)
class BSVal(Expr):
    val: Any
    def __repr__(self): return str(self.val)

@dataclass(frozen=True)
class BSApp(Expr):
    func: str
    args: tuple
    def __repr__(self): 
        return f"{self.func}({', '.join(repr(a) for a in self.args)})"


# ==============================================================================
# Neuro Interpreter
# ==============================================================================
class NeuroInterpreter:
    PRIMS = {
        'add': lambda a, b: a + b,
        'mul': lambda a, b: a * b,
        'sub': lambda a, b: a - b,
        'div': lambda a, b: int(a / b) if b != 0 else 0,
        'mod': lambda a, b: int(math.fmod(a, b)) if b != 0 else 0,
        'if_gt': lambda a, b, c, d: c if a > b else d,
    }

    def run(self, expr, env):
        try:
            return self._eval(expr, env, 50)
        except:
            return None

    def _eval(self, expr, env, gas):
        if gas <= 0: return None
        if isinstance(expr, BSVar): return env.get(expr.name, 0)
        if isinstance(expr, BSVal): return expr.val
        if isinstance(expr, BSApp):
            fn = expr.func
            if fn in self.PRIMS:
                args = [self._eval(a, env, gas-1) for a in expr.args]
                if None in args: return None
                try: return self.PRIMS[fn](*args)
                except: return None
        return None

    def register_primitive(self, name: str, func: Callable):
        """Add a new primitive (discovered concept) to the interpreter."""
        self.PRIMS[name] = func



# ==============================================================================
# Novelty Detection (N-gram Rarity)
# ==============================================================================
class NoveltyScorer:
    def __init__(self):
        self.ngram_counts = collections.defaultdict(int)
        
    def _extract_ngrams(self, expr, n=3):
        ops = []
        def visit(e):
            if isinstance(e, BSApp):
                ops.append(e.func)
                for a in e.args: visit(a)
        visit(expr)
        if len(ops) < n: return []
        return [tuple(ops[i:i+n]) for i in range(len(ops)-n+1)]

    def score(self, expr) -> float:
        ngrams = self._extract_ngrams(expr)
        if not ngrams: return 0.0
        rarity_sum = 0.0
        for ng in ngrams:
            count = self.ngram_counts[ng]
            rarity_sum += 1.0 / (1.0 + math.log(1 + count))
            self.ngram_counts[ng] += 1
        return rarity_sum / len(ngrams)

# ==============================================================================
# Meta-Controller (Self-Adaptive Logic)
# ==============================================================================
@dataclass
class MetaParams:
    mutation_rate: float
    crossover_prob: float
    population_size: int

class MetaController:
    """
    RSI Engine: Monitors evolutionary progress and adapts hyperparameters.
    If stagnating -> Increase exploration (Mutation).
    If progressing -> Increase exploitation (Selection/Crossover).
    """
    def __init__(self):
        self.history = []
        self.params = MetaParams(mutation_rate=0.3, crossover_prob=0.7, population_size=200)
        self.stagnation_counter = 0
        self.last_best_fitness = -1.0
        
    def update(self, current_best_fitness: float):
        self.history.append(current_best_fitness)
        
        # Stagnation Detection
        if current_best_fitness <= self.last_best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_best_fitness = current_best_fitness
            
        # Recursive Adaptation Logic
        if self.stagnation_counter > 5:
            # Stagnating: Turbocharge Mutation (Exploration)
            self.params.mutation_rate = min(0.9, self.params.mutation_rate * 1.5)
            self.params.crossover_prob = max(0.1, self.params.crossover_prob * 0.8)
            print(f"[RSI-Meta] Stagnation detected ({self.stagnation_counter}). Increasing Mutation to {self.params.mutation_rate:.2f}")
            
        elif self.stagnation_counter == 0 and len(self.history) > 1:
            # Progressing: Stabilize (Exploitation)
            self.params.mutation_rate = max(0.1, self.params.mutation_rate * 0.9)
            self.params.crossover_prob = min(0.9, self.params.crossover_prob * 1.1)
            # print(f"[RSI-Meta] Progress detected. Stabilizing Mutation to {self.params.mutation_rate:.2f}")

    def get_params(self) -> MetaParams:
        return self.params

# ==============================================================================
# Neuro-Genetic Synthesizer (Island Model)
# ==============================================================================
class NeuroGeneticSynthesizer:
    def __init__(self, neural_guide=None, pop_size=200, generations=20, islands=3):
        self.guide = neural_guide
        self.meta = MetaController() # Initialize RSI Meta-Controller
        self.meta.params.population_size = pop_size # Sync initial param
        
        self.pop_size = pop_size
        self.generations = generations
        self.num_islands = islands
        self.interp = NeuroInterpreter()
        self.rng = random.Random()
        self.novelty = NoveltyScorer() # Novelty detection
        
        if self.guide is None:
            self.internal_nn = SimpleNN(input_dim=20, hidden_dim=16, output_dim=6, rng=self.rng)
            print("[NeuroGen] Internal Pure-Python Neural Network initialized.")
        else:
            self.internal_nn = None

        self.atoms = [BSVar('n'), BSVal(0), BSVal(1), BSVal(2), BSVal(3)]
        self.ops = list(NeuroInterpreter.PRIMS.keys())
        self.structural_bias = {}

    def register_primitive(self, name: str, func: Callable):
        self.interp.register_primitive(name, func)
        if name not in self.ops:
            self.ops.append(name)
            print(f"[NeuroGen] Registered new primitive: {name}")

    def synthesize(self, io_pairs: List[Dict[str, Any]], deadline=None, task_id="", task_params=None, **kwargs) -> List[Tuple[str, Expr, float, float]]:
        # 1. Neural Guidance (Priors)
        priors = {op: 1.0 for op in self.ops} 
        if 'mod' in priors: priors['mod'] = 0.5
        if 'if_gt' in priors: priors['if_gt'] = 0.1
        
        if self.guide:
            learned_priors = self.guide.get_priors(io_pairs)
            if learned_priors: priors.update(learned_priors)
        elif self.internal_nn:
            features = self._extract_features(io_pairs)
            nn_probs = self.internal_nn.forward(features)
            op_keys = ['add', 'mul', 'sub', 'div', 'mod', 'if_gt']
            for i, op in enumerate(op_keys):
                if i < len(nn_probs) and op in priors: priors[op] = nn_probs[i] * 5.0
            self.internal_nn.mutate(self.rng, rate=0.01)

        # Apply Structural Bias
        for op, bias in self.structural_bias.items():
            if op in priors: priors[op] *= bias

        total_p = sum(priors.values())
        op_probs = {k: v/total_p for k, v in priors.items()}

        # 2. Initialize Islands
        island_pop_size = self.pop_size // self.num_islands
        islands = [[self._random_expr(2, op_probs) for _ in range(island_pop_size)] for _ in range(self.num_islands)]
        
        best_solution = None
        best_fitness = -1.0

        for gen in range(self.generations):
            if deadline and time.time() > deadline: break
            
            # --- META-CONTROLLER UPDATE ---
            # 1. Update Meta-Controller with current best fitness
            if best_fitness > 0:
                self.meta.update(best_fitness)
            
            # 2. Retrieve Dynamic Hyperparameters
            meta_params = self.meta.get_params()
            current_mutation_rate = meta_params.mutation_rate
            current_crossover_prob = meta_params.crossover_prob
            # ------------------------------

            # Migration (Ring Topology)
            if gen > 0 and gen % 5 == 0:
                for i in range(self.num_islands):
                    target_i = (i + 1) % self.num_islands
                    # Move top 5%
                    migrants = sorted(islands[i], key=lambda e: self._fitness(e, io_pairs, fast=True), reverse=True)[:int(island_pop_size*0.05)]
                    # Replace worst in target
                    islands[target_i].sort(key=lambda e: self._fitness(e, io_pairs, fast=True))
                    islands[target_i] = migrants + islands[target_i][len(migrants):]
                    # print(f"  [Island] Migration {i}->{target_i} ({len(migrants)} units)")

            # Evolve each island
            for i in range(self.num_islands):
                scored_pop = []
                for expr in islands[i]:
                    raw_fit = self._fitness(expr, io_pairs)
                    nov_score = self.novelty.score(expr)
                    final_fit = raw_fit + (nov_score * 5.0) # Bonus for novelty
                    
                    scored_pop.append((final_fit, expr, raw_fit))
                    
                    if raw_fit >= 100.0:
                        # [Library Learning Hook]
                        # Could trigger library registration here if size is small enough
                        return [(str(expr), expr, self._size(expr), raw_fit)]

                scored_pop.sort(key=lambda x: x[0], reverse=True)
                
                # Global Best Tracking
                if scored_pop[0][2] > best_fitness:
                    best_fitness = scored_pop[0][2]
                    best_solution = (scored_pop[0][0], scored_pop[0][1], scored_pop[0][2])

                # Selection
                next_gen = [p[1] for p in scored_pop[:5]] # Elitism
                while len(next_gen) < island_pop_size:
                    p1 = self._tournament(scored_pop)
                    p2 = self._tournament(scored_pop)
                    # DYNAMIC CROSSOVER PROBABILITY
                    child = self._crossover(p1, p2) if self.rng.random() < current_crossover_prob else p1
                    # DYNAMIC MUTATION RATE
                    if self.rng.random() < current_mutation_rate: child = self._mutate(child, op_probs)
                    next_gen.append(child)
                islands[i] = next_gen

        if best_solution:
            print(f"[NeuroGen] Best fitness: {best_fitness:.2f}")
            return [(str(best_solution[1]), best_solution[1], self._size(best_solution[1]), best_fitness)]
        return []

    def _extract_features(self, io_pairs):
        features = []
        for i in range(10):
            if i < len(io_pairs):
                val_in, val_out = io_pairs[i]['input'], io_pairs[i]['output']
                features.append(float(val_in) if isinstance(val_in, (int, float)) else 0.0)
                features.append(float(val_out) if isinstance(val_out, (int, float)) else 0.0)
            else: features.extend([0.0, 0.0])
        return features

    def _fitness(self, expr, ios, fast=False):
        # 1. Try Rust Acceleration
        jit_score = None
        if HAS_RUST_VM and self.num_islands > 0: # Ensure we are in a valid state
            try:
                compiler = RustCompiler()
                instructions = compiler.compile(expr)
                if instructions:
                    # Execute on Rust VM
                    # Note: We need a fresh VM or reuse one. Creating generic VM is cheap?
                    # Systemtest.py uses VirtualMachine(max_steps=400, ...)
                    # We can instantiate rs_machine.VirtualMachine directly.
                    # rs_machine.VirtualMachine(max_steps, mem_size, stack_limit)
                    vm = rs_machine.VirtualMachine(100, 64, 16)
                    
                    score = 0
                    for io in ios:
                        # Prepare input memory. 'n' is mapped to mem[0].
                        # Rust VM execute takes inputs list.
                        inp_val = float(io['input'])
                        # If input is list, use it; if scalar, wrap.
                        if isinstance(io['input'], list):
                             # Not supported by simple compiler yet (scalar assumption)
                             raise ValueError("List input not supported in JIT")
                        
                        st = vm.execute(instructions, [inp_val])
                        
                        # Result in regs[0] (target_reg 0)
                        # We verify clean halt?
                        # if not st.halted_cleanly: score -= ...?
                        # NeuroGen simple fitness just checks equality.
                        
                        # rs_machine.ExecutionState.regs is a list
                        out = st.regs[0]
                        
                        expected = io['output']
                        if abs(out - expected) < 1e-9:
                             score += 1
                        elif not fast:
                             diff = abs(out - expected)
                             if diff < 100: score += 1.0 / (1.0 + diff)
                             
                    return (score / len(ios)) * 100.0
            except Exception as e:
                # Fallback to Python if compilation/execution fails (e.g. MOD op, depth)
                pass

        # 2. Python Fallback
        score = 0
        for io in ios:
            out = self.interp.run(expr, { 'n': io['input'] })
            if out == io['output']: score += 1
            else:
                 if not fast and isinstance(out, (int, float)) and isinstance(io['output'], (int, float)):
                     diff = abs(out - io['output'])
                     if diff < 100: score += 1.0 / (1.0 + diff)
        return (score / len(ios)) * 100.0

    def _random_expr(self, depth, op_probs):
        if depth <= 0 or self.rng.random() < 0.3:
            return self.rng.choice(self.atoms)

        # Choose op based on Neural Priors
        op = self.rng.choices(list(op_probs.keys()), weights=list(op_probs.values()))[0]

        # Arity check (special case for if_gt which is 4-ary)
        arity = 4 if op == 'if_gt' else 2
        args = tuple(self._random_expr(depth-1, op_probs) for _ in range(arity))
        return BSApp(op, args)



    def _tournament(self, scored_pop):
        # Pick k random, return best
        k = 5
        candidates = self.rng.sample(scored_pop, k)
        return max(candidates, key=lambda x: x[0])[1]

    def _crossover(self, p1, p2):
        # Subtree Exchange
        if isinstance(p1, BSApp) and isinstance(p2, BSApp) and self.rng.random() < 0.5:
            # Swap arguments
            new_args = list(p1.args)
            idx = self.rng.randint(0, len(new_args)-1)
            new_args[idx] = p2 # Graft p2 onto p1
            return BSApp(p1.func, tuple(new_args))
        return p1 # Fallback

    def _mutate(self, p, op_probs):
        # Point Mutation or Subtree Regrowth
        if self.rng.random() < 0.5:
            # Regrowth
            return self._random_expr(2, op_probs)
        else:
            # Op mutation
            if isinstance(p, BSApp):
                new_op = self.rng.choices(list(op_probs.keys()), weights=list(op_probs.values()))[0]
                arity = 4 if new_op == 'if_gt' else 2
                current_arity = 4 if p.func == 'if_gt' else 2

                if arity == current_arity:
                    return BSApp(new_op, p.args)
        return p

    def _size(self, expr):
        if isinstance(expr, BSApp):
            return 1 + sum(self._size(a) for a in expr.args)
        return 1
