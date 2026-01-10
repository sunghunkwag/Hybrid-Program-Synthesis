"""
NEURO-GENETIC SYNTHESIZER
Combines Evolutionary Search (Genetic Algorithm) with Neural Guidance.
NO TRANSFORMERS. Uses simple probability distributions from the Neural Guide.
"""
import random
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable


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
        'div': lambda a, b: a // b if b != 0 else 0,
        'mod': lambda a, b: a % b if b != 0 else 0,
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
# Neuro-Genetic Synthesizer
# ==============================================================================
class NeuroGeneticSynthesizer:
    """
    Combines Evolutionary Search (Genetic Algorithm) with Neural Guidance.
    NO TRANSFORMERS. Uses simple probability distributions from the Neural Guide.
    """
    def __init__(self, neural_guide=None, pop_size=200, generations=20):
        self.guide = neural_guide  # Object with get_priors(io_pairs) -> Dict[op, prob]
        self.pop_size = pop_size
        self.generations = generations
        self.interp = NeuroInterpreter()
        self.rng = random.Random()
        
        # [Fallback Neural Network]
        # If no external guide (e.g. absent Torch), use internal SimpleNN
        if self.guide is None:
            # Input: 10 sample points (flattened I/O), Hidden: 16, Output: len(ops)
            self.internal_nn = SimpleNN(input_dim=20, hidden_dim=16, output_dim=6, rng=self.rng)
            print("[NeuroGen] Internal Pure-Python Neural Network initialized (No Torch dependency).")
        else:
            self.internal_nn = None

        # Base Atoms
        self.atoms = [BSVar('n'), BSVal(0), BSVal(1), BSVal(2), BSVal(3)]
        self.ops = list(NeuroInterpreter.PRIMS.keys())
        
        # [L6 Actuation] Structural Bias (Controlled by MetaBrain)
        # Multipliers for operator probabilities: {'op_name': multiplier}
        self.structural_bias = {}

    def register_primitive(self, name: str, func: Callable):
        """Register a new primitive for synthesis."""
        self.interp.register_primitive(name, func)
        if name not in self.ops:
            self.ops.append(name)
            # Resize NN output if needed (advanced feature, skipped for simple fallback)
            print(f"[NeuroGen] Registered new primitive: {name}")

    def synthesize(self, io_pairs: List[Dict[str, Any]], deadline=None, task_id="", task_params=None, **kwargs) -> List[Tuple[str, Expr, float, float]]:
        start_time = time.time()

        # 1. Get Neural Guidance (Priors)
        # [DYNAMIC EXPANSION] Initialize priors for ALL registered ops (including new inventions)
        priors = {op: 1.0 for op in self.ops} 
        # Default biases for base ops if needed, but uniform is honest start.
        if 'mod' in priors: priors['mod'] = 0.5
        if 'if_gt' in priors: priors['if_gt'] = 0.1
        
        if self.guide:
            learned_priors = self.guide.get_priors(io_pairs)
            if learned_priors:
                priors.update(learned_priors)
        elif self.internal_nn:
            # [Neural Fallback]
            # Flatten I/O pairs to feature vector for SimpleNN
            features = []
            for i in range(10): # Take up to 10 pairs
                if i < len(io_pairs):
                    val_in = io_pairs[i]['input']
                    val_out = io_pairs[i]['output']
                    
                    # Handle non-numeric input (e.g. string for transform tasks)
                    if isinstance(val_in, (int, float)):
                        features.append(float(val_in))
                    else:
                        # Simple hack: length + hash for strings
                        features.append(float(len(str(val_in))) + (hash(str(val_in)) % 100) / 100.0)
                        
                    if isinstance(val_out, (int, float)):
                        features.append(float(val_out))
                    else:
                        features.append(float(len(str(val_out))) + (hash(str(val_out)) % 100) / 100.0)
                else:
                    features.append(0.0)
                    features.append(0.0)
            
            # Forward pass
            nn_probs = self.internal_nn.forward(features)
            
            # Update priors mapping
            op_keys = ['add', 'mul', 'sub', 'div', 'mod', 'if_gt']
            for i, op in enumerate(op_keys):
                if i < len(nn_probs):
                    priors[op] = nn_probs[i] * 5.0 # Scale up
            
            # [Neuro-Evolution] Mutate the internal brain slightly each task?
            # Ideally this happens on success, but for now we effectively do 'online learning' via mutation
            # Actually, let's mutate it if we fail (in wrapper), or just random drift here.
            self.internal_nn.mutate(self.rng, rate=0.01)

        # [L6 Actuation] Apply Structural Bias
        for op, bias in self.structural_bias.items():
            if op in priors:
                priors[op] *= bias
            elif op == 'loop' and 'if_gt' in priors: # Map abstract 'loop' to recursive depth or equivalent
                 pass # NeuroGenetic uses implicit recursion via tree depth, handled in _random_expr?
            elif op == 'branch' and 'if_gt' in priors:
                 priors['if_gt'] *= bias # Map 'branch' to if_gt

        # Normalize priors to probabilities
        total_p = sum(priors.values())
        op_probs = {k: v/total_p for k, v in priors.items()}

        # 2. Initialize Population
        population = [self._random_expr(2, op_probs) for _ in range(self.pop_size)]

        best_solution = None
        best_fitness = -1.0

        for gen in range(self.generations):
            if deadline and time.time() > deadline: break

            # Evaluate Fitness
            scored_pop = []
            for expr in population:
                fit = self._fitness(expr, io_pairs)
                scored_pop.append((fit, expr))

                if fit >= 100.0:
                    # Early Exit on Solution
                    # [Neuro Reinforcement] If internal NN exists, reinforcement learning could happen here
                    # For simple fallback, we skip backprop
                    return [(str(expr), expr, self._size(expr), fit)]

            # Sort by fitness
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            current_best = scored_pop[0]

            if current_best[0] > best_fitness:
                best_fitness = current_best[0]
                best_solution = current_best

            # Selection (Elitism + Tournament)
            next_gen = [p[1] for p in scored_pop[:10]] # Elitism

            while len(next_gen) < self.pop_size:
                parent1 = self._tournament(scored_pop)
                parent2 = self._tournament(scored_pop)

                if self.rng.random() < 0.7:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1

                if self.rng.random() < 0.3:
                    child = self._mutate(child, op_probs)

                next_gen.append(child)

            population = next_gen

        if best_solution and self.internal_nn:
            # Verbose log for debugging learning
            print(f"[NeuroGen] Best fitness: {best_fitness:.2f}")

            if best_fitness >= 99.0:
                # [REAL LEARNING]
                # If we found a perfect solution, train the neural network to prefer these operators.
                # This is Reinforcement Learning (Policy Gradient-ish) via Backprop.
                print(f"[NeuroGen] Learning from logic: {best_solution[1]}")
            
            # Count operator usage in the solution
            op_counts = {'add':0, 'mul':0, 'sub':0, 'div':0, 'mod':0, 'if_gt':0}
            def visit(e):
                if isinstance(e, BSApp):
                    if e.func in op_counts: op_counts[e.func] += 1
                    for a in e.args: visit(a)
            visit(best_solution[1])
            
            # Find dominant operator (most used)
            # Simple strategy: Train towards the most frequent operator
            best_op = max(op_counts, key=op_counts.get)
            if op_counts[best_op] > 0:
                op_keys = ['add', 'mul', 'sub', 'div', 'mod', 'if_gt']
                try:
                    target_idx = op_keys.index(best_op)
                    # Train multiple steps to reinforce
                    for _ in range(5):
                        self.internal_nn.train(target_idx)
                    print(f"[NeuroGen] Brain updated! Reinforced '{best_op}' for this pattern.")
                except ValueError: pass

        return [(str(best_solution[1]), best_solution[1], self._size(best_solution[1]), best_fitness)] if best_solution else []

    def _random_expr(self, depth, op_probs):
        if depth <= 0 or self.rng.random() < 0.3:
            return self.rng.choice(self.atoms)

        # Choose op based on Neural Priors
        op = self.rng.choices(list(op_probs.keys()), weights=list(op_probs.values()))[0]

        # Arity check (special case for if_gt which is 4-ary)
        arity = 4 if op == 'if_gt' else 2
        args = tuple(self._random_expr(depth-1, op_probs) for _ in range(arity))
        return BSApp(op, args)

    def _fitness(self, expr, ios):
        score = 0
        hits = 0
        for io in ios:
            out = self.interp.run(expr, { 'n': io['input'] })
            if out == io['output']:
                hits += 1
                score += 1
            else:
                # Distance-based partial credit?
                if isinstance(out, (int, float)) and isinstance(io['output'], (int, float)):
                   diff = abs(out - io['output'])
                   if diff < 100: score += 1.0 / (1.0 + diff)

        # Normalize to 0-100
        return (score / len(ios)) * 100.0

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
