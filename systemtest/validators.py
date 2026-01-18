"""Validation helpers for Systemtest code generation."""
from __future__ import annotations

import ast
from typing import Any, Callable, Dict, Optional, Set, Tuple

SAFE_BUILTINS: Dict[str, Callable] = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "len": len,
    "range": range,
    "list": list,
    "sorted": sorted,
    "reversed": reversed,
    "sum": sum,
}

SAFE_VARS: Set[str] = {"x"} | {f"v{i}" for i in range(10)}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "locals"}


def _forbidden_name(name: str) -> bool:
    return name.startswith("__") or name in FORBIDDEN_NAMES


class BaseValidator(ast.NodeVisitor):
    """Shared validation logic for AST validators."""

    ALLOWED: Tuple[type, ...] = tuple()

    def __init__(self, safe_builtins: Optional[Set[str]] = None):
        self.ok, self.err = (True, None)
        self.safe_builtins = safe_builtins or set(SAFE_BUILTINS.keys())

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden: {type(node).__name__}")
            return
        if isinstance(node, ast.Name) and _forbidden_name(node.id):
            self.ok, self.err = (False, f"Forbidden name: {node.id}")
            return
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
            return
        if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Name, ast.Attribute)):
            self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
            return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in self.safe_builtins:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)


def _allowed_nodes(base: Tuple[type, ...], extra: Tuple[type, ...]) -> Tuple[type, ...]:
    allowed = list(base) + list(extra)
    if hasattr(ast, "Index"):
        allowed.append(ast.Index)
    return tuple(allowed)


_COMMON_ALLOWED = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Name,
    ast.Constant,
    ast.Expr,
    ast.If,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Load,
    ast.Store,
    ast.IfExp,
    ast.operator,
    ast.boolop,
    ast.unaryop,
    ast.cmpop,
)


class CodeValidator(BaseValidator):
    """Allow a safe subset of Python with loops and basic control flow."""

    ALLOWED = _allowed_nodes(
        _COMMON_ALLOWED,
        (
            ast.While,
            ast.For,
            ast.Break,
            ast.Continue,
        ),
    )


class ProgramValidator(BaseValidator):
    """Strict program-mode validator: no loops or attributes."""

    ALLOWED = _allowed_nodes(_COMMON_ALLOWED, tuple())


class AlgoProgramValidator(BaseValidator):
    """Algo-mode validator with bounded structure and constrained attribute access."""

    ALLOWED = _allowed_nodes(
        _COMMON_ALLOWED,
        (
            ast.For,
            ast.While,
            ast.BoolOp,
        ),
    )


class ExprValidator(ast.NodeVisitor):
    """Validate a single expression (mode='eval') allowing only safe names and call forms."""

    ALLOWED = _allowed_nodes(
        (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.IfExp,
            ast.Call,
            ast.Attribute,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Set,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Subscript,
            ast.Slice,
            ast.operator,
            ast.unaryop,
            ast.boolop,
            ast.cmpop,
        ),
        tuple(),
    )

    def __init__(self, allowed_names: Set[str], safe_builtins: Optional[Set[str]] = None):
        self.allowed_names = allowed_names
        self.safe_builtins = safe_builtins or set(SAFE_BUILTINS.keys())
        self.ok = True
        self.err: Optional[str] = None

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden expr node: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if _forbidden_name(node.id):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
            if node.id not in self.allowed_names:
                self.ok, self.err = (False, f"Unknown name: {node.id}")
                return
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
            return
        if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Name, ast.Attribute)):
            self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
            return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in self.safe_builtins:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)


def validate_code(code: str, safe_builtins: Optional[Set[str]] = None) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = CodeValidator(safe_builtins=safe_builtins)
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))


def validate_program(code: str, safe_builtins: Optional[Set[str]] = None) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = ProgramValidator(safe_builtins=safe_builtins)
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))


def ast_depth(code: str) -> int:
    try:
        tree = ast.parse(code)
    except Exception:
        return 0
    max_depth = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
    return max_depth


def program_limits_ok(code: str, max_nodes: int = 200, max_depth: int = 20, max_locals: int = 16) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    nodes = sum(1 for _ in ast.walk(tree))
    depth = ast_depth(code)
    locals_set = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return nodes <= max_nodes and depth <= max_depth and len(locals_set) <= max_locals


def algo_program_limits_ok(
    code: str,
    max_nodes: int = 420,
    max_depth: int = 32,
    max_funcs: int = 8,
    max_locals: int = 48,
    max_consts: int = 128,
    max_subscripts: int = 64,
) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    nodes = sum(1 for _ in ast.walk(tree))
    depth = ast_depth(code)
    funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    locals_set = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    consts = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant))
    subs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Subscript))
    return (
        nodes <= max_nodes
        and depth <= max_depth
        and funcs <= max_funcs
        and len(locals_set) <= max_locals
        and consts <= max_consts
        and subs <= max_subscripts
    )


def validate_algo_program(code: str, safe_builtins: Optional[Set[str]] = None) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = AlgoProgramValidator(safe_builtins=safe_builtins)
        v.visit(tree)
        if not v.ok:
            return (False, v.err or "")
        if not algo_program_limits_ok(code):
            return (False, "algo_program_limits")
        return (True, "")
    except Exception as e:
        return (False, str(e))


def validate_expr(
    expr: str,
    safe_funcs: Dict[str, Callable],
    safe_builtins: Optional[Dict[str, Callable]] = None,
    safe_vars: Optional[Set[str]] = None,
    extra: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    try:
        extra = extra or set()
        safe_builtins = safe_builtins or SAFE_BUILTINS
        safe_vars = safe_vars or SAFE_VARS
        allowed = set(safe_funcs.keys()) | set(safe_builtins.keys()) | set(safe_vars) | set(extra)
        tree = ast.parse(expr, mode="eval")
        v = ExprValidator(allowed, safe_builtins=set(safe_builtins.keys()))
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))
