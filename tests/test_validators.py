import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from systemtest.validators import (
    SAFE_BUILTINS,
    SAFE_VARS,
    validate_algo_program,
    validate_code,
    validate_expr,
    validate_program,
)


def test_validate_code_allows_loops():
    code = textwrap.dedent(
        """
        def solve(x):
            total = 0
            for i in range(3):
                total += i
            return total
        """
    )
    ok, err = validate_code(code, safe_builtins=set(SAFE_BUILTINS.keys()))
    assert ok, err


def test_validate_program_rejects_loops():
    code = textwrap.dedent(
        """
        def solve(x):
            while x > 0:
                x -= 1
            return x
        """
    )
    ok, err = validate_program(code, safe_builtins=set(SAFE_BUILTINS.keys()))
    assert not ok
    assert "Forbidden" in err


def test_validate_algo_program_accepts_loop():
    code = textwrap.dedent(
        """
        def solve(inp):
            total = 0
            for x in inp:
                total += x
            return total
        """
    )
    ok, err = validate_algo_program(code, safe_builtins=set(SAFE_BUILTINS.keys()))
    assert ok, err


def test_validate_expr_rejects_import():
    safe_funcs = {"sin": lambda x: x}
    ok, err = validate_expr("__import__('os')", safe_funcs, SAFE_BUILTINS, SAFE_VARS)
    assert not ok
    assert "Forbidden" in err
