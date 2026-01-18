"""Execution helpers that enforce WatchdogExecutor isolation."""
from __future__ import annotations

import re
from typing import Any, Callable, Optional, Tuple

from watchdog_executor import WatchdogExecutor


def extract_function_name(code: str) -> Optional[str]:
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else None


def build_watchdog_callable(code: str, func_name: str, timeout: float = 2.0) -> Callable[..., Any]:
    def _call(*args: Any) -> Any:
        wrapper = (
            "from neuro_genetic_synthesizer import LibraryManager\n"
            "lib = LibraryManager()\n"
            "globals().update(lib.runtime_primitives)\n"
            f"{code}\n"
            f"result = {func_name}(*{args!r})\n"
        )
        result = WatchdogExecutor(timeout=timeout).run_safe(wrapper)
        if result.get("killed") or not result.get("success"):
            return None
        return result.get("result")

    return _call


def build_watchdog_callable_from_lambda(lambda_src: str, timeout: float = 2.0) -> Callable[..., Any]:
    def _call(*args: Any) -> Any:
        wrapper = (
            "from neuro_genetic_synthesizer import LibraryManager\n"
            "lib = LibraryManager()\n"
            "globals().update(lib.runtime_primitives)\n"
            f"func = {lambda_src}\n"
            f"result = func(*{args!r})\n"
        )
        result = WatchdogExecutor(timeout=timeout).run_safe(wrapper)
        if result.get("killed") or not result.get("success"):
            return None
        return result.get("result")

    return _call
