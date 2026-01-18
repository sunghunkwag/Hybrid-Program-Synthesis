"""
Checkpoint inspection utility for RSI runs.

WARNING: Pickle files can execute arbitrary code. Only inspect trusted checkpoints.
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import Any, Dict, List


def _load_checkpoint(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        return data
    return {"value": data}


def inspect_checkpoints(directory: str, limit: int) -> List[Dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    entries = sorted(os.listdir(directory))
    results: List[Dict[str, Any]] = []
    for name in entries[:limit]:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            payload = _load_checkpoint(path)
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        results.append({"file": name, "payload": payload})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect RSI checkpoint pickle files")
    parser.add_argument("--directory", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--limit", type=int, default=10, help="Max files to inspect")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Directory not found: {args.directory}")
        return 1

    results = inspect_checkpoints(args.directory, args.limit)
    for result in results:
        print(f"=== {result['file']} ===")
        for key, value in result["payload"].items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
