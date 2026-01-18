import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from library_manager import LibraryManager


def test_library_manager_loads_legacy_registry(tmp_path, monkeypatch):
    legacy_registry = tmp_path / "rsi_library_registry.json"
    legacy_registry.write_text(
        json.dumps(
            {
                "version": "2.0",
                "timestamp": 0,
                "primitives": {
                    "prim0": {
                        "name": "prim0",
                        "level": 0,
                        "expr": {"type": "val", "value": 1},
                        "semantic_hash": "hash0",
                        "dependencies": [],
                        "usage_weight": 1.0,
                        "usage_count": 0,
                        "compression_ratio": 1.0,
                        "created_at": 0,
                    }
                },
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    manager = LibraryManager()
    assert "prim0" in manager.primitives
