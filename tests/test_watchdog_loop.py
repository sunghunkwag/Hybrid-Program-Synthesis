import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import rsi_watchdog_loop


def test_loop_respects_caps(tmp_path):
    with pytest.raises(ValueError):
        rsi_watchdog_loop.run_loop(
            rounds=rsi_watchdog_loop.MAX_ROUNDS + 1,
            trials=1,
            output_root=str(tmp_path),
            synth_timeout=0.1,
            watchdog_timeout=0.1,
        )


def test_watchdog_timeout_enforced():
    result = rsi_watchdog_loop.watchdog_run_code("while True:\n    pass\n", timeout=0.1)
    assert result.get("killed") is True


def test_persistence_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rsi_meta_weights.json").write_text(json.dumps({"seed": 1}))
    (tmp_path / "rsi_primitive_registry.json").write_text(json.dumps({"version": "2.0", "primitives": {}}))
    run_dir = rsi_watchdog_loop.run_loop(
        rounds=1,
        trials=1,
        output_root=str(tmp_path / "runs"),
        synth_timeout=0.1,
        watchdog_timeout=0.2,
    )
    assert os.path.isdir(run_dir)
    assert os.path.isfile(os.path.join(run_dir, "meta_weights_before_after.json"))
    assert os.path.isfile(os.path.join(run_dir, "registry_before_after.json"))
