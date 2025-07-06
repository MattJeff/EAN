"""Tests for epsilon-greedy ControllerAssembly."""
from __future__ import annotations

import numpy as np

from hx_core.controller import ControllerAssembly, PRIMITIVES


def test_converges_to_best_primitive():
    # Environment: rotate90 gives reward 1, others 0
    ctrl = ControllerAssembly(epsilon=0.2)
    grid = np.arange(4).reshape(2, 2)
    best_name = "rotate90"
    for _ in range(300):
        name, _ = ctrl.act(grid)
        reward = 1.0 if name == best_name else 0.0
        ctrl.update(reward)
    # Turn off exploration
    name, _ = ctrl.act(grid, epsilon=0.0)
    assert name == best_name


def test_save_load_roundtrip(tmp_path):
    ctrl = ControllerAssembly(epsilon=0.05)
    # Simulate one update
    grid = np.zeros((2, 2), dtype=int)
    name, _ = ctrl.act(grid)
    ctrl.update(0.3)
    p = tmp_path / "ctrl.json"
    ctrl.save(p)
    loaded = ControllerAssembly.load(p)
    assert loaded.epsilon == ctrl.epsilon
    assert loaded._values == ctrl._values
    assert loaded._counts == ctrl._counts
