"""Tests for GlobalWorkspace aggregator."""
from __future__ import annotations

import numpy as np
import pytest

from hx_core.workspace import GlobalWorkspace


def test_selects_highest_confidence():
    ws = GlobalWorkspace(grid_shape=(2, 2))
    g1 = np.zeros((2, 2), dtype=int)
    g2 = np.ones((2, 2), dtype=int)
    ws.submit("a", g1, confidence=0.3)
    ws.submit("b", g2, confidence=0.8)
    out = ws.decide()
    assert np.array_equal(out, g2)


def test_tie_break_first_submitted():
    ws = GlobalWorkspace((1, 1))
    g1 = np.array([[1]])
    g2 = np.array([[2]])
    ws.submit("first", g1, 0.5)
    ws.submit("second", g2, 0.5)
    out = ws.decide()
    # max with key keeps first occurrence on tie
    assert np.array_equal(out, g1)


def test_reset():
    ws = GlobalWorkspace((2, 2))
    ws.submit("x", np.zeros((2, 2)), 0.1)
    ws.reset()
    assert len(ws) == 0
    with pytest.raises(RuntimeError):
        ws.decide()


def test_histogram():
    ws = GlobalWorkspace((1, 1))
    for c in [0.1, 0.2, 0.4, 0.4]:
        ws.submit("s", np.array([[0]]), c)
    counts, edges = ws.confidence_histogram(bins=4)
    assert counts.sum() == 4
    assert len(edges) == 5
