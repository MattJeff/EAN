"""Unit tests for the minimal `Neuron` primitive.

Run with `pytest`.
"""
from __future__ import annotations

import numpy as np

from hx_core import Neuron


def test_basic_fire():
    n = Neuron(position=np.zeros(3), threshold=1.0)
    n.receive(1.5)
    n.integrate()
    out = n.fire()

    assert out > 0.0
    assert n.energy == 0.0
