"""Test spike propagation through network connections."""
from __future__ import annotations

import numpy as np

from hx_core import Assembly, Network, Neuron


def test_signal_propagation():
    n1 = Neuron(np.zeros(3))
    n2 = Neuron(np.zeros(3))

    # Pre-charge n1 so it will fire
    n1.receive(1.5)

    asm = Assembly("A1", [n1, n2])
    net = Network()
    net.add_assembly(asm)
    net.connect(n1, n2, weight=0.5)

    net.step()

    assert n2.energy > 0.0, "Energy should have been received from n1"
