"""Test Hebbian weight increase when pre and post fire together."""
from __future__ import annotations

import numpy as np

from hx_core import Assembly, Network, Neuron


def test_hebbian_increase():
    n1 = Neuron(np.zeros(3))
    n2 = Neuron(np.zeros(3))

    # charge both neurons so they fire same tick
    n1.receive(1.5)
    n2.receive(1.5)

    net = Network()
    net.add_assembly(Assembly("A", [n1, n2]))

    net.connect(n1, n2, weight=0.3)
    old_w = net.connections[n1][n2]

    net.step()

    new_w = net.connections[n1][n2]
    assert new_w > old_w, "Weight should increase via Hebbian learning"
