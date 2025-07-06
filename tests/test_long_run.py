"""Ensure energy stays bounded over a long simulation."""
from __future__ import annotations

import numpy as np

from hx_core import Assembly, Network, Neuron


def test_energy_stability():
    neurons = [Neuron(np.random.randn(3)) for _ in range(10)]
    asm = Assembly("A", neurons)
    net = Network()
    net.add_assembly(asm)

    # Randomly connect neurons
    rng = np.random.default_rng(0)
    for src in neurons:
        for tgt in neurons:
            if src is tgt or rng.random() > 0.1:
                continue
            net.connect(src, tgt, weight=rng.random() * 0.5)

    # Stimulate one neuron
    neurons[0].receive(5.0)

    # Run 1000 ticks
    for _ in range(1000):
        net.step()

    # Assert no neuron energy exceeded 10x threshold (sanity)
    assert max(n.energy for n in neurons) < 10.0
