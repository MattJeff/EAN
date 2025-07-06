"""Quick network smoke test covering assembly integration + pruning."""
from __future__ import annotations

import numpy as np

from hx_core import Assembly, Network, Neuron


def test_network_prune():
    # Create two neurons with low energy so assembly gets pruned
    n1 = Neuron(np.zeros(3))
    n2 = Neuron(np.zeros(3))

    asm = Assembly("A1", [n1, n2])
    net = Network()
    net.add_assembly(asm)

    # After one step, mean energy ~ 0 => should be pruned
    net.step()
    assert len(net.assemblies) == 0
