"""Unit tests for SpikeRateController and inhibitory connectors."""
from __future__ import annotations

import numpy as np

from hx_bio.homeostasis import SpikeRateController
from hx_bio.inhibitory import NeuronInhibitory
from hx_core.connectors import InhibitoryConnector
from hx_core.network import Network
from hx_core.neuron import Neuron
from hx_core.assembly import Assembly


def test_homeostasis_adjusts_threshold():
    neurons = [Neuron([0, 0, 0], threshold=1.0, decay=0.0) for _ in range(10)]
    fired = set(neurons[:5])  # 50% spiked
    ctrl = SpikeRateController(target_rate=0.1, lr=0.5)
    old_thr = neurons[0].threshold
    # Run multiple updates so EMA reflects high rate
    for _ in range(5):
        ctrl.update(neurons=neurons, fired=fired)
    assert neurons[0].threshold > old_thr, "Threshold should increase when rate too high"


def test_inhibitory_connector_adds_neurons():
    net = Network()
    base_neurons = [Neuron([0, 0, 0]) for _ in range(4)]
    net.add_assembly(Assembly("BASE", base_neurons))

    conn = InhibitoryConnector(k=1, fan_out=2)
    conn.apply(network=net, neurons=base_neurons)

    # One more neuron added
    assert len(base_neurons) == 5
    inhib = [n for n in base_neurons if isinstance(n, NeuronInhibitory)]
    assert len(inhib) == 1
    # Check at least one negative connection
    negatives = [w for src in net.connections for w in net.connections[src].values() if w < 0]
    assert negatives, "Should have negative inhibitory weights"
