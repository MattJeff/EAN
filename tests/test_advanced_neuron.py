"""Basic checks for NeuronAdvanced with STDP."""
from __future__ import annotations

import numpy as np

from hx_bio import NeuronAdvanced


def test_fire_and_spike_count():
    n = NeuronAdvanced(np.zeros(3), threshold=0.5)
    n.receive(1.0)
    # Integrate then fire
    n.integrate()
    out = n.fire(current_time=0.0)
    assert out > 0
    assert n.get_statistics()["n_spikes"] == 1


def test_stdp_weight_update():
    n = NeuronAdvanced(np.zeros(3))
    initial_w = 0.5
    # Both pre and post spiked same tick -> potentiation
    new_w = n.apply_plasticity(
        fired_pre=True,
        fired_post=True,
        weight=initial_w,
    )
    assert new_w >= initial_w
