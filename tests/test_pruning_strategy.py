"""Test CompetitiveEnergyPruning behaviour."""
from __future__ import annotations

import numpy as np

from hx_core import Assembly, Network, Neuron
from hx_core.strategies import CompetitiveEnergyPruning


def test_competitive_pruning():
    # Create assembly with low stability score
    n = Neuron(np.zeros(3))
    asm = Assembly("A", [n])
    asm.stability_score = 0.01
    asm.protection_counter = 0

    strategy = CompetitiveEnergyPruning(energy_cost=0.05, reward_factor=1.0)
    assert strategy.should_dissolve(asm) is True

    asm.protection_counter = 3
    # Protected, should not dissolve
    assert strategy.should_dissolve(asm) is False
