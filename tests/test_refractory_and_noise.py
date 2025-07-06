"""Test new robustness features: synaptic noise & refractory period."""
import numpy as np
from hx_core.neuron import Neuron
from hx_core.network import Network
from hx_core.assembly import Assembly


def test_refractory():
    n = Neuron([0, 0, 0], threshold=0.5, refractory_period=2)
    n.energy = 0.6
    # First fire
    out1 = n.fire()
    assert out1 > 0
    # During refractory, integrate should do nothing
    n.receive(1.0)
    n.integrate()
    assert n.energy == 0.0  # still refractory, no accumulation
    n.integrate()  # counter 1 -> 0
    # Now integrate again
    n.receive(0.6)
    n.integrate()
    assert n.energy > 0.0


def test_synaptic_noise():
    net = Network(noise_std=0.1)
    pre = Neuron([0, 0, 0])
    post = Neuron([1, 0, 0])
    asm = Assembly("A", [pre, post])
    net.add_assembly(asm)
    net.connect(pre, post, weight=1.0)

    pre.energy = 1.1  # will spike
    net.step()
    # post energy should be close to 1 with noise
    assert abs(post.energy) < 2.0
