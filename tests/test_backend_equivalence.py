"""Compare CPU vs Torch backend on a simple Assembly."""
import os

import numpy as np
import pytest

from hx_core.backend import get_backend
from hx_core.neuron import Neuron
from hx_core.assembly import Assembly


def _make_assembly(n=16):
    neurons = [Neuron([0, 0, 0], threshold=0.5, decay=0.1) for _ in range(n)]
    # random initial energies
    for nd in neurons:
        nd.energy = np.random.rand()
    return Assembly("A", neurons)


def test_cpu_vs_cpu():
    asm = _make_assembly()
    cpu = get_backend("cpu")
    cpu.integrate(asm)
    out = cpu.fire(asm)
    assert isinstance(out, dict)


@pytest.mark.skipif("torch" not in get_backend.__globals__["_BACKENDS"], reason="Torch not available")
def test_cpu_vs_torch_equiv():
    asm1 = _make_assembly()
    asm2 = _make_assembly()
    # clone energies
    for a, b in zip(asm1.neurons, asm2.neurons):
        b.energy = a.energy

    cpu = get_backend("cpu")
    torch_backend = get_backend("torch")

    cpu.integrate(asm1)
    torch_backend.integrate(asm2)

    np.testing.assert_allclose(
        [n.energy for n in asm1.neurons],
        [n.energy for n in asm2.neurons],
        rtol=1e-5,
    )
