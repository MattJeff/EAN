"""Test MirrorHintConnector for generic NxN grids."""
import numpy as np

from hx_adapters import ArcAdapter
from hx_core.connectors import MirrorHintConnector


def _neurons_by_xy(neurons, z_layer):
    return {
        (int(n.position[0]), int(n.position[1])): n
        for n in neurons
        if n.position[2] == z_layer
    }


def _check_mirror(adapter, axis):
    net = adapter.network
    inputs = _neurons_by_xy(adapter.neurons, 0.0)
    outputs = _neurons_by_xy(adapter.neurons, 1.0)

    side = int(np.sqrt(len(inputs)))

    def mirror(x, y, n):
        if axis == "vertical":
            return n - 1 - x, y
        return x, n - 1 - y  # horizontal

    for (x, y), n_in in inputs.items():
        mx, my = mirror(x, y, side)
        target_neuron = outputs[(mx, my)]
        assert target_neuron in net.connections.get(n_in, {}), f"Missing conn ({x},{y})->({mx},{my}) axis {axis}"


def test_mirror_connector_generic():
    for side in (3, 5):
        grid = np.arange(side * side).reshape(side, side) % 10
        for axis in ("vertical", "horizontal"):
            adapter = ArcAdapter()
            adapter.encode(grid)
            MirrorHintConnector(axis=axis, weight=1.0).apply(network=adapter.network, neurons=adapter.neurons)
            _check_mirror(adapter, axis)
