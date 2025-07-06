"""Test RotationHintConnector for generic NxN grids."""
import numpy as np

from hx_adapters import ArcAdapter
from hx_core.connectors import RotationHintConnector


def _neurons_by_xy(neurons, z_layer):
    return {
        (int(n.position[0]), int(n.position[1])): n
        for n in neurons
        if n.position[2] == z_layer
    }


def _check_connections(adapter, angle):
    net = adapter.network
    inputs = _neurons_by_xy(adapter.neurons, 0.0)
    outputs = _neurons_by_xy(adapter.neurons, 1.0)

    side = int(np.sqrt(len(inputs)))

    def rotate(x, y, n):
        if angle == 0:
            return x, y
        if angle == 90:
            return n - 1 - y, x
        if angle == 180:
            return n - 1 - x, n - 1 - y
        if angle == 270:
            return y, n - 1 - x
        raise ValueError

    for (x, y), n_in in inputs.items():
        rx, ry = rotate(x, y, side)
        target_neuron = outputs[(rx, ry)]
        assert target_neuron in net.connections.get(n_in, {}), f"Missing conn ({x},{y})->({rx},{ry}) for angle {angle}"


def test_rotation_connector_generic():
    for side in (3, 5):
        grid = np.arange(side * side).reshape(side, side) % 10
        for angle in (90, 180, 270):
            adapter = ArcAdapter()
            adapter.encode(grid)
            RotationHintConnector(angle=angle, weight=1.0).apply(network=adapter.network, neurons=adapter.neurons)
            _check_connections(adapter, angle)
