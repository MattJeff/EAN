"""Pluggable connection initialisation strategies for HyperCubeX.

A *connector* inspects the freshly encoded neurons and adds synaptic
connections to the network before any learning/optimiser step occurs.
By keeping this logic separate from adapters and optimisers we respect
modularity and allow multiple strategies (random, geometric, distance-based
etc.) to coexist.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

from .neuron import Neuron
from .network import Network
from .assembly import Assembly

from hx_bio.inhibitory import NeuronInhibitory

__all__ = [
    "ConnectionInitializer",
    "RandomConnector",
    "InhibitoryConnector",
    "RotationHintConnector",
    "MirrorHintConnector",
]


class ConnectionInitializer(ABC):  # pylint: disable=too-few-public-methods
    """Strategy interface – create initial synapses."""

    @abstractmethod
    def apply(self, *, network: Network, neurons: List[Neuron]) -> None:  # noqa: D401
        """Populate *network.connections* as desired."""
        raise NotImplementedError


class RandomConnector(ConnectionInitializer):
    """Create a fixed number of random directed connections."""

    def __init__(self, n_connections: int = 8, weight: float = 0.5):  # noqa: D401
        self.n_connections = int(n_connections)
        self.weight = float(weight)

    # ------------------------------------------------------------------
    def apply(self, *, network: Network, neurons: List[Neuron]) -> None:  # noqa: D401
        if len(neurons) < 2:
            return
        for _ in range(self.n_connections):
            src, tgt = random.sample(neurons, 2)
            network.connect(src, tgt, weight=self.weight)


class MirrorHintConnector(ConnectionInitializer):  # pylint: disable=too-few-public-methods
    """Create deterministic connections for mirror symmetry.

    axis="vertical" → flip left\/right (x axis).  axis="horizontal" → flip top\/bottom (y axis).
    Works for square grids encoded by ArcAdapter.
    """

    def __init__(self, axis: str = "vertical", weight: float = 1.0) -> None:  # noqa: D401
        if axis not in {"vertical", "horizontal"}:
            raise ValueError("axis must be 'vertical' or 'horizontal'")
        self.axis = axis
        self.weight = float(weight)

    def _mirror(self, x: int, y: int, n: int) -> tuple[int, int]:  # noqa: D401
        if self.axis == "vertical":  # left\/right
            return n - 1 - x, y
        # horizontal
        return x, n - 1 - y

    # ------------------------------------------------------------------
    def apply(self, *, network: "Network", neurons: List["Neuron"]) -> None:  # noqa: D401
        inputs = [n for n in neurons if n.position[2] == 0.0]
        outputs = [n for n in neurons if n.position[2] == 1.0]
        if not inputs or not outputs:
            return
        out_map = {(int(o.position[0]), int(o.position[1])): o for o in outputs}
        n_side = int(max(max(n.position[0] for n in inputs), max(n.position[1] for n in inputs))) + 1
        for inp in inputs:
            x, y = int(inp.position[0]), int(inp.position[1])
            mx, my = self._mirror(x, y, n_side)
            tgt = out_map.get((mx, my))
            if tgt is not None:
                network.connect(inp, tgt, weight=self.weight)


class RotationHintConnector(ConnectionInitializer):  # pylint: disable=too-few-public-methods
    """Create deterministic connections from input neurons to rotated output neurons.

    Assumes neurons are laid out by :pyclass:`hx_adapters.ArcAdapter` such that
    input layer has z=0 and output layer z=1. Only works for square grids.
    """

    def __init__(self, angle: int = 90, weight: float = 1.0) -> None:  # noqa: D401
        self.angle = angle % 360
        self.weight = float(weight)

    # ------------------------------------------------------------------
    def _rotate(self, x: int, y: int, n: int) -> tuple[int, int]:
        if self.angle == 0:
            return x, y
        if self.angle == 90:
            return n - 1 - y, x
        if self.angle == 180:
            return n - 1 - x, n - 1 - y
        if self.angle == 270:
            return y, n - 1 - x
        raise ValueError(f"Unsupported angle {self.angle}")

    # ------------------------------------------------------------------
    def apply(self, *, network: Network, neurons: List[Neuron]) -> None:  # noqa: D401
        inputs = [n for n in neurons if n.position[2] == 0.0]
        outputs = [n for n in neurons if n.position[2] == 1.0]
        if not inputs or not outputs:
            return
        # Build lookup for outputs by (x,y)
        out_map = {(int(o.position[0]), int(o.position[1])): o for o in outputs}
        # Infer grid size (assume square)
        n_side = int(max(max(n.position[0] for n in inputs), max(n.position[1] for n in inputs))) + 1
        for inp in inputs:
            x, y = int(inp.position[0]), int(inp.position[1])
            rx, ry = self._rotate(x, y, n_side)
            tgt = out_map.get((rx, ry))
            if tgt is not None:
                network.connect(inp, tgt, weight=self.weight)


class InhibitoryConnector(ConnectionInitializer):  # pylint: disable=too-few-public-methods(ConnectionInitializer):  # pylint: disable=too-few-public-methods
    """Insert *k* inhibitory neurons and connect them to the existing graph."""

    def __init__(
        self,
        k: int = 2,
        fan_out: int | None = None,
        weight: float | None = None,
    ) -> None:  # noqa: D401
        self.k = int(k)
        self.fan_out = fan_out  # if None: connect to all
        self.weight = weight  # if None: NeuronInhibitory.default_weight()

    # ------------------------------------------------------------------
    def apply(self, *, network: Network, neurons: List[Neuron]) -> None:  # noqa: D401
        if self.k <= 0:
            return
        for _ in range(self.k):
            pos = [0.0, 0.0, 0.0]  # could randomise later
            inhib = NeuronInhibitory(position=pos)
            neurons.append(inhib)
            network.add_assembly(Assembly("INHIB", [inhib]))  # type: ignore[name-defined]

            w = self.weight if self.weight is not None else NeuronInhibitory.default_weight()
            targets = neurons if self.fan_out is None else random.sample(neurons, min(self.fan_out, len(neurons)))
            for tgt in targets:
                if tgt is inhib:
                    continue
                network.connect(inhib, tgt, weight=w)
