"""ARC adapter – bridges a 2-D colour grid with the HyperCubeX core.

Design goals:
1. Keep the logic linear and non-smart: strictly encoding/decoding.
2. Never import private internals; rely only on public API of `hx_core`.
3. Stateless after encode/decode except for helper attributes (shape, neurons).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from hx_core import Assembly, Network, Neuron

__all__: List[str] = ["ArcAdapter"]


class ArcAdapter:  # pylint: disable=too-few-public-methods
    """Adapter for Abstraction & Reasoning Corpus (ARC) grids."""

    def __init__(self, network: Network | None = None) -> None:  # noqa: D401
        self.network: Network = network or Network()
        self.shape: Tuple[int, int] | None = None
        self._input_neurons: List[Neuron] = []
        self._output_neurons: List[Neuron] = []
        self._neurons: List[Neuron] = []  # concat of both for connectors

    # ------------------------------------------------------------------
    # Encoding – grid  → network
    # ------------------------------------------------------------------
    def encode(self, grid: np.ndarray) -> None:  # noqa: D401
        """Convert an integer colour grid into neuron energies.

        Each grid cell gets a dedicated neuron placed at (x, y, 0).  The
        neuron's *energy* is initialised to colour / 9.  No connections are
        created – higher-level discovery mechanisms will do that.
        """
        if grid.ndim != 2:
            raise ValueError("ARC grid must be 2-D")

        self.shape = grid.shape
        self._input_neurons.clear()
        self._output_neurons.clear()
        self._neurons.clear()

        h, w = grid.shape
        in_neurons: List[Neuron] = []
        out_neurons: List[Neuron] = []
        for y in range(h):
            for x in range(w):
                colour = float(grid[y, x])
                # Input neuron with higher initial energy
                n_in = Neuron(position=np.array([x, y, 0.0]), threshold=0.5, decay=0.0)
                n_in.energy = colour  # keep original colour value (0-9)
                in_neurons.append(n_in)

                # Output neuron starts near-silent (energy=0.21) to survive pruning
                n_out = Neuron(position=np.array([x, y, 1.0]), threshold=0.3, decay=0.0)
                n_out.energy = 0.21
                out_neurons.append(n_out)
        self._input_neurons = in_neurons
        self._output_neurons = out_neurons
        self._neurons = in_neurons + out_neurons

        self.network.add_assembly(Assembly("ARC_INPUT", in_neurons))
        self.network.add_assembly(Assembly("ARC_OUTPUT", out_neurons))

    # ------------------------------------------------------------------
    # Decoding – network → grid
    # ------------------------------------------------------------------
    def decode(self) -> np.ndarray:  # noqa: D401
        """Read neuron energies back into a binary grid (0/1)."""
        if self.shape is None:
            raise RuntimeError("Must call encode() first")

        h, w = self.shape
        out = np.zeros((h, w), dtype=int)
        idx = 0
        for y in range(h):
            for x in range(w):
                n = self._output_neurons[idx]
                out[y, x] = 1 if n.last_output > 0 else 0
                idx += 1
        return out

    # ------------------------------------------------------------------
    # Runtime helper
    # ------------------------------------------------------------------
    def recharge(self, grid: np.ndarray) -> None:  # noqa: D401
        """Re-inject energies for a new episode without recreating neurons."""
        if self.shape is None:
            raise ValueError("Adapter must be encoded before recharge")
        assert grid.shape == self.shape, "Shape mismatch in recharge"
        for n, val in zip(self._input_neurons, grid.flat):
            n.energy = float(val)
            n.last_input = 0.0
            n.last_output = 0.0
        for n in self._output_neurons:
            n.energy = 0.21
            n.last_input = 0.0
            n.last_output = 0.0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def neurons(self) -> List[Neuron]:  # noqa: D401
        """Return all neurons (input + output)."""
        return self._neurons

    @property
    def input_neurons(self) -> List[Neuron]:  # noqa: D401
        return self._input_neurons

    @property
    def output_neurons(self) -> List[Neuron]:  # noqa: D401
        return self._output_neurons
