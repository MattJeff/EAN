"""Inhibitory neuron primitive (negative outgoing influence)."""
from __future__ import annotations

import numpy as np

from hx_core.neuron import Neuron

__all__ = ["NeuronInhibitory"]


class NeuronInhibitory(Neuron):  # pylint: disable=too-few-public-methods
    """Neuron whose outgoing synapses are inhibitory (negative weight)."""

    def __init__(self, position: np.ndarray | list[float]):  # noqa: D401
        # Use same base params but lower threshold for quicker response
        super().__init__(position, threshold=0.5, decay=0.02)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @staticmethod
    def default_weight() -> float:  # noqa: D401
        """Return default inhibitory weight (negative)."""
        return -0.6
