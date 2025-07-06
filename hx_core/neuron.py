"""Minimal, task-agnostic neuron primitive for HyperCubeX EAN.

This class is intentionally simple; behaviour will evolve through
emergent network dynamics rather than ad-hoc rules.

Public API is frozen to keep the rest of the system decoupled from
implementation details – only these attributes / methods should be
used externally.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

__all__ = ["Neuron"]


class Neuron:  # pylint: disable=too-few-public-methods
    """Energy-based point neuron.

    Parameters
    ----------
    position : np.ndarray | list[float]
        Spatial coordinates in arbitrary continuous space (3-D by default).
    threshold : float, default 1.0
        Firing threshold – when *energy* >= *threshold*, the neuron spikes.
    decay : float, default 0.02
        Exponential decay coefficient applied at every integration step.
    state_dim : int, default 1
        Length of the *state_vector* used for advanced dynamics.  This can be
        extended later but is kept minimal for now.
    """

    def __init__(
        self,
        position: np.ndarray | list[float],
        *,
        threshold: float = 1.0,
        decay: float = 0.02,
        state_dim: int = 1,
        refractory_period: int = 0,
    ) -> None:
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.threshold = float(threshold)
        self.decay = float(decay)

        # Runtime state -----------------------------------------------------
        self.energy: float = 0.0
        self.state_vector: np.ndarray = np.zeros(state_dim, dtype=float)
        self.last_input: float = 0.0
        self.last_output: float = 0.0
        # Refractory handling ---------------------------------------------
        self.refractory_period = int(refractory_period)
        self._refractory_counter: int = 0

    # ---------------------------------------------------------------------
    # Public API (stable)
    # ---------------------------------------------------------------------
    def receive(self, signal: float, weight: float = 1.0) -> None:
        """Accumulate weighted *signal* into *energy*."""
        # Ignore incoming signals while refractory
        if self._refractory_counter:
            return
        self.energy += signal * weight
        self.last_input = signal

    def integrate(self, dt: float = 1.0) -> None:
        """Time integration – apply passive decay.

        For now, a simple exponential decay is used:  *energy* ←
        *energy* · (1 − decay · dt).  Negative energy is clipped at 0.
        """
        # Skip integration while neuron is refractory
        if self._refractory_counter:
            self._refractory_counter -= 1
            return
        self.energy *= max(0.0, 1.0 - self.decay * dt)

    def fire(self) -> float:
        """Emit a spike if threshold reached and reset energy.

        Returns
        -------
        float
            The amount of energy released (0.0 if no spike).
        """
        if self.energy >= self.threshold:
            self.last_output = self.energy
            self.energy = 0.0
            # Enter refractory state
            if self.refractory_period:
                self._refractory_counter = self.refractory_period
            return self.last_output
        return 0.0

    def apply_plasticity(
        self,
        *,
        fired_pre: bool,
        fired_post: bool,
        weight: float,
        eta: float = 0.01,
    ) -> float:
        """Very simple Hebbian learning rule.

        Parameters
        ----------
        fired_pre : bool
            Whether the *presynaptic* neuron (self) fired this tick.
        fired_post : bool
            Whether the *postsynaptic* neuron fired this tick.
        weight : float
            Current synaptic weight.
        eta : float, default 0.01
            Learning rate.

        Returns
        -------
        float
            Updated (clipped) weight.
        """
        if fired_pre and fired_post:
            weight += eta
        # Optional decay else: pass
        return float(min(weight, 1.0))

    def get_statistics(self) -> Dict[str, Any]:
        """Return raw metrics useful to strategies/adapters."""
        return {
            "energy": self.energy,
            "last_input": self.last_input,
            "last_output": self.last_output,
        }

    def reset(self) -> None:
        """Re-initialise runtime state without reallocating buffers."""
        self.energy = 0.0
        self.last_input = 0.0
        self.last_output = 0.0
        self.state_vector.fill(0.0)

    # ------------------------------------------------------------------
    # Helper / magic methods
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Neuron(pos={self.position.tolist()}, energy={self.energy:.3f}, "
            f"threshold={self.threshold})"
        )
