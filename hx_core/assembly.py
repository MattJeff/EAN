"""Lightweight container for a group of neurons.

Assemblies expose the minimal surface needed by pruning/scheduler layers.
No task-specific logic is allowed here.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np

from .neuron import Neuron

__all__ = ["Assembly"]


class Assembly:  # pylint: disable=too-few-public-methods
    """A collection of neurons acting as a functional unit."""

    def __init__(self, id: str, neurons: List[Neuron] | None = None) -> None:  # noqa: D401
        self.id = id
        self.neurons: List[Neuron] = neurons[:] if neurons else []

        # Metrics for pruning strategies
        self.stability_score: float = 0.0  # EMA of mean_energy
        self.protection_counter: int = 0
        self._alpha: float = 0.05  # smoothing factor
        self._protection_duration: int = 5

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------
    def add(self, neuron: Neuron) -> None:
        if neuron not in self.neurons:
            self.neurons.append(neuron)
            if hasattr(neuron, "join_assembly"):
                neuron.join_assembly(self.id)  # type: ignore[arg-type]

    def remove(self, neuron: Neuron) -> None:
        if neuron in self.neurons:
            self.neurons.remove(neuron)
            if hasattr(neuron, "leave_assembly"):
                neuron.leave_assembly()  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    def integrate(self, dt: float = 1.0, backend: str | None = None) -> None:  # noqa: D401
        from hx_core.backend import get_backend

        backend_cls = get_backend(backend)
        backend_cls.integrate(self, dt)
        mean_energy = float(np.mean([n.energy for n in self.neurons])) if self.neurons else 0.0
        self.stability_score = (1 - self._alpha) * self.stability_score + self._alpha * mean_energy

    def fire(self, backend: str | None = None) -> Dict[Neuron, float]:
        """Trigger potential spikes for each neuron and return outputs."""
        from hx_core.backend import get_backend

        backend_cls = get_backend(backend)
        spikes = backend_cls.fire(self)
        outputs: Dict[Neuron, float] = {}
        for n, spike in spikes.items():
            if spike:
                outputs[n] = spike
        if spikes:
            self.protection_counter = self._protection_duration
        elif self.protection_counter:
            self.protection_counter -= 1
        return outputs

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_statistics(self) -> dict:
        energies = [n.energy for n in self.neurons]
        return {
            "size": len(self.neurons),
            "mean_energy": float(np.mean(energies)) if energies else 0.0,
            "stability_score": self.stability_score,
            "protection_counter": self.protection_counter,
        }

    # ------------------------------------------------------------------
    # Magic / representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"Assembly(id={self.id!r}, size={len(self.neurons)})"
