"""Network orchestrates assemblies and simulation cycles."""
from __future__ import annotations

from typing import Dict, List

from .assembly import Assembly
from .strategies import PruningStrategy, SimpleThresholdPruning
from .neuron import Neuron

__all__ = ["Network"]


class Network:  # pylint: disable=too-few-public-methods
    """Container managing assemblies + pruning cycle."""

    def __init__(
        self,
        *,
        pruning_strategy: PruningStrategy | None = None,
        backend: str | None = None,
        noise_std: float = 0.0,
    ) -> None:  # noqa: D401
        from hx_core.backend import select_backend

        self.backend: str = select_backend() if backend is None else backend
        self.pruning_strategy = pruning_strategy or SimpleThresholdPruning()
        self.assemblies: Dict[str, Assembly] = {}
        self.connections: Dict[Neuron, Dict[Neuron, float]] = {}
        self.tick: int = 0
        self.noise_std = float(noise_std)

    # ------------------------------------------------------------------
    # Assembly management
    # ------------------------------------------------------------------
    def add_assembly(self, assembly: Assembly) -> None:
        self.assemblies[assembly.id] = assembly

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self, source: Neuron, target: Neuron, weight: float = 1.0) -> None:
        """Create (or update) a directed connection *source → target*."""
        self.connections.setdefault(source, {})[target] = weight

    def disconnect(self, source: Neuron, target: Neuron) -> None:
        """Remove a connection if it exists."""
        if target in self.connections.get(source, {}):
            del self.connections[source][target]
            if not self.connections[source]:
                del self.connections[source]

    # ------------------------------------------------------------------
    # Simulation loop helpers
    # ------------------------------------------------------------------
    def step(self, dt: float = 1.0) -> None:
        """Advance the whole network by one time step and propagate spikes."""
        self.tick += 1
        spikes: Dict[Neuron, float] = {}
        for asm in list(self.assemblies.values()):
            asm.integrate(dt, backend=self.backend)
            spikes.update(asm.fire(backend=self.backend))
            if self.pruning_strategy.should_dissolve(asm):
                del self.assemblies[asm.id]

        # Propagate spikes to targets
        for src, signal in spikes.items():
            for tgt, weight in list(self.connections.get(src, {}).items()):
                noisy_signal = signal
                if self.noise_std:
                    import random, math
                    noisy_signal += random.gauss(0.0, self.noise_std)
                tgt.receive(noisy_signal, weight)
                # Hebbian plasticity update
                new_w = src.apply_plasticity(
                    fired_pre=True,
                    fired_post=tgt in spikes,
                    weight=weight,
                )
                self.connections[src][tgt] = new_w

    # ------------------------------------------------------------------
    # Metrics / debug helpers
    # ------------------------------------------------------------------
    def get_statistics(self) -> List[dict]:
        return [asm.get_statistics() for asm in self.assemblies.values()]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Network(tick={self.tick}, n_assemblies={len(self.assemblies)})"
