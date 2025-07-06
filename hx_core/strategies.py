"""Pruning strategy interface and a simple baseline implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

# Forward import avoided (no heavy dep) – we only need the protocol.
class _AssemblyProtocol(Protocol):
    def get_statistics(self) -> dict:  # noqa: D401
        ...


class PruningStrategy(ABC):
    """Decide whether an assembly should be dissolved."""

    @abstractmethod
    def should_dissolve(self, assembly: _AssemblyProtocol) -> bool:  # noqa: D401
        """Return *True* to signal that *assembly* must be removed."""
        raise NotImplementedError


class SimpleThresholdPruning(PruningStrategy):
    """Dissolve assemblies whose mean energy falls below *threshold*."""

    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = float(threshold)

    def should_dissolve(self, assembly: _AssemblyProtocol) -> bool:  # noqa: D401
        stats = assembly.get_statistics()
        return bool(stats.get("mean_energy", 0.0) < self.threshold)


class CompetitiveEnergyPruning(PruningStrategy):
    """Prune assemblies with low *stability_score* vs energy cost.

    Parameters
    ----------
    energy_cost : float
        Baseline cost per tick to keep an assembly alive.
    reward_factor : float
        Multiplier applied to stability_score when comparing to cost.
    protection_window : int
        Minimum ticks after a successful spike before dissolution allowed.
    """

    def __init__(
        self,
        *,
        energy_cost: float = 0.05,
        reward_factor: float = 1.0,
        protection_window: int = 5,
    ) -> None:
        self.energy_cost = float(energy_cost)
        self.reward_factor = float(reward_factor)
        self.protection_window = protection_window

    def should_dissolve(self, assembly: _AssemblyProtocol) -> bool:  # noqa: D401
        stats = assembly.get_statistics()
        if stats.get("protection_counter", 0) > 0:
            return False  # protected after recent success
        stability = stats.get("stability_score", 0.0)
        return stability * self.reward_factor < self.energy_cost
