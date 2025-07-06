"""Scheduler interface for running simulations."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .network import Network

__all__ = ["Scheduler", "SimpleScheduler"]


class Scheduler(ABC):
    """Control the timing of network updates."""

    def __init__(self, network: Network) -> None:  # noqa: D401
        self.network = network

    @abstractmethod
    def step(self, dt: float = 1.0) -> None:  # noqa: D401
        ...

    # Simple helper to run multiple steps
    def run(self, steps: int | None = None, *, dt: float = 1.0, max_steps: int | None = None) -> None:  # noqa: D401
        """Run the simulation.

        If *steps* is *None*, run indefinitely until *max_steps* ticks are
        reached (useful for fuzz-testing without infinite loops).
        """
        tick = 0
        while True:
            if steps is not None and tick >= steps:
                break
            if max_steps is not None and tick >= max_steps:
                break
            self.step(dt)
            tick += 1


class SimpleScheduler(Scheduler):
    """Call `network.step` directly each tick."""

    def step(self, dt: float = 1.0) -> None:  # noqa: D401
        self.network.step(dt)
