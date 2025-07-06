"""Simple monitoring utilities for HyperCubeX."""
from __future__ import annotations

from pathlib import Path
from typing import List
import csv

from hx_core import Network

from .advanced import UnifiedLogger, SpikeEnergyVisualizer

__all__: List[str] = [
    "SimpleCSVLogger",
    "UnifiedLogger",
    "SpikeEnergyVisualizer",
]


class SimpleCSVLogger:
    """Log high-level network stats to a CSV file each tick."""

    def __init__(self, filepath: str | Path) -> None:  # noqa: D401
        self.filepath = Path(filepath)
        # Write header
        with self.filepath.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick",
                "n_assemblies",
                "mean_energy_global",
                "reward",
            ])

    def log(self, network: Network, *, reward: float | None = None) -> None:  # noqa: D401
        from statistics import mean

        energies: List[float] = []
        for asm in network.assemblies.values():
            energies.extend([n.energy for n in asm.neurons])

        with self.filepath.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    network.tick,
                    len(network.assemblies),
                    mean(energies) if energies else 0.0,
                    reward if reward is not None else "",
                ]
            )
