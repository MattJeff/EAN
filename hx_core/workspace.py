"""Global Workspace / Arbiter for HyperCubeX-EAN.

This lightweight component collects *votes* (candidate output grids) coming
from multiple *assemblies* or agents and returns the selected consensus.

Current implementation supports a simple *winner-takes-all* strategy based on
scalar confidence values, with deterministic tie-breaking (first submitter).
Future work can plug more elaborate aggregation methods such as majority
voting per cell, energy-weighted blending, etc., behind the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

from .assemblies import AssemblyStore  # noqa: E401

__all__: List[str] = ["GlobalWorkspace", "Vote"]


@dataclass(slots=True)
class Vote:  # pylint: disable=too-few-public-methods
    """A single candidate output proposed by an *assembly*."""

    source: Any  # identifier (int, str, Assembly…)
    grid: np.ndarray  # 2-D integer array
    confidence: float  # higher = better

    def __post_init__(self) -> None:  # noqa: D401
        if self.grid.ndim != 2:
            raise ValueError("Vote grid must be 2-D")
        if not np.issubdtype(self.grid.dtype, np.integer):
            raise TypeError("Vote grid must contain integers")
        self.confidence = float(self.confidence)


class GlobalWorkspace:  # pylint: disable=too-few-public-methods
    """Aggregate candidate grids and select a consensus output."""

    def __init__(self, grid_shape: Tuple[int, int]):  # noqa: D401
        self.grid_shape = tuple(grid_shape)
        self._votes: List[Vote] = []
        self._latest_winner: Vote | None = None

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    # ------------------------------------------------------------------
    def submit(self, source: Any, grid: np.ndarray, confidence: float | int) -> None:  # noqa: D401
        """Register a candidate *grid* from *source* with *confidence*."""
        grid = np.asarray(grid, dtype=int)
        if grid.shape != self.grid_shape:
            raise ValueError("All votes must share the same grid shape")
        self._votes.append(Vote(source, grid, float(confidence)))

    def decide(self) -> np.ndarray:  # noqa: D401
        """Return consensus grid (highest confidence)."""
        if not self._votes:
            raise RuntimeError("No votes submitted to workspace")
        self._latest_winner = max(self._votes, key=lambda v: v.confidence)
        return np.array(self._latest_winner.grid, copy=True)

    def votes(self) -> List[Vote]:  # noqa: D401
        """Return *immutable* snapshot of current votes list."""
        return list(self._votes)

    def winner(self) -> Vote | None:  # noqa: D401
        """Return the most recent winning vote (if *decide* was called)."""
        return self._latest_winner

    def reset(self) -> None:  # noqa: D401
        """Clear all stored votes and winner cache."""
        self._votes.clear()
        self._latest_winner = None

    # ------------------------------------------------------------------
    # Memory injection --------------------------------------------------
    # ------------------------------------------------------------------
    def inject_assemblies(self, store: AssemblyStore) -> None:  # noqa: D401
        """Pre-populate workspace with votes from an ``AssemblyStore``.

        Each assembly’s binary mask is cast to an integer grid (cells==1) and
        submitted as a vote whose *confidence* equals ``assembly.strength``.
        """
        for asm in store:
            mask = asm.to_mask(self.grid_shape)
            grid = mask.astype(int)
            self.submit(source=asm, grid=grid, confidence=asm.strength)

    # ------------------------------------------------------------------
    # Statistics helpers ------------------------------------------------
    # ------------------------------------------------------------------
    def confidence_histogram(self, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:  # noqa: D401
        """Return histogram (counts, bin_edges) of vote confidences."""
        if not self._votes:
            return np.array([], dtype=int), np.array([], dtype=float)
        confidences = np.array([v.confidence for v in self._votes])
        return np.histogram(confidences, bins=bins)

    # Convenience -------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self._votes)

    def __iter__(self):  # noqa: D401
        return iter(self._votes)
