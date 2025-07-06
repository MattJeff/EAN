"""Persistent *assemblies* registry for HyperCubeX-EAN.

In la théorie des *Assemblies* (Litwin & Buzsáki, 2018), a memory is captured
by a distributed activation pattern of neurones.  Here we give a lightweight
implementation whose sole responsibility is to *store* such patterns between
runs so they can be re-used or analysed.

This module **does not** try to enforce a particular learning rule – that is
handled elsewhere (optimizers, schedulers, teachers…).  It just offers:

    • `Assembly` – minimal data structure (pattern + strength).
    • `AssemblyStore` – append-only collection with size cap and JSON
      persistence.

The grid pattern is kept as a list of (y, x) coordinates for compactness and
backend-agnostic storage.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

__all__ = ["Assembly", "AssemblyStore"]

Coord = Tuple[int, int]


@dataclass(slots=True)
class Assembly:  # pylint: disable=too-few-public-methods
    """Simple immutable representation of a neuronal assembly."""

    coords: List[Coord]  # list of active neuron coordinates (y, x)
    strength: float = 1.0  # usefulness / salience score
    meta: dict | None = None  # optional arbitrary metadata

    # ------------------------------------------------------------------
    @classmethod
    def from_mask(cls, mask: np.ndarray, *, strength: float = 1.0) -> "Assembly":  # noqa: D401
        """Create from a boolean mask (True = active)."""
        if mask.ndim != 2:
            raise ValueError("Mask must be 2-D")
        coords = [(int(y), int(x)) for y, x in zip(*np.where(mask))]
        return cls(coords=coords, strength=float(strength))

    def update_strength(self, reward: float, alpha: float = 0.1) -> None:  # noqa: D401
        """Update salience using EMA rule: new = (1-α)*old + α*reward."""
        self.strength = (1 - alpha) * self.strength + alpha * float(reward)

    def to_mask(self, shape: Tuple[int, int]) -> np.ndarray:  # noqa: D401
        mask = np.zeros(shape, dtype=bool)
        for y, x in self.coords:
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                mask[y, x] = True
        return mask


class AssemblyStore:  # pylint: disable=too-few-public-methods
    """Fixed-capacity list of *Assembly* objects with JSON persistence."""

    def __init__(self, max_size: int = 1000):  # noqa: D401
        self.max_size = int(max_size)
        self._assemblies: List[Assembly] = []

    # ------------------------------------------------------------------
    def add(self, assembly: Assembly) -> None:  # noqa: D401
        self._assemblies.append(assembly)
        # Keep list sorted by descending strength for quick pruning
        self._assemblies.sort(key=lambda a: a.strength, reverse=True)
        # Simple memory management
        if len(self._assemblies) > self.max_size:
            self._assemblies = self._assemblies[: self.max_size]

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:  # noqa: D401
        path = Path(path)
        data = {
            "max_size": self.max_size,
            "assemblies": [asdict(a) for a in self._assemblies],
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "AssemblyStore":  # noqa: D401
        data = json.loads(Path(path).read_text())
        store = cls(max_size=int(data.get("max_size", 1000)))
        for a in data.get("assemblies", []):
            store._assemblies.append(Assembly(coords=[tuple(c) for c in a["coords"]], strength=float(a["strength"]), meta=a.get("meta")))
        return store

    # Convenience -------------------------------------------------------
    def __iter__(self):  # noqa: D401
        return iter(self._assemblies)

    def __len__(self) -> int:  # noqa: D401
        return len(self._assemblies)

    def __getitem__(self, idx: int) -> Assembly:  # noqa: D401
        return self._assemblies[idx]
