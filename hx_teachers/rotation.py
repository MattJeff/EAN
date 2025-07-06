"""RotationTeacher – oracle for rotation tasks (ARC style).

Given an input grid, the teacher predicts the grid rotated by a fixed set of
clockwise rotations (90°, 180°, 270°).  The *angle* can be specified
explicitly or discovered by matching examples through brute-force.

For the MVP we support two modes:
1. *angle* passed at construction → deterministic rotation.
2. *discover* given a list of (inp, out) examples → returns the inferred angle.

The API mirrors the existing `RecursiveTeacher` for consistency.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

__all__: List[str] = ["RotationTeacher"]


class RotationTeacher:  # pylint: disable=too-few-public-methods
    """High-level teacher that outputs rotated grids."""

    def __init__(self, grid_shape: Tuple[int, int], angle: int | None = None, *, binary: bool = True):  # noqa: D401
        self.grid_shape = grid_shape
        self.angle = angle  # degrees clockwise (90, 180, 270) or None
        self.binary = binary

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate(grid: np.ndarray, angle: int) -> np.ndarray:
        """Rotate *grid* clockwise by *angle* degrees."""
        k = (angle // 90) % 4  # np.rot90 uses counter-clockwise
        return np.rot90(grid, k=4 - k)  # convert to clockwise

    @classmethod
    def discover_angle(cls, examples: List[Tuple[np.ndarray, np.ndarray]]) -> int | None:  # noqa: D401
        """Infer rotation angle from example pairs (inp, out)."""
        for angle in (90, 180, 270):
            if all(np.array_equal(cls._rotate(inp, angle), out) for inp, out in examples):
                return angle
        return None

    # ------------------------------------------------------------------
    def predict(self, inp: np.ndarray) -> np.ndarray:  # noqa: D401
        """Return rotated prediction for *inp*."""
        if self.angle is None:
            # Default to 90° if no angle specified
            angle = 90
        else:
            angle = self.angle
        out = self._rotate(inp, angle)
        if self.binary:
            out = (out > 0).astype(int)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def reward(pred: np.ndarray, target: np.ndarray, *, binary: bool = True) -> float:  # noqa: D401
        if binary:
            pred = (pred > 0).astype(int)
            target = (target > 0).astype(int)
        total = pred.size
        wrong = (pred != target).sum()
        return 1.0 - wrong / total

    # ------------------------------------------------------------------
    # Serialization helpers (placeholder)
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:  # noqa: D401
        with path.open("wb") as f:
            f.write(str(self.angle or 90).encode())

    @classmethod
    def load(cls, path: Path, grid_shape: Tuple[int, int]):  # noqa: D401
        angle = int(path.read_bytes().decode())
        return cls(grid_shape, angle=angle)
