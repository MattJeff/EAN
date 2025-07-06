"""MirrorTeacher – oracle for mirror symmetry tasks (ARC style).

Supports vertical (flip left/right) and horizontal (flip top/bottom) reflections.
Similar API to RotationTeacher to ease curriculum composition.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

__all__: List[str] = ["MirrorTeacher"]


class MirrorTeacher:  # pylint: disable=too-few-public-methods
    """High-level teacher returning mirrored grids."""

    def __init__(self, grid_shape: Tuple[int, int], axis: str | None = None, *, binary: bool = True):  # noqa: D401
        if axis is not None and axis not in {"vertical", "horizontal"}:
            raise ValueError("axis must be 'vertical', 'horizontal', or None to autodetect")
        self.grid_shape = grid_shape
        self.axis = axis  # or None (discover)
        self.binary = binary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mirror(grid: np.ndarray, axis: str) -> np.ndarray:
        if axis == "vertical":
            return np.fliplr(grid)
        if axis == "horizontal":
            return np.flipud(grid)
        raise ValueError

    @classmethod
    def discover_axis(cls, examples: List[Tuple[np.ndarray, np.ndarray]]) -> str | None:  # noqa: D401
        for axis in ("vertical", "horizontal"):
            if all(np.array_equal(cls._mirror(inp, axis), out) for inp, out in examples):
                return axis
        return None

    # ------------------------------------------------------------------
    def predict(self, inp: np.ndarray) -> np.ndarray:  # noqa: D401
        if self.axis is None:
            axis = "vertical"
        else:
            axis = self.axis
        out = self._mirror(inp, axis)
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
    def save(self, path: Path) -> None:  # noqa: D401
        val = self.axis or "vertical"
        path.write_text(val)

    @classmethod
    def load(cls, path: Path, grid_shape: Tuple[int, int]):  # noqa: D401
        axis = path.read_text().strip()
        return cls(grid_shape, axis=axis)
