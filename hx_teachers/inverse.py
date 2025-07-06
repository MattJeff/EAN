"""InverseTeacher – oracle for colour inversion tasks (ARC style).

Two flavours:
1. *binary=True* (default) – treat grid as boolean (0 / >0) and invert.
2. *binary=False* – map colour `c` → `9 - c` (ARC palette is 0–9).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

__all__: List[str] = ["InverseTeacher"]


class InverseTeacher:  # pylint: disable=too-few-public-methods
    """Colour inversion oracle for ARC grids."""

    def __init__(self, grid_shape: Tuple[int, int], *, binary: bool = True):  # noqa: D401
        self.grid_shape = grid_shape
        self.binary = binary

    # ------------------------------------------------------------------
    def predict(self, inp: np.ndarray) -> np.ndarray:  # noqa: D401
        if self.binary:
            return (inp == 0).astype(int)  # invert 0↔1 (others → 0)
        return 9 - inp  # colour complement in palette 0–9

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
        path.write_text("binary" if self.binary else "full")

    @classmethod
    def load(cls, path: Path, grid_shape: Tuple[int, int]):  # noqa: D401
        mode = path.read_text().strip()
        return cls(grid_shape, binary=(mode == "binary"))
