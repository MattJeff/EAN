"""GroundTruthTeacher – universal oracle using target grid directly.

Unlike *RotationTeacher* or *MirrorTeacher*, this teacher does **no**
transformation.  It simply stores the *target* grid and computes reward as the
fraction of matching cells.  This allows HyperCubeX-EAN to learn arbitrary
mappings without hand-crafted rules.

API mirrors other teachers for drop-in replacement.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Any

import numpy as np

__all__: List[str] = ["GroundTruthTeacher"]


class GroundTruthTeacher:  # pylint: disable=too-few-public-methods
    """Teacher that provides the exact target grid for reward computation.

    Parameters
    ----------
    target_grid : np.ndarray
        Desired output grid.  Must be 2-D array (H×W) of ints.
    binary : bool, default True
        If *True*, reward is computed on binarised grids (0 / >0).  This is
        useful when only the *shape* of the pattern matters.  If *False*, the
        full colour palette (0–9) is compared.
    """

    def __init__(self, target_grid: np.ndarray, *, binary: bool = True) -> None:  # noqa: D401
        if target_grid.ndim != 2:
            raise ValueError("target_grid must be 2-D")
        self.target_grid = np.asarray(target_grid, dtype=int)
        self.grid_shape: Tuple[int, int] = tuple(self.target_grid.shape)  # for consistency
        self.binary = bool(binary)

    # ------------------------------------------------------------------
    # Prediction – simply return stored target -------------------------
    # ------------------------------------------------------------------
    def predict(self, _inp: np.ndarray | None = None) -> np.ndarray:  # noqa: D401
        """Return the stored *target_grid* (ignores input)."""
        return np.array(self.target_grid, copy=True)

    # ------------------------------------------------------------------
    # Reward – fraction of matching cells ------------------------------
    # ------------------------------------------------------------------
    def reward(self, pred: np.ndarray, _unused_target: np.ndarray | None = None) -> float:  # noqa: D401
        """Compute accuracy between *pred* and ``self.target_grid``.

        The *target* argument is accepted for signature compatibility but
        ignored (ground-truth is internal).
        """
        if self.binary:
            pred = (pred > 0).astype(int)
            tgt = (self.target_grid > 0).astype(int)
        else:
            pred = np.asarray(pred, dtype=int)
            tgt = self.target_grid
        # Handle shape mismatch gracefully ---------------------------------
        if pred.shape != tgt.shape:
            return 0.0

        total = pred.size
        wrong = int((pred != tgt).sum())
        return 1.0 - wrong / total

    # ------------------------------------------------------------------
    # (De)serialisation helpers ----------------------------------------
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:  # noqa: D401
        """Save target grid as .npy file."""
        np.save(path, self.target_grid)

    @classmethod
    def load(cls, path: Path, *, binary: bool = True) -> "GroundTruthTeacher":  # noqa: D401
        tgt = np.load(path)
        return cls(tgt, binary=binary)
