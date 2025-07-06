"""Wrapper around legacy RecursivePatternDetector to act as a teacher.

Given an input ARC grid, the teacher predicts the expected output grid using
symbolic recursion rules.  This can be used as:
    * Oracle for reward computation (RL).
    * Generator of training pairs for supervised fine-tuning.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

# Import legacy detector without hard dependency
from core.discovery.recursive_detector import RecursivePatternDetector

__all__ = ["RecursiveTeacher"]


class RecursiveTeacher:  # pylint: disable=too-few-public-methods
    """High-level teacher using RecursivePatternDetector under the hood."""

    def __init__(self, grid_shape: Tuple[int, int]):  # noqa: D401
        self.detector = RecursivePatternDetector(grid_shape)

    # ------------------------------------------------------------------
    def predict(self, inp: np.ndarray) -> np.ndarray:  # noqa: D401
        """Return a binary target grid aligned with current spike criterion.

        For the present MVP we simply mark cells whose colour value is > 4 as 1,
        mirroring the adapter's decoding logic.  This provides a consistent
        and straightforward reward signal before full recursive operations are
        implemented.
        """
        return (inp > 4).astype(int)

    # ------------------------------------------------------------------
    @staticmethod
    def reward(pred: np.ndarray, target: np.ndarray) -> float:  # noqa: D401
        """Simple negative Hamming distance normalised to [0,1]."""
        total = pred.size
        wrong = (pred != target).sum()
        return 1.0 - wrong / total

    # Optional serialization helpers -----------------------------------
    def save(self, path: Path) -> None:  # noqa: D401
        path.write_bytes(b"placeholder")

    @classmethod
    def load(cls, path: Path, grid_shape: Tuple[int, int]):  # noqa: D401
        _ = path.read_bytes()
        return cls(grid_shape)
