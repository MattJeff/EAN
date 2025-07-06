"""Minimal ARC JSON loader.

The Abstraction & Reasoning Corpus stores each task as a JSON file with
`train` and `test` lists of {"input": [[...]], "output": [[...]]}.
This helper exposes two utilities:

• `load_task(path)` → returns `(train_pairs, test_pairs)` where each pair is
  `(np.ndarray, np.ndarray)`.
• `random_grid(path, split="train")` → returns a random *input* grid from the
  specified split (useful for quick demos).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

__all__ = ["load_task", "random_grid"]


def _to_np(grid: List[List[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=int)


def load_task(path: str | Path) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:  # noqa: D401
    path = Path(path)
    obj = json.loads(path.read_text())
    train = [(_to_np(p["input"]), _to_np(p["output"])) for p in obj["train"]]
    test = [(_to_np(p["input"]), _to_np(p["output"])) for p in obj["test"]]
    return train, test


def random_grid(path: str | Path, split: str = "train") -> np.ndarray:  # noqa: D401
    train, test = load_task(path)
    pairs = train if split == "train" else test
    inp, _ = random.choice(pairs)
    return inp.copy()
