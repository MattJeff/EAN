"""ControllerAssembly – policy deciding which primitive to apply.

This *controller* is a lightweight façade around a list of *primitives* (callables
``f(grid) -> grid``).  It exposes a :py:meth:`act` method that returns the chosen
primitive's output and its identifier.

The initial implementation is *random uniform* – this lets us plug integration
and tests quickly.  Future iterations can substitute an RL policy (e.g.
ε-greedy with a Q-table) behind the same interface.
"""
from __future__ import annotations

import random
import json
from pathlib import Path
from typing import Callable, List, Tuple, Any

import numpy as np

__all__: List[str] = [
    "ControllerAssembly",
    "rotate90",
    "mirror_h",
    "translate",
    "scale2x",
    "shrink2x",
    "flood_fill",
    "recolor",
]

Grid = np.ndarray
PrimitiveFn = Callable[[Grid], Grid]


# -----------------------------------------------------------------------------
# Basic primitive transformations ---------------------------------------------
# NOTE: All primitives must preserve the *shape* of the grid to remain
# compatible with the GlobalWorkspace.  Operations such as scaling therefore
# include a post-processing step (crop/pad) so the output grid matches the
# input dimensions.

# -----------------------------------------------------------------------------

def rotate90(grid: Grid) -> Grid:  # noqa: D401
    return np.rot90(grid, k=1)


def mirror_h(grid: Grid) -> Grid:  # noqa: D401
    return np.fliplr(grid)


def translate(grid: Grid, dx: int = 1, dy: int = 0) -> Grid:  # noqa: D401
    """Wrap-around translation."""
    return np.roll(np.roll(grid, dx, axis=1), dy, axis=0)


def scale2x(grid: Grid) -> Grid:  # noqa: D401
    """Nearest-neighbour up-scaling by factor 2 then crop to original size."""
    h, w = grid.shape
    scaled = np.kron(grid, np.ones((2, 2), dtype=grid.dtype))  # shape 2h×2w
    return scaled[:h, :w]


def shrink2x(grid: Grid) -> Grid:  # noqa: D401
    """Down-scale by factor 2 via strided slicing, then repeat to restore size."""
    h, w = grid.shape
    down = grid[::2, ::2]
    # If original dimension is odd, pad before repeating
    down = down[: (h + 1) // 2, : (w + 1) // 2]
    return np.kron(down, np.ones((2, 2), dtype=grid.dtype))[:h, :w]


def flood_fill(grid: Grid) -> Grid:  # noqa: D401
    """Flood-fill from a random starting pixel with a random color."""
    h, w = grid.shape
    start_y, start_x = np.random.randint(0, h), np.random.randint(0, w)
    target_color = grid[start_y, start_x]
    # Choose new color different from target
    colors = [c for c in np.unique(grid) if c != target_color]
    if not colors:
        return grid.copy()
    new_color = int(np.random.choice(colors))
    mask = np.zeros_like(grid, dtype=bool)
    stack = [(start_y, start_x)]
    while stack:
        y, x = stack.pop()
        if not (0 <= y < h and 0 <= x < w):
            continue
        if grid[y, x] != target_color or mask[y, x]:
            continue
        mask[y, x] = True
        stack.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])
    out = grid.copy()
    out[mask] = new_color
    return out


def recolor(grid: Grid) -> Grid:  # noqa: D401
    """Permute colors cyclically."""
    uniq = np.unique(grid)
    if len(uniq) <= 1:
        return grid.copy()
    mapping = {c: uniq[(i + 1) % len(uniq)] for i, c in enumerate(uniq)}
    vectorized = np.vectorize(lambda v: mapping[v])
    return vectorized(grid)


PRIMITIVES: List[Tuple[str, PrimitiveFn]] = [
    ("rotate90", rotate90),
    ("mirror_h", mirror_h),
    ("translate", translate),
    ("scale2x", scale2x),
    ("shrink2x", shrink2x),
    ("flood_fill", flood_fill),
    ("recolor", recolor),
]


# -----------------------------------------------------------------------------
# Controller ------------------------------------------------------------------
# -----------------------------------------------------------------------------

class ControllerAssembly:  # pylint: disable=too-few-public-methods
    """Epsilon-greedy policy over a fixed set of primitives.

    Parameters
    ----------
    epsilon : float, default 0.1
        Exploration probability.  Can be overridden at each :py:meth:`act` call.
    """

    def __init__(
        self,
        primitives: List[Tuple[str, PrimitiveFn]] | None = None,
        *,
        epsilon: float = 0.1,
    ) -> None:  # noqa: D401
        self.primitives = primitives if primitives is not None else list(PRIMITIVES)
        self.epsilon = float(epsilon)

        self._values: dict[str, float] = {name: 0.0 for name, _ in self.primitives}
        self._counts: dict[str, int] = {name: 0 for name, _ in self.primitives}
        self._last_choice: str | None = None
        # Convenience mapping
        self._name_to_fn: dict[str, PrimitiveFn] = {name: fn for name, fn in self.primitives}

    # ------------------------------------------------------------------
    def act(self, grid: Grid, *, epsilon: float | None = None) -> Tuple[str, Grid]:  # noqa: D401
        """Choose primitive via ε-greedy and return (name, transformed)."""
        eps = self.epsilon if epsilon is None else float(epsilon)
        if random.random() < eps:
            name = random.choice(list(self._name_to_fn.keys()))
        else:
            # Break ties at random among best
            max_val = max(self._values.values())
            best = [n for n, v in self._values.items() if v == max_val]
            name = random.choice(best)
        self._last_choice = name
        out_grid = self._name_to_fn[name](grid)
        return name, out_grid

    # ------------------------------------------------------------------
    def update(self, reward: float) -> None:  # noqa: D401
        """Incremental mean update for the last chosen primitive."""
        if self._last_choice is None:
            return  # act() not called yet
        name = self._last_choice
        self._counts[name] += 1
        n = self._counts[name]
        value = self._values[name]
        self._values[name] = value + (reward - value) / n
        # Clear last so we cannot double-update
        self._last_choice = None

    # ------------------------------------------------------------------
    # Persistence ------------------------------------------------------
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:  # noqa: D401
        data = {
            "epsilon": self.epsilon,
            "values": self._values,
            "counts": self._counts,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "ControllerAssembly":  # noqa: D401
        data = json.loads(Path(path).read_text())
        obj = cls(epsilon=data["epsilon"])
        obj._values = {k: float(v) for k, v in data["values"].items()}
        obj._counts = {k: int(c) for k, c in data["counts"].items()}
        return obj

    # Convenience -------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.primitives)

    def __iter__(self):  # noqa: D401
        return iter(self.primitives)

    # Convenience -------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.primitives)

    def __iter__(self):  # noqa: D401
        return iter(self.primitives)
