"""ARC adapter integration smoke test."""
from __future__ import annotations

import numpy as np

from hx_adapters import ArcAdapter
from hx_core import SimpleScheduler


def test_arc_adapter_roundtrip():
    grid = np.random.randint(0, 10, size=(3, 3))
    adapter = ArcAdapter()
    adapter.encode(grid)

    # run few ticks
    scheduler = SimpleScheduler(adapter.network)
    scheduler.run(steps=5)

    out = adapter.decode()

    assert out.shape == grid.shape
    assert set(np.unique(out)) <= {0, 1}
