"""Quick experiment script: rotation 90° learning demo.

NOTE: placeholder; full implementation will come later.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to PYTHONPATH for direct execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hx_adapters import ArcAdapter
from hx_core import SimpleScheduler
from hx_monitoring import SimpleCSVLogger


def run_once(seed: int = 0) -> None:  # noqa: D401
    rng = np.random.default_rng(seed)

    # Create a 2x2 pattern and its rotated version (dummy task)
    grid = rng.integers(0, 10, size=(2, 2))

    adapter = ArcAdapter()
    adapter.encode(grid)

    logger = SimpleCSVLogger(Path("logs") / f"run_{seed}.csv")
    scheduler = SimpleScheduler(adapter.network)

    for _ in range(50):
        scheduler.step()
        logger.log(adapter.network)

    decoded = adapter.decode()
    print("Input:\n", grid)
    print("Decoded after 50 ticks:\n", decoded)


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    run_once()
