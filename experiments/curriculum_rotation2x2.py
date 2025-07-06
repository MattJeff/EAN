"""Curriculum experiment using RecursiveTeacher reward loop."""
from __future__ import annotations

import sys
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hx_adapters import ArcAdapter  # noqa: E402
from hx_core import Network, SimpleScheduler  # noqa: E402
from hx_teachers import RecursiveTeacher  # noqa: E402
from hx_monitoring import SimpleCSVLogger  # noqa: E402


def run_episode(seed: int = 0) -> float:  # noqa: D401
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 10, size=(2, 2))
    teacher = RecursiveTeacher(grid.shape)
    target = teacher.predict(grid)

    adapter = ArcAdapter()
    adapter.encode(grid)
    network: Network = adapter.network
    scheduler = SimpleScheduler(network)

    logger = SimpleCSVLogger(Path("logs") / f"curr_{seed}.csv")

    for _ in range(20):
        scheduler.step()
        logger.log(network)

    output = adapter.decode()
    reward = teacher.reward(output, target)
    print("Reward:", reward)
    return reward


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    mean_r = np.mean([run_episode(s) for s in range(5)])
    print("Mean reward:", mean_r)
