"""Demo of PolicyScheduler with WeightPerturbOptimizer on a 2×2 grid."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hx_adapters import ArcAdapter  # noqa: E402
from hx_teachers import RecursiveTeacher  # noqa: E402
from hx_core.connectors import RandomConnector  # noqa: E402
from hx_core.optimizers import PolicyScheduler  # noqa: E402


def main(seed: int = 0) -> None:  # noqa: D401
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 10, size=(2, 2))
    print("Input grid:\n", grid)

    adapter = ArcAdapter()
    adapter.encode(grid)
    # Initialise random connections
    RandomConnector(n_connections=8, weight=0.5).apply(
        network=adapter.network, neurons=adapter.neurons
    )

    teacher = RecursiveTeacher(grid.shape)

    target = teacher.predict(grid)

    policy = PolicyScheduler(
        adapter=adapter, teacher=teacher, target_grid=target
    )
    best = policy.run()
    print("Best reward:", best)
    print("Decoded after optimization:\n", adapter.decode())


if __name__ == "__main__":
    main()
