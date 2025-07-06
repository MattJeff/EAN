"""Curriculum experiment – multiple 3×3 rotations with PolicyScheduler."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.arc_loader import load_task  # noqa: E402
from hx_adapters import ArcAdapter  # noqa: E402
from hx_teachers import RotationTeacher  # noqa: E402
from hx_core.optimizers import PolicyScheduler, ReinforceOptimizer  # noqa: E402
from hx_monitoring import SimpleCSVLogger  # noqa: E402
from hx_core.connectors import RotationHintConnector  # noqa: E402


def build_scheduler(grid: np.ndarray, angle: int) -> PolicyScheduler:  # noqa: D401
    """Initialise adapter + scheduler once and reuse across episodes."""
    adapter = ArcAdapter()
    adapter.encode(grid)

    # Keep assemblies alive long enough for learning
    from hx_core.strategies import CompetitiveEnergyPruning  # inline import to avoid circulars
    adapter.network.pruning_strategy = CompetitiveEnergyPruning(energy_cost=-1.0, protection_window=5)

    # Deterministic hint connections
    RotationHintConnector(angle=angle, weight=1.0).apply(
        network=adapter.network, neurons=adapter.neurons
    )

    teacher = RotationTeacher(grid.shape, angle)
    target = teacher.predict(grid)

    logger = SimpleCSVLogger("rotation3x3_log.csv")
    return PolicyScheduler(
        adapter=adapter,
        teacher=teacher,
        optimizer=ReinforceOptimizer(step_size=0.3, log_std=-1.0),
        target_grid=target,
        max_ticks=300,
        eval_interval=2,
        target_reward=0.95,
        logger=logger,
    )


def main(path: str, angle: int, episodes: int) -> None:  # noqa: D401
    train, _ = load_task(path)
    grid, _ = train[0]

    sched = build_scheduler(grid.copy(), angle)

    best = -1.0
    total = 0.0
    for ep in range(1, episodes + 1):
        # Re-inject energies for new episode
        sched.adapter.recharge(grid)
        r = sched.run()
        total += r
        if r > best:
            best = r
        print(f"Episode {ep}: reward={r}")
        if best >= 0.95:
            print("Target reward reached – early stop.")
            break

    mean = total / ep
    print(f"Mean reward: {mean}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python curriculum_rotation3x3.py <task.json> <angle> <episodes>")
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
