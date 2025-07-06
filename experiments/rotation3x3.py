"""Simple 3×3 rotation experiment using ArcAdapter and RotationTeacher."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.arc_loader import random_grid  # noqa: E402
from hx_adapters import ArcAdapter  # noqa: E402
from hx_teachers import RotationTeacher  # noqa: E402


def main(task_path: str | Path, angle: int = 90) -> None:  # noqa: D401
    grid = random_grid(task_path, split="train")
    # Ensure 3×3 for demo; if not, pad or crop
    if grid.shape != (3, 3):
        grid = grid[:3, :3]
    print("Input:\n", grid)

    adapter = ArcAdapter()
    adapter.encode(grid)

    teacher = RotationTeacher(grid.shape, angle=angle)
    target = teacher.predict(grid)
    print("Target (teacher):\n", target)

    # Run network naïvely for 50 ticks
    for _ in range(50):
        adapter.network.step()
    out = adapter.decode()
    print("Decoded after 50 ticks:\n", out)
    print("Reward:", teacher.reward(out, target))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rotation3x3.py path/to/arc_task.json [angle]")
        sys.exit(1)
    path = sys.argv[1]
    ang = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    main(path, angle=ang)
