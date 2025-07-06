"""Train HyperCubeX-EAN model on ARC tasks and save checkpoints.

This high-level script assembles a *minimal* curriculum based on simple
transformations (rotation, mirror, inverse, identity), launches a
`PolicyScheduler` for each task, logs basic metrics, and stores lightweight
checkpoints (<100 MB total).

Rapid experimentation is possible via `--max_tasks` and `--max_ticks`.

Example (full training):
    python scripts/train_model.py --train_dir data/training --weights_dir weights

Example (quick smoke-test on 5 tasks / 100 ticks):
    python scripts/train_model.py --max_tasks 5 --max_ticks 100
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Local imports – ensure project root is on sys.path ---------------------------
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hx_adapters.arc_adapter import ArcAdapter  # noqa: E402
from hx_core.optimizers import PolicyScheduler  # noqa: E402
from hx_monitoring import SimpleCSVLogger  # noqa: E402
from hx_teachers.rotation import RotationTeacher  # noqa: E402
from hx_teachers.mirror import MirrorTeacher  # noqa: E402
from hx_teachers.inverse import InverseTeacher  # noqa: E402
from hx_teachers.ground_truth import GroundTruthTeacher  # noqa: E402

# Re-use transform detector from submission helper -----------------------------
from arc_submission import _detect_transform  # type: ignore  # noqa: E402

__all__: List[str] = ["main"]

# -----------------------------------------------------------------------------
# Helper teachers --------------------------------------------------------------
# -----------------------------------------------------------------------------


class IdentityTeacher:  # pylint: disable=too-few-public-methods
    """Fallback teacher performing *identity* transformation."""

    def __init__(self, grid_shape: Tuple[int, int]):  # noqa: D401
        self.grid_shape = grid_shape

    # Identity prediction ---------------------------------------------------
    @staticmethod
    def predict(inp: np.ndarray) -> np.ndarray:  # noqa: D401
        return np.array(inp)

    # 0/1 reward (fraction of matching cells) -------------------------------
    @staticmethod
    def reward(pred: np.ndarray, target: np.ndarray, **_kwargs) -> float:  # noqa: D401
        total = pred.size
        wrong = int((pred != target).sum())
        return 1.0 - wrong / total

    # Dummy (de)serialisation helpers ---------------------------------------
    def save(self, _path: Path) -> None:  # noqa: D401
        pass

    @classmethod
    def load(cls, _path: Path, grid_shape: Tuple[int, int]):  # noqa: D401
        return cls(grid_shape)


# -----------------------------------------------------------------------------
# Factory to build appropriate teacher from transform --------------------------
# -----------------------------------------------------------------------------

def _build_teacher(transform: Tuple[str, Any], grid_shape: Tuple[int, int]):  # noqa: D401
    kind, param = transform
    if kind == "rotation":
        return RotationTeacher(grid_shape, angle=param)
    if kind == "mirror":
        return MirrorTeacher(grid_shape, axis=param)
    if kind == "inverse-binary":
        return InverseTeacher(grid_shape, binary=True)
    if kind == "inverse-color":
        return InverseTeacher(grid_shape, binary=False)
    # identity fallback
    return IdentityTeacher(grid_shape)


# -----------------------------------------------------------------------------
# Main training loop -----------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="HyperCubeX-EAN final training")
    parser.add_argument("--train_dir", type=Path, default=Path("data/training"))
    parser.add_argument("--weights_dir", type=Path, default=Path("weights"))
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit number of tasks for a quick run")
    parser.add_argument("--max_ticks", type=int, default=300, help="Maximum ticks per task")
    parser.add_argument("--target_reward", type=float, default=0.95, help="Early-stop reward threshold")
    parser.add_argument("--teacher", choices=["auto", "groundtruth"], default="auto", help="Teacher type: 'auto' heuristic or 'groundtruth' universal")
    args = parser.parse_args()

    # Prepare output folders ------------------------------------------------
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    (args.weights_dir / "logs").mkdir(exist_ok=True)

    # ---------------------------------------------------------------------
    # Iterate over tasks ---------------------------------------------------
    # ---------------------------------------------------------------------
    transform_map: Dict[str, Tuple[str, Any]] = {}

    json_files = sorted(args.train_dir.glob("*.json"))
    if args.max_tasks is not None:
        json_files = json_files[: args.max_tasks]

    for idx, json_file in enumerate(json_files, start=1):
        task_id = json_file.stem
        task = json.loads(json_file.read_text())
        train_examples = [
            (np.array(p["input"], dtype=int), np.array(p["output"], dtype=int))
            for p in task["train"]
        ]

        grid_inp = train_examples[0][0]
        target_grid = train_examples[0][1]

        if args.teacher == "auto":
            # Detect transformation + build heuristic teacher
            transform = _detect_transform(train_examples)
            transform_map[task_id] = transform
            teacher = _build_teacher(transform, grid_inp.shape)
            # Let teacher compute target (may differ from example for identity)
            target_grid = teacher.predict(grid_inp)
        else:  # groundtruth teacher
            transform_map[task_id] = ("groundtruth", None)
            teacher = GroundTruthTeacher(target_grid, binary=False)

        # Assemble adapter + scheduler ------------------------------------
        adapter = ArcAdapter()
        adapter.encode(grid_inp)

        log_path = args.weights_dir / "logs" / f"{task_id}.csv"
        logger = SimpleCSVLogger(log_path)

        scheduler = PolicyScheduler(
            adapter=adapter,
            teacher=teacher,
            optimizer="reinforce",
            target_grid=target_grid,
            max_ticks=args.max_ticks,
            eval_interval=5,
            target_reward=args.target_reward,
            logger=logger,
        )

        best_reward = scheduler.run()
        print(f"[{idx}/{len(json_files)}] {task_id}: reward={best_reward:.3f}")

        # Save checkpoint if decent performance ---------------------------
        if best_reward >= args.target_reward:
            ckpt_path = args.weights_dir / f"{task_id}.pkl.gz"
            with gzip.open(ckpt_path, "wb") as f:
                pickle.dump(adapter.network, f)

    # ---------------------------------------------------------------------
    # Save transformation map (for fast inference) -------------------------
    # ---------------------------------------------------------------------
    map_path = args.weights_dir / "transform_map.json"
    map_path.write_text(json.dumps(transform_map))
    size_mb = map_path.stat().st_size / 1_048_576
    print(f"Saved transform map for {len(transform_map)} tasks to {map_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
