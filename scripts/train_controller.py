"""Offline training loop for ControllerAssembly.

Runs the controller across a set of ARC tasks, updating its value table
incrementally and finally saving it to disk.  Uses the same simulation
pipeline as *run_benchmark.py* but shares a single controller instance across
all tasks.

Usage
-----
    python scripts/train_controller.py \
        --data_dir data/training \
        --out_path weights/controller.json \
        --max_tasks 400 --max_ticks 300

If an existing controller JSON is passed via ``--in_path``, the script resumes
training from it; otherwise a fresh controller (ε = 0.1) is created.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm  # type: ignore

# Project root for relative imports -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports ---------------------------------------------------------------------
from hx_adapters.arc_adapter import ArcAdapter  # noqa: E402
from hx_core.optimizers import PolicyScheduler  # noqa: E402
from hx_core.connectors import RandomConnector  # noqa: E402
from hx_core.assemblies import AssemblyStore  # noqa: E402
from hx_core.controller import ControllerAssembly  # noqa: E402
from hx_teachers.ground_truth import GroundTruthTeacher  # noqa: E402

#####################################################################################
# Helpers                                                                            
#####################################################################################

def _load_first_pair(task_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (input_grid, output_grid) of the first *train* pair."""
    data = json.loads(task_path.read_text())
    pair = data["train"][0]
    return np.array(pair["input"]), np.array(pair["output"])

#####################################################################################
# Main routine                                                                       
#####################################################################################

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train ControllerAssembly across tasks")
    parser.add_argument("--data_dir", type=Path, default=Path("data/training"))
    parser.add_argument("--in_path", type=Path, default=None, help="Existing controller JSON to resume from")
    parser.add_argument("--out_path", type=Path, default=Path("weights/controller.json"))
    parser.add_argument("--load_assemblies", type=Path, default=None, help="Path to assemblies JSON")
    parser.add_argument("--save_assemblies", type=Path, default=None, help="Where to save assemblies after training")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_ticks", type=int, default=300)
    parser.add_argument("--target_reward", type=float, default=0.95)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Prepare tasks list
    task_files = sorted(Path(args.data_dir).glob("*.json"))

    def _same_shape(p: Path) -> bool:
        gi, go = _load_first_pair(p)
        return gi.shape == go.shape

    task_files = [p for p in task_files if _same_shape(p)]
    if args.max_tasks is not None:
        task_files = task_files[: args.max_tasks]
    print(f"Training controller on {len(task_files)} tasks")

    # ------------------------------------------------------------------
    # Load or create controller
    if args.in_path and args.in_path.exists():
        controller = ControllerAssembly.load(args.in_path)
        print(f"Loaded controller from {args.in_path}")
    else:
        controller = ControllerAssembly(epsilon=0.1)

    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # Load / create assembly store
    assembly_store: AssemblyStore | None = None
    if args.load_assemblies and args.load_assemblies.exists():
        assembly_store = AssemblyStore.load(args.load_assemblies)
        print(f"Loaded assemblies from {args.load_assemblies} (n={len(assembly_store)})")
    elif args.save_assemblies:
        assembly_store = AssemblyStore()

    # ------------------------------------------------------------------
    # Loop over tasks
    for tp in tqdm(task_files):
        grid_in, grid_out = _load_first_pair(tp)

        adapter = ArcAdapter()
        adapter.encode(grid_in)

        teacher = GroundTruthTeacher(grid_out)
        scheduler = PolicyScheduler(
            adapter=adapter,
            teacher=teacher,
            connector=RandomConnector(n_connections=8),
            target_grid=grid_out,
            target_reward=args.target_reward,
            max_ticks=args.max_ticks,
            controller=controller,
            assembly_store=assembly_store,
        )
        start = time.perf_counter()
        best_reward = scheduler.run()
        duration = time.perf_counter() - start
        tqdm.write(f"{tp.stem}: reward={best_reward:.3f}  time={duration:.2f}s")

    # ------------------------------------------------------------------
        # Save assemblies --------------------------------------------------
    if args.save_assemblies and assembly_store is not None:
        assembly_store.save(args.save_assemblies)
        print(f"Saved assemblies to {args.save_assemblies}")

    # Save controller
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    controller.save(args.out_path)
    print(f"Saved trained controller to {args.out_path}")


if __name__ == "__main__":
    main()
