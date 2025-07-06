"""Benchmark script: compare heuristic, groundtruth, and controller modes.

Usage
-----
    python scripts/run_benchmark.py \
        --data_dir data/training \
        --out_dir bench_results \
        --modes heuristic,groundtruth,controller \
        --max_tasks 400 --max_ticks 300

This will produce:
    bench_results/metrics.csv
    bench_results/reward_curves.png

Dependencies: numpy, matplotlib, tqdm
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm  # type: ignore
import matplotlib.pyplot as plt

# Ensure project root on path ---------------------------------------------------
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports (lazy to avoid heavy deps if only plotting) --------------------
from hx_adapters.arc_adapter import ArcAdapter  # noqa: E402
from hx_core.optimizers import PolicyScheduler  # noqa: E402
from hx_core.connectors import RandomConnector  # noqa: E402
from hx_core.assemblies import AssemblyStore  # noqa: E402
from hx_core.controller import (  # noqa: E402
    ControllerAssembly,
    rotate90,
    mirror_h,
    translate,
)
from hx_teachers.ground_truth import GroundTruthTeacher  # noqa: E402
from hx_monitoring import SimpleCSVLogger  # noqa: E402

# Helpers re-used from training script -----------------------------------------
from arc_submission import _detect_transform  # type: ignore  # noqa: E402
from scripts.train_model import _build_teacher  # type: ignore  # noqa: E402

################################################################################
# Core runner
################################################################################

def _load_first_pair(task_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (input_grid, output_grid) of the first *train* pair."""
    data = json.loads(task_path.read_text())
    pair = data["train"][0]
    return np.array(pair["input"]), np.array(pair["output"])


def _run_single_task(task_path: Path, *, mode: str, max_ticks: int, target_reward: float,
                     asm_alpha: float = 0.1, asm_decay: float = 0.999,
                     shared_controller: ControllerAssembly | None = None,
                     assembly_store: AssemblyStore | None = None) -> Dict[str, Any]:
    """Run PolicyScheduler on one task and return metrics dict."""
    grid_in, grid_out = _load_first_pair(task_path)
    start_t = time.perf_counter()

    adapter = ArcAdapter()
    adapter.encode(grid_in)

    # Teacher selection --------------------------------------------------
    if mode == "heuristic":
        transform = _detect_transform([(grid_in, grid_out)])
        teacher = _build_teacher(transform, grid_in.shape)

        # Direct heuristic prediction – no scheduler run -----------------
        pred = teacher.predict(grid_in)
        best_reward = teacher.reward(pred, grid_out)
        duration = time.perf_counter() - start_t
        return {
            "task_id": task_path.stem,
            "mode": mode,
            "best_reward": round(best_reward, 4),
            "ticks": 0,
            "duration_s": round(duration, 2),
        }
    elif mode in {"groundtruth", "controller"}:
        teacher = GroundTruthTeacher(grid_out)
        workspace = None  # default created inside PolicyScheduler

        # Build / reuse controller with primitives safe for current grid shape ----------
        h, w = grid_in.shape
        primitives = [("identity", lambda g: g)]
        if mode == "controller":
            # Include additional transforms; exclude rotate90 if non-square
            if h == w:
                primitives.append(("rotate90", rotate90))
            primitives.append(("mirror_h", mirror_h))
            primitives.append(("translate", translate))
            eps = 0.1
        else:  # groundtruth – identity only
            eps = 0.0
        if shared_controller is not None:
            controller = shared_controller
        else:
            controller = ControllerAssembly(primitives=primitives, epsilon=eps)
    else:
        raise ValueError(f"Unknown mode {mode}")

    connector = RandomConnector(n_connections=8)

    scheduler = PolicyScheduler(
        adapter=adapter,
        teacher=teacher,
        connector=connector,
        target_grid=grid_out,
        target_reward=target_reward,
        max_ticks=max_ticks,
        workspace=workspace,
        controller=controller,
        assembly_store=assembly_store,
        assembly_alpha=asm_alpha,
        decay_rate=asm_decay,
    )

    best_reward = scheduler.run()
    duration = time.perf_counter() - start_t
    n_ticks = adapter.network.tick

    return {
        "task_id": task_path.stem,
        "mode": mode,
        "best_reward": round(best_reward, 4),
        "ticks": n_ticks,
        "duration_s": round(duration, 2),
    }

################################################################################
# CLI
################################################################################

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="HyperCubeX-EAN benchmark runner")
    parser.add_argument("--data_dir", type=Path, default=Path("data/training"))
    parser.add_argument("--out_dir", type=Path, default=Path("bench_results"))
    parser.add_argument("--modes", type=str, default="heuristic,groundtruth,controller")
    parser.add_argument("--load_controller", type=Path, default=None, help="Path to existing controller JSON")
    parser.add_argument("--save_controller", type=Path, default=None, help="Where to save controller after run")
    parser.add_argument("--load_assemblies", type=Path, default=None, help="Path to assemblies JSON")
    parser.add_argument("--save_assemblies", type=Path, default=None, help="Where to save assemblies after run")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_ticks", type=int, default=300)
    parser.add_argument("--target_reward", type=float, default=0.95)
    parser.add_argument("--asm_alpha", type=float, default=0.1, help="EMA alpha for assembly strength update")
    parser.add_argument("--asm_decay", type=float, default=0.999, help="Exponential decay rate for stale assemblies")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.csv"

    # Collect task files -------------------------------------------------
    task_files = sorted(Path(args.data_dir).glob("*.json"))
    # Keep only tasks whose first train pair has matching input/output shape
    def _same_shape(p: Path) -> bool:
        grid_in, grid_out = _load_first_pair(p)
        return grid_in.shape == grid_out.shape
    task_files = [p for p in task_files if _same_shape(p)]
    if args.max_tasks is not None:
        task_files = task_files[: args.max_tasks]
    print(f"Running benchmark on {len(task_files)} tasks with matching shapes, modes={modes}")

    # ------------------------------------------------------------------
    # Shared controller instance (for controller mode)  #
    # ------------------------------------------------------------------
    controller_shared: ControllerAssembly | None = None
    if "controller" in modes:
        if args.load_controller and args.load_controller.exists():
            controller_shared = ControllerAssembly.load(args.load_controller)
            print(f"Loaded controller from {args.load_controller}")
        else:
            controller_shared = ControllerAssembly(epsilon=0.1)

        # Shared assembly store -------------------------------------------------
    assembly_store_shared: AssemblyStore | None = None
    if args.load_assemblies and args.load_assemblies.exists():
        assembly_store_shared = AssemblyStore.load(args.load_assemblies)
        print(f"Loaded assemblies from {args.load_assemblies} (n={len(assembly_store_shared)})")
    elif args.save_assemblies:
        assembly_store_shared = AssemblyStore()

    rows: List[Dict[str, Any]] = []
    for mode in modes:
        for tp in tqdm(task_files, desc=f"Mode={mode}"):
            rows.append(
                _run_single_task(
                    tp,
                    mode=mode,
                    max_ticks=args.max_ticks,
                    target_reward=args.target_reward,
                    asm_alpha=args.asm_alpha,
                    asm_decay=args.asm_decay,
                    shared_controller=controller_shared if mode == "controller" else None,
                    assembly_store=assembly_store_shared,
                )
            )

    # Write CSV ----------------------------------------------------------
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {metrics_path}")

    # Plot reward distributions -----------------------------------------
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode in modes:
            vals = df[df["mode"] == mode]["best_reward"].values
            ax.hist(vals, bins=20, alpha=0.5, label=mode)
        ax.set_xlabel("Best reward")
        ax.set_ylabel("Count")
        ax.set_title("Reward distribution per mode")
        ax.legend()
        fig.tight_layout()
        plot_path = args.out_dir / "reward_hist.png"
        fig.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
    except ImportError:
        print("pandas not installed – skipping plot")

    # Save assemblies after run ----------------------------------------
    if args.save_assemblies and assembly_store_shared is not None:
        assembly_store_shared.save(args.save_assemblies)
        print(f"Saved assemblies to {args.save_assemblies}")

    # Save controller after run ----------------------------------------
    if args.save_controller and controller_shared is not None:
        controller_shared.save(args.save_controller)
        print(f"Saved controller to {args.save_controller}")


if __name__ == "__main__":
    main()
