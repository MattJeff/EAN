"""Quick benchmark utility for HyperCubeX EAN.

Usage
-----
python benchmarks/bench_ean.py --grid 10 --ticks 300 --backend cpu
python benchmarks/bench_ean.py --grid 10 --ticks 300 --backend torch

Results are printed as JSON for easy parsing.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hx_adapters import ArcAdapter  # noqa: E402
from hx_core.backend import select_backend  # noqa: E402


def run_bench(grid_size: int, ticks: int, backend: str) -> dict[str, Any]:  # noqa: D401
    os.environ["EAN_BACKEND"] = backend

    # Synthetic random grid (colours 0-9)
    grid = np.random.randint(0, 10, size=(grid_size, grid_size), dtype=int)

    adapter = ArcAdapter()  # Network auto-selects backend via env-var
    adapter.encode(grid)

    net = adapter.network

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(ticks):
        net.step()
    duration = time.perf_counter() - t0
    current, peak = map(lambda x: x / 1024 / 1024, tracemalloc.get_traced_memory())
    tracemalloc.stop()

    return {
        "grid": grid_size,
        "ticks": ticks,
        "backend": backend,
        "time_s": duration,
        "tps": ticks / duration,
        "mem_current_mb": round(current, 3),
        "mem_peak_mb": round(peak, 3),
        "assemblies": len(net.assemblies),
    }


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="EAN micro-benchmark")
    parser.add_argument("--grid", type=int, default=3, help="Grid dimension N (NxN)")
    parser.add_argument("--ticks", type=int, default=300, help="# of simulation ticks")
    parser.add_argument("--backend", choices=list({"cpu", "torch"}), default=select_backend(), help="Backend to test")
    args = parser.parse_args()

    res = run_bench(args.grid, args.ticks, args.backend)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
