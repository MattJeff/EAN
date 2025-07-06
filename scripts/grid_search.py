"""Mini grid-search over assembly hyper-parameters.

This helper launches *train_controller.py* multiple times with different
combinations of ``--asm-alpha`` and ``--asm-decay`` values, logging the
aggregate reward for each pair in a CSV file.  It is intentionally simple
and relies on subprocess calls so users may interrupt and resume freely.

Usage
-----
    python scripts/grid_search.py --alphas 0.05 0.1 0.2 --decays 0.99 0.995

The script writes *grid_search_results.csv* in the current directory.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import List

PYTHON = sys.executable  # Use same interpreter as parent process
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(alpha: float, decay: float, extra_args: List[str]) -> float:
    """Return mean reward of *train_controller.py* with given params."""
    cmd = [
        PYTHON,
        str(PROJECT_ROOT / "scripts" / "train_controller.py"),
        "--target_reward",
        "0.9",
        "--max_tasks",
        "50",
        "--asm_alpha",
        str(alpha),
        "--asm_decay",
        str(decay),
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Training failed")

    # Expect last line like: "Mean reward: 0.83"
    for line in result.stdout.splitlines()[::-1]:
        if "Mean reward" in line:
            return float(line.rsplit(" ", 1)[-1])
    raise ValueError("Could not parse reward from output")


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Mini grid-search over assembly hyper-parameters")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1])
    parser.add_argument("--decays", type=float, nargs="+", default=[0.999])
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Additional args passed to train_controller.py")
    args = parser.parse_args()

    results_path = Path("grid_search_results.csv")
    with results_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["alpha", "decay", "mean_reward"])
        for alpha, decay in product(args.alphas, args.decays):
            print(f"Running alpha={alpha}, decay={decay} ...")
            reward = _run(alpha, decay, args.extra or [])
            writer.writerow([alpha, decay, reward])
            fp.flush()
            print(f" -> mean reward={reward:.3f}")

    print(f"Results written to {results_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
