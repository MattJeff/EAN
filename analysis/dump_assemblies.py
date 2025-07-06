"""Quick inspection tool for a saved AssemblyStore.

Usage
-----
    python analysis/dump_assemblies.py --in_path weights/assemblies.json \
        --out_dir analysis/reports

Generates:
    • heatmap_<name>.png – frequency of active cells across assemblies.
    • strength_hist_<name>.png – histogram of assembly strengths.
    • assemblies_stats.csv – basic stats (count, mean, median, max).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

# Ensure project root in path ---------------------------------------------------
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hx_core.assemblies import AssemblyStore  # noqa: E402


###############################################################################
# Helpers
###############################################################################

def _frequency_map(store: AssemblyStore, shape: tuple[int, int]) -> np.ndarray:
    """Return 2-D array counting presence of coordinates across assemblies."""
    freq = np.zeros(shape, dtype=int)
    for asm in store:
        for y, x in asm.coords:
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                freq[y, x] += 1
    return freq

###############################################################################
# Main
###############################################################################

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Visualise AssemblyStore contents")
    parser.add_argument("--in_path", type=Path, required=True, help="JSON file produced by AssemblyStore.save")
    parser.add_argument("--out_dir", type=Path, default=Path("analysis/reports"))
    parser.add_argument("--grid_h", type=int, default=30)
    parser.add_argument("--grid_w", type=int, default=30)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    store = AssemblyStore.load(args.in_path)
    print(f"Loaded {len(store)} assemblies from {args.in_path}")

    # ------------------------------------------------------------------
    # Heatmap of frequency
    shape = (args.grid_h, args.grid_w)
    freq = _frequency_map(store, shape)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(freq, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Frequency")
    ax.set_title("Assembly cell frequency")
    heat_path = args.out_dir / f"heatmap_{args.in_path.stem}.png"
    fig.savefig(heat_path)
    print(f"Saved heatmap to {heat_path}")

    # ------------------------------------------------------------------
    # Strength histogram
    strengths: List[float] = [asm.strength for asm in store]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(strengths, bins=20, color="steelblue", alpha=0.8)
    ax.set_xlabel("Strength")
    ax.set_ylabel("Count")
    ax.set_title("Assembly strength histogram")
    hist_path = args.out_dir / f"strength_hist_{args.in_path.stem}.png"
    fig.savefig(hist_path)
    print(f"Saved histogram to {hist_path}")

    # ------------------------------------------------------------------
    # CSV stats
    stats = {
        "count": len(strengths),
        "mean": np.mean(strengths) if strengths else 0.0,
        "median": np.median(strengths) if strengths else 0.0,
        "std": np.std(strengths) if strengths else 0.0,
        "max": max(strengths) if strengths else 0.0,
    }
    csv_path = args.out_dir / "assemblies_stats.csv"
    pd.DataFrame([stats]).to_csv(csv_path, index=False)
    print(f"Saved stats to {csv_path}")


if __name__ == "__main__":
    main()
