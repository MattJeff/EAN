"""Evaluate heuristic transform map on ARC training tasks.

This utility is *offline* (no network training involved).  It replays each
ARC training task, applies the transform inferred in `transform_map.json`
(or falls back to auto-detection) and reports per-task accuracy plus a global
summary.

Usage
-----
python scripts/evaluate_transform_map.py \
    --train_dir data/training \
    --mapping   weights/transform_map.json

If *--mapping* is omitted, transforms are detected on the fly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arc_submission import _apply_transform, _detect_transform  # type: ignore  # noqa: E402

# -----------------------------------------------------------------------------


def _calc_reward(pred: np.ndarray, target: np.ndarray) -> float:
    """Return fraction of correctly matched cells (0–1).

    If *pred* and *target* shapes differ, accuracy is 0.0 (failure)."""
    if pred.shape != target.shape:
        return 0.0
    total = pred.size
    wrong = int((pred != target).sum())
    return 1.0 - wrong / total


# -----------------------------------------------------------------------------
# Evaluation loop --------------------------------------------------------------
# -----------------------------------------------------------------------------

def evaluate(train_dir: Path, mapping_path: Path | None = None) -> None:  # noqa: D401
    mapping: Dict[str, Tuple[str, Any]] = {}
    if mapping_path is not None and mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text())
        print(f"Loaded mapping with {len(mapping)} entries from {mapping_path}\n")

    header = (
        f"{'task':<10} | {'kind':<10} | {'param':<8} | {'pairs':<5} | {'avg_acc':<7}"
    )
    print(header)
    print("-" * len(header))

    acc_values: List[float] = []
    perfect = 0

    for json_file in sorted(train_dir.glob("*.json")):
        task_id = json_file.stem
        task = json.loads(json_file.read_text())
        examples = [
            (np.array(p["input"], dtype=int), np.array(p["output"], dtype=int))
            for p in task["train"]
        ]

        # Get transform ----------------------------------------------------
        if task_id in mapping:
            transform = tuple(mapping[task_id])  # type: ignore[assignment]
        else:
            transform = _detect_transform(examples)

        # Compute accuracy over training pairs ----------------------------
        accs: List[float] = []
        for inp, target in examples:
            pred = _apply_transform(inp, transform)
            accs.append(_calc_reward(pred, target))
        mean_acc = float(np.mean(accs))

        acc_values.append(mean_acc)
        if mean_acc == 1.0:
            perfect += 1

        kind, param = transform
        param_str = "-" if param is None else str(param)
        print(f"{task_id:<10} | {kind:<10} | {param_str:<8} | {len(examples):<5} | {mean_acc:<7.3f}")

    # ------------------------------------------------------------------
    print("\nSummary")
    print("-------")
    n = len(acc_values)
    mean_all = float(np.mean(acc_values))
    median_all = float(np.median(acc_values))
    print(f"Tasks evaluated : {n}")
    print(f"Perfect tasks  : {perfect} ({perfect / n:.1%})")
    print(f"Mean accuracy  : {mean_all:.3f}")
    print(f"Median accuracy: {median_all:.3f}")
    print(f"≥0.90 accuracy : {(np.sum(np.array(acc_values) >= 0.9) / n):.1%}")


# -----------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Evaluate heuristic transforms on ARC training tasks")
    parser.add_argument("--train_dir", type=Path, default=Path("data/training"))
    parser.add_argument("--mapping", type=Path, default=None, help="Path to transform_map.json (optional)")
    args = parser.parse_args()

    evaluate(args.train_dir, args.mapping)


if __name__ == "__main__":
    main()
