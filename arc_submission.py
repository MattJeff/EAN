"""Generate submission.json for ARC Prize 2025 Kaggle competition.

Usage (offline):
    python arc_submission.py --challenges arc-agi_test-challenges.json --output submission.json

This script loads the challenges file (train+test pairs) and calls a
`predict_task` helper which uses HyperCubeX-EAN components (Adapter,
Teacher guesser, Scheduler) to produce *two* prediction attempts per test
output, as mandated by the rules.  The default implementation here is a
placeholder returning zeros grids – replace with your trained policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Import your system (lightweight parts only)
from hx_adapters.arc_adapter import ArcAdapter
from hx_teachers.rotation import RotationTeacher
from hx_teachers.mirror import MirrorTeacher
from hx_teachers.inverse import InverseTeacher

# -----------------------------------------------------------------------------
# Core prediction per task -----------------------------------------------------
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------
# Transformation helpers ------------------------------------------
# ------------------------------------------------------------------

def _detect_transform(examples: List[tuple[np.ndarray, np.ndarray]]):
    """Infer simple transformation from training (input, output) pairs.

    Returns a tuple (kind, param) where kind ∈ {
        'rotation', 'mirror', 'inverse-binary', 'inverse-color', 'identity'
    } and *param* is an extra parameter (angle, axis, etc.).
    """
    # --- Rotation -------------------------------------------------
    angle = RotationTeacher.discover_angle(examples)
    if angle is not None:
        return ("rotation", angle)

    # --- Mirror ---------------------------------------------------
    axis = MirrorTeacher.discover_axis(examples)
    if axis is not None:
        return ("mirror", axis)

    # --- Inverse (binary / colour) --------------------------------
    inv_binary = all(np.array_equal(out, (inp == 0).astype(int)) for inp, out in examples)
    if inv_binary:
        return ("inverse-binary", None)

    inv_colour = all(np.array_equal(out, 9 - inp) for inp, out in examples)
    if inv_colour:
        return ("inverse-color", None)

    # Fallback -----------------------------------------------------
    return ("identity", None)


def _apply_transform(grid: np.ndarray, transform: tuple[str, Any]) -> np.ndarray:
    """Apply *transform* returned by `_detect_transform` to *grid*."""
    kind, param = transform
    if kind == "rotation":
        return RotationTeacher._rotate(grid, param)  # type: ignore[arg-type]
    if kind == "mirror":
        return MirrorTeacher._mirror(grid, param)  # type: ignore[arg-type]
    if kind == "inverse-binary":
        return (grid == 0).astype(int)
    if kind == "inverse-color":
        return 9 - grid
    # identity
    return np.array(grid)


# Path to optional pre-computed mapping ---------------------------------
_MAPPING_PATH = Path(__file__).with_suffix("").parent / "weights" / "transform_map.json"


def _load_mapping():  # noqa: D401
    if _MAPPING_PATH.is_file():
        return json.loads(_MAPPING_PATH.read_text())
    return {}


_MAPPING_CACHE = _load_mapping()


def predict_task(task: Dict[str, Any], task_id: str | None = None):  # noqa: D401
    """Produce 2 prediction attempts for each *test* output in *task*."""
    # Option 1: mapping ------------------------------------------------
    if task_id and task_id in _MAPPING_CACHE:
        transform = tuple(_MAPPING_CACHE[task_id])  # type: ignore[assignment]
    else:
        # Fallback: infer from training examples ------------------------
        train_examples = [
            (np.array(p["input"], dtype=int), np.array(p["output"], dtype=int))
            for p in task["train"]
        ]
        transform = _detect_transform(train_examples)

    # 2. Apply to each test input -----------------------------------
    preds = []
    for pair in task["test"]:
        inp = np.array(pair["input"], dtype=int)
        attempt1 = _apply_transform(inp, transform)
        attempt2 = np.array(inp)  # identity fallback
        preds.append((attempt1, attempt2))
    return preds


# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tasks = json.loads(args.challenges.read_text())

    submission: Dict[str, List[Dict[str, List[List[int]]]]] = {}
    for task_id, task in tasks.items():
        preds = predict_task(task)
        # Build JSON structure
        out_list: List[Dict[str, List[List[int]]]] = []
        for attempt1, attempt2 in preds:
            out_list.append(
                {
                    "attempt_1": attempt1.tolist(),
                    "attempt_2": attempt2.tolist(),
                }
            )
        submission[task_id] = out_list

    args.output.write_text(json.dumps(submission))
    print(f"Saved submission to {args.output} with {len(submission)} tasks.")


if __name__ == "__main__":
    main()
