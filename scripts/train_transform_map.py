"""Generate a mapping task_id -> (kind, param) based on training examples.

This serves as a lightweight "model" under 100 MB that can be shipped to
Kaggle offline.  It relies on the same heuristic transformation detection
as used in `arc_submission.py`.

Usage:
    python scripts/train_transform_map.py --train_dir data/training \
        --output weights/transform_map.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from hx_teachers.rotation import RotationTeacher
from hx_teachers.mirror import MirrorTeacher

# ---------------------------------------------------------------------
# Re-use helpers from arc_submission ----------------------------------
# ---------------------------------------------------------------------


def _detect_transform(examples: List[tuple[np.ndarray, np.ndarray]]):  # noqa: D401
    """Infer simple transformation (rotation, mirror, inverse, identity)."""
    angle = RotationTeacher.discover_angle(examples)
    if angle is not None:
        return ("rotation", angle)
    axis = MirrorTeacher.discover_axis(examples)
    if axis is not None:
        return ("mirror", axis)
    inv_binary = all(np.array_equal(out, (inp == 0).astype(int)) for inp, out in examples)
    if inv_binary:
        return ("inverse-binary", None)
    inv_colour = all(np.array_equal(out, 9 - inp) for inp, out in examples)
    if inv_colour:
        return ("inverse-color", None)
    return ("identity", None)


# ---------------------------------------------------------------------
# CLI -----------------------------------------------------------------
# ---------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mapping: Dict[str, Any] = {}
    for json_file in sorted(args.train_dir.glob("*.json")):
        task_id = json_file.stem
        task = json.loads(json_file.read_text())
        train_examples = [
            (np.array(p["input"], dtype=int), np.array(p["output"], dtype=int))
            for p in task["train"]
        ]
        mapping[task_id] = _detect_transform(train_examples)

    # Save mapping ----------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping))
    size_mb = args.output.stat().st_size / 1_048_576
    print(f"Saved mapping for {len(mapping)} tasks to {args.output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
