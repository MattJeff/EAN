"""Teacher modules for curriculum / reinforcement learning.

Teachers are symbolic or heuristic solvers that can provide target outputs
or rewards to the emergent HyperCubeX network.  They live outside of the
minimal *hx_core* engine and can be imported only when supervision is needed.
"""
from __future__ import annotations

from .recursive import RecursiveTeacher  # noqa: F401
from .rotation import RotationTeacher  # noqa: F401
from .mirror import MirrorTeacher  # noqa: F401
from .inverse import InverseTeacher  # noqa: F401
from .ground_truth import GroundTruthTeacher  # noqa: F401

__all__ = ["RecursiveTeacher", "RotationTeacher", "MirrorTeacher", "InverseTeacher", "GroundTruthTeacher"]
