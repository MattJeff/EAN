"""Smoke test for RecursiveTeacher."""
from __future__ import annotations

import numpy as np

from hx_teachers import RecursiveTeacher


def test_teacher_predict_shape():
    grid = np.random.randint(0, 5, size=(3, 3))
    teacher = RecursiveTeacher(grid.shape)
    out = teacher.predict(grid)
    assert out.shape == grid.shape
