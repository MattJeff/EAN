"""Unit tests for GroundTruthTeacher."""
from __future__ import annotations

import numpy as np

from hx_teachers import GroundTruthTeacher


def test_predict_returns_target_copy():
    tgt = np.array([[1, 0], [0, 1]])
    teacher = GroundTruthTeacher(tgt)
    out = teacher.predict(np.zeros_like(tgt))
    # Should equal but not be same object
    assert np.array_equal(out, tgt)
    assert out is not tgt


def test_reward_full_accuracy():
    tgt = np.random.randint(0, 10, size=(3, 3))
    teacher = GroundTruthTeacher(tgt, binary=False)
    score = teacher.reward(tgt)
    assert score == 1.0


def test_reward_partial_accuracy():
    tgt = np.array([[1, 2], [3, 4]])
    teacher = GroundTruthTeacher(tgt, binary=False)
    pred = np.array([[1, 2], [0, 0]])
    score = teacher.reward(pred)
    # 2 / 4 cells correct -> 0.5
    assert score == 0.5


def test_binary_mode():
    tgt = np.array([[0, 2], [0, 0]])  # one non-zero cell
    teacher = GroundTruthTeacher(tgt, binary=True)
    pred = np.array([[5, 0], [1, 0]])  # two non-zero cells at diff pos
    score = teacher.reward(pred)
    # only bottom-right cell matches -> 1/4 correct = 0.25
    assert np.isclose(score, 0.25)
