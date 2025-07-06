"""End-to-end check of arc_submission.predict_task structure."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import arc_submission


def _make_mock_task():
    # Identity task: output == input
    grid = [[0, 1], [2, 3]]
    train_pair = {"input": grid, "output": grid}
    test_pair = {"input": grid}
    return {"train": [train_pair], "test": [test_pair]}


def test_predict_task_structure():  # noqa: D401
    task = _make_mock_task()
    preds = arc_submission.predict_task(task)
    # One test pair -> list of length 1
    assert len(preds) == 1
    attempt1, attempt2 = preds[0]
    # Both attempts are numpy arrays same shape as input
    inp = np.array(task["test"][0]["input"], dtype=int)
    assert attempt1.shape == inp.shape
    assert attempt2.shape == inp.shape
    # Values within 0-9
    assert int(attempt1.min()) >= 0 and int(attempt1.max()) <= 9
    assert int(attempt2.min()) >= 0 and int(attempt2.max()) <= 9


def test_cli_submission_file(tmp_path: Path):  # noqa: D401
    challenges_path = tmp_path / "challenges.json"
    submission_path = tmp_path / "submission.json"
    challenges_data = {"mock_task": _make_mock_task()}
    challenges_path.write_text(json.dumps(challenges_data))

    # Call arc_submission.main via patched argv
    import sys

    argv_backup = sys.argv.copy()
    sys.argv = [
        "arc_submission.py",
        "--challenges",
        str(challenges_path),
        "--output",
        str(submission_path),
    ]
    try:
        arc_submission.main()
    finally:
        sys.argv = argv_backup

    # Check file produced
    assert submission_path.exists()
    sub = json.loads(submission_path.read_text())
    assert "mock_task" in sub
    assert isinstance(sub["mock_task"], list)
    assert len(sub["mock_task"]) == 1
    entry = sub["mock_task"][0]
    assert "attempt_1" in entry and "attempt_2" in entry
    assert len(entry["attempt_1"]) == len(entry["attempt_2"]) == 2
