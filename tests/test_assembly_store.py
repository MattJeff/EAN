"""Unit tests for AssemblyStore and GlobalWorkspace integration."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hx_core.assemblies import Assembly, AssemblyStore
from hx_core.workspace import GlobalWorkspace


def _make_mask(coords: list[tuple[int, int]], shape: tuple[int, int] = (5, 5)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for y, x in coords:
        mask[y, x] = True
    return mask


def test_add_and_trim() -> None:
    """Adding beyond capacity should keep strongest assemblies only."""
    store = AssemblyStore(max_size=3)
    for strength in [1.0, 2.0, 3.0, 4.0]:
        mask = _make_mask([(0, int(strength - 1))])
        store.add(Assembly.from_mask(mask, strength=strength))

    assert len(store) == 3
    strengths = [a.strength for a in store]
    # Should keep 4,3,2
    assert sorted(strengths, reverse=True) == [4.0, 3.0, 2.0]


def test_serialization_roundtrip(tmp_path: Path) -> None:
    """Store.save then load should preserve assemblies exactly."""
    store = AssemblyStore(max_size=5)
    for idx in range(3):
        mask = _make_mask([(idx, idx)])
        store.add(Assembly.from_mask(mask, strength=idx + 1))

    json_path = tmp_path / "store.json"
    store.save(json_path)

    store2 = AssemblyStore.load(json_path)
    assert len(store2) == len(store)
    for a1, a2 in zip(store, store2):
        assert a1.coords == a2.coords
        assert pytest.approx(a1.strength) == a2.strength


def test_workspace_injection() -> None:
    """inject_assemblies should convert assemblies into votes with given strength."""
    store = AssemblyStore()
    store.add(Assembly.from_mask(_make_mask([(0, 0)]), strength=0.8))
    store.add(Assembly.from_mask(_make_mask([(1, 1)]), strength=0.5))

    gw = GlobalWorkspace(grid_shape=(5, 5))
    gw.inject_assemblies(store)

    assert len(gw) == len(store)
    strengths = [v.confidence for v in gw.votes()]
    assert sorted(strengths, reverse=True) == [0.8, 0.5]
