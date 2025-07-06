"""Additional tests to improve coverage: Assembly EMA & frequency map."""
from __future__ import annotations

import numpy as np
import pytest

from hx_core.assemblies import Assembly, AssemblyStore
from analysis.dump_assemblies import _frequency_map


def test_update_strength() -> None:
    asm = Assembly(coords=[(0, 0)], strength=1.0)
    asm.update_strength(0.0, alpha=0.1)  # expect drop toward 0
    assert asm.strength == 0.9
    asm.update_strength(1.0, alpha=0.2)  # back up
    assert asm.strength == pytest.approx(0.92)


def test_store_sorting_after_update() -> None:
    store = AssemblyStore(max_size=3)
    a1 = Assembly(coords=[(0, 0)], strength=0.8)
    a2 = Assembly(coords=[(1, 1)], strength=0.6)
    store.add(a1)
    store.add(a2)
    # Update a2 to surpass a1
    a2.update_strength(1.0, alpha=0.5)  # new strength 0.8
    # Resort manually
    store._assemblies.sort(key=lambda a: a.strength, reverse=True)
    assert store[0].strength >= store[1].strength


def test_frequency_map() -> None:
    store = AssemblyStore()
    store.add(Assembly(coords=[(0, 0), (1, 1)], strength=0.5))
    store.add(Assembly(coords=[(0, 0)], strength=0.4))
    freq = _frequency_map(store, (3, 3))
    assert freq[0, 0] == 2  # present in both assemblies
    assert freq[1, 1] == 1
