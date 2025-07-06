"""Backend abstraction layer – CPU vs future GPU back-ends.

This module introduces a *very thin* indirection so that the rest of
`hx_core` can stay agnostic of the numerical implementation.  For now we
only provide a reference *CPUBackend* that keeps the exact semantics of
our existing Python/Numpy loops.  A *TorchGPUBackend* stub illustrates
how a Metal-/CUDA-accelerated version could be plugged later (e.g. on an
Apple M-series MacBook via `torch.device("mps")`).

The design goal is *zero* overhead when using the default backend while
allowing more optimised paths to be introduced incrementally.
"""
from __future__ import annotations

from typing import Dict, List, Protocol, Type

import numpy as np

# -----------------------------------------------------------------------------
# Public interface ----------------------------------------------------------------
# -----------------------------------------------------------------------------
class _NeuronProtocol(Protocol):
    """Minimal subset of `hx_core.neuron.Neuron` used by the backend."""

    energy: float
    threshold: float
    decay: float

    def integrate(self, dt: float = 1.0) -> None:  # noqa: D401
        ...

    def fire(self) -> float:  # noqa: D401
        ...


class Backend(Protocol):  # pylint: disable=too-few-public-methods
    """Backend contract.

    Backends are stateless utility classes; no per-instance data is kept
    so they can be shared across networks.
    """

    @staticmethod
    def integrate(assembly: "Any", dt: float = 1.0) -> None:  # noqa: D401
        ...

    @staticmethod
    def fire(assembly: "Any") -> Dict["Any", float]:  # noqa: D401
        ...


# Registry to allow dynamic selection ------------------------------------------------
_BACKENDS: Dict[str, Type[Backend]] = {}


def select_backend() -> str:  # noqa: D401
    """Return backend name from env var *EAN_BACKEND* or default *cpu*."""
    import os

    return os.getenv("EAN_BACKEND", "cpu").lower()

def register_backend(name: str):  # noqa: D401
    """Decorator to register a backend implementation under *name*."""

    def _decorator(cls):
        _BACKENDS[name.lower()] = cls
        return cls

    return _decorator


def get_backend(name: str | None = None) -> Type[Backend]:  # noqa: D401
    if name is None:
        return _BACKENDS["cpu"]
    key = name.lower()
    if key not in _BACKENDS:
        raise KeyError(f"Unknown backend: {name}")
    return _BACKENDS[key]


# -----------------------------------------------------------------------------
# CPU backend – reference implementation ---------------------------------------
# -----------------------------------------------------------------------------
@register_backend("cpu")
class CPUBackend:  # pylint: disable=too-few-public-methods
    """Original pure-Python behaviour (baseline)."""

    @staticmethod
    def integrate(assembly, dt: float = 1.0) -> None:  # noqa: D401
        for n in assembly.neurons:  # type: ignore[attr-defined]
            n.integrate(dt)

    @staticmethod
    def fire(assembly) -> Dict["Any", float]:  # noqa: D401
        outputs: Dict["Any", float] = {}
        for n in assembly.neurons:  # type: ignore[attr-defined]
            out = n.fire()
            if out:
                outputs[n] = out
        return outputs


# -----------------------------------------------------------------------------
# Torch backend – placeholder ---------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import torch  # type: ignore

    @register_backend("torch")  # noqa: D401
    class TorchGPUBackend:  # pylint: disable=too-few-public-methods
        """Vectorised implementation using Torch tensors (CPU/GPU/MPS).

        This is *not* wired into the rest of the engine yet – it only
        shows the intended API so future work can focus on replacing the
        loops by batched tensor ops without touching high-level logic.
        """

        @staticmethod
        def integrate(assembly, dt: float = 1.0) -> None:  # noqa: D401
            # Lazy initialisation of tensor views (energies, decay, thresholds)
            if not hasattr(assembly, "_tensor_energy"):
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                )
                assembly._device = device  # type: ignore[attr-defined]

                energy = torch.tensor([n.energy for n in assembly.neurons], dtype=torch.float32, device=device)
                decay = torch.tensor([n.decay for n in assembly.neurons], dtype=torch.float32, device=device)
                threshold = torch.tensor([n.threshold for n in assembly.neurons], dtype=torch.float32, device=device)
                assembly._tensor_energy = energy  # type: ignore[attr-defined]
                assembly._tensor_decay = decay  # type: ignore[attr-defined]
                assembly._tensor_threshold = threshold  # type: ignore[attr-defined]
            # E_t+1 = (1 - decay*dt) * E_t
            assembly._tensor_energy.mul_(1.0 - assembly._tensor_decay * dt)
            # Reflect back to Neuron objects occasionally (cheap for small batches)
            for idx, n in enumerate(assembly.neurons):
                n.energy = float(assembly._tensor_energy[idx])

        @staticmethod
        def fire(assembly) -> Dict["Any", float]:  # noqa: D401
            outs: Dict["Any", float] = {}
            energies = assembly._tensor_energy  # type: ignore[attr-defined]
            thresholds = assembly._tensor_threshold  # type: ignore[attr-defined]
            mask = energies >= thresholds
            fired_idx = mask.nonzero(as_tuple=False).flatten()
            for idx in fired_idx.tolist():
                neuron = assembly.neurons[idx]
                spike_val = float(energies[idx])
                outs[neuron] = spike_val
                energies[idx] = 0.0
                neuron.energy = 0.0
            return outs

except ModuleNotFoundError:
    # Torch not available – skip registration
    pass
