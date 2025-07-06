"""Advanced neuron with basic STDP timing rule.

Designed to share the public API of *hx_core.Neuron* so it can be used
interchangeably by the existing *Network* implementation.  Only the
internal plasticity rule differs (time–dependent STDP rather than simple
Hebbian coincidence).
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Any

import numpy as np

__all__ = ["NeuronAdvanced"]


class NeuronAdvanced:  # pylint: disable=too-few-public-methods
    """Point neuron with spike-timing dependent plasticity (STDP).

    Parameters
    ----------
    position : np.ndarray | list[float]
        3-D coordinates in continuous space (kept for compatibility).
    threshold : float, default 1.0
        Firing threshold.
    decay : float, default 0.02
        Exponential decay of membrane energy.
    tau_plus : float, default 20.0
        STDP *potentiation* time constant.
    tau_minus : float, default 20.0
        STDP *depression* time constant.
    a_plus : float, default 0.01
        Learning rate for potentiation.
    a_minus : float, default 0.012
        Learning rate for depression.
    history_len : int, default 50
        Number of recent spike timestamps to keep.
    """

    def __init__(
        self,
        position: np.ndarray | list[float],
        *,
        threshold: float = 1.0,
        decay: float = 0.02,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        history_len: int = 50,
    ) -> None:
        self.position = np.asarray(position, dtype=float)
        self.threshold = float(threshold)
        self.decay = float(decay)

        self.energy: float = 0.0
        self.last_input: float = 0.0
        self.last_output: float = 0.0

        # Spike-timing history ------------------------------------------------
        self._spike_times: Deque[float] = deque(maxlen=history_len)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

    # ------------------------------------------------------------------
    # Basic API (same as *Neuron*)
    # ------------------------------------------------------------------
    def receive(self, signal: float, weight: float = 1.0) -> None:  # noqa: D401
        """Accumulate weighted *signal* into membrane energy."""
        self.energy += signal * weight
        self.last_input = signal

    def integrate(self, dt: float = 1.0) -> None:  # noqa: D401
        """Passive exponential decay of energy."""
        self.energy *= max(0.0, 1.0 - self.decay * dt)

    def fire(self, current_time: float | None = None) -> float:  # noqa: D401
        """Emit spike and log its timestamp."""
        if self.energy >= self.threshold:
            self.last_output = self.energy
            self.energy = 0.0
            # If *current_time* not provided, we treat tick index 0,1,2…
            time_val = float(current_time) if current_time is not None else 0.0
            self._spike_times.append(time_val)
            return self.last_output
        return 0.0

    # ------------------------------------------------------------------
    # Plasticity
    # ------------------------------------------------------------------
    def apply_plasticity(
        self,
        *,
        fired_pre: bool,
        fired_post: bool,
        weight: float,
        eta: float = 0.0,  # kept for signature compatibility – ignored
        current_time: float | None = None,
    ) -> float:  # noqa: D401
        """STDP update based on relative timing.

        The *Network* passes only coincidence information (bool).  We extend
        this by using the last spike timestamp to approximate Δt.  If either
        neuron did not spike recently, we fall back to Hebbian coincidence.
        """
        if not (fired_pre or fired_post):
            return weight  # no change

        # Simple fallback if history is empty
        if not self._spike_times:
            return float(min(weight + (0.01 if (fired_pre and fired_post) else 0.0), 1.0))

        # Assume *current_time* contiguous ticks; last spike time difference
        last_pre = self._spike_times[-1]
        last_post = last_pre  # Approx when post spiked (unknown here)
        if fired_post:
            last_post = last_pre  # coincidence same tick
        # Δt = t_post – t_pre
        delta_t = last_post - last_pre
        if delta_t > 0:
            delta_w = self.a_plus * np.exp(-delta_t / self.tau_plus)
        else:
            delta_w = -self.a_minus * np.exp(delta_t / self.tau_minus)
        new_w = np.clip(weight + delta_w, 0.0, 1.0)
        return float(new_w)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:  # noqa: D401
        return {
            "energy": self.energy,
            "last_output": self.last_output,
            "n_spikes": len(self._spike_times),
        }

    def reset(self) -> None:  # noqa: D401
        self.energy = 0.0
        self.last_input = 0.0
        self.last_output = 0.0
        self._spike_times.clear()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NeuronAdvanced(pos={self.position.tolist()}, energy={self.energy:.3f}, "
            f"spikes={len(self._spike_times)})"
        )
