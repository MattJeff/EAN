"""Spike-rate homeostasis helper.

`SpikeRateController` dynamically adjusts neuron firing thresholds to steer
 the population towards a *target firing rate* expressed as fraction of
 neurons spiking per tick (e.g. ``0.1`` ⇒ 10 % of neurons fire on average).

It is *pure* and task-agnostic: any caller can feed it the list of neurons and
 the set of neurons that fired during the last tick.  The controller maintains
 an exponential moving average (EMA) of the spike rate and nudges each
 neuron's threshold up/down by a small factor proportional to the deviation.
"""
from __future__ import annotations

from typing import Iterable, Set

from hx_core.neuron import Neuron

__all__ = ["SpikeRateController"]


class SpikeRateController:  # pylint: disable=too-few-public-methods
    """Maintain population spike-rate around a target value."""

    def __init__(  # noqa: D401
        self,
        target_rate: float = 0.1,
        ema_alpha: float = 0.05,
        lr: float = 0.1,
        min_threshold: float = 0.1,
        max_threshold: float = 10.0,
    ) -> None:
        if not 0.0 < target_rate < 1.0:
            raise ValueError("target_rate must be in (0,1)")
        self.target_rate = float(target_rate)
        self.ema_alpha = float(ema_alpha)
        self.lr = float(lr)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)

        self._ema_rate: float = 0.0

    # ------------------------------------------------------------------
    def update(self, *, neurons: Iterable[Neuron], fired: Set[Neuron]) -> None:  # noqa: D401
        """Update EMA and adjust thresholds in-place.

        Parameters
        ----------
        neurons : Iterable[Neuron]
            Population to regulate.
        fired : Set[Neuron]
            Subset of *neurons* that spiked during the last tick.
        """
        neurons = list(neurons)  # may be generator
        if not neurons:
            return
        current_rate = len(fired) / len(neurons)
        # EMA update for monitoring
        self._ema_rate = (
            (1 - self.ema_alpha) * self._ema_rate + self.ema_alpha * current_rate
        )
        # Compute deviation using *instant* rate for responsiveness
        diff = current_rate - self.target_rate
        if abs(diff) < 1e-3:
            return  # close enough
        # Direction: >0 ⇒ too much activity ⇒ increase threshold, else decrease
        scale = 1.0 + self.lr * diff  # diff may be negative
        for n in neurons:
            new_thr = float(n.threshold * scale)
            n.threshold = float(
                max(self.min_threshold, min(self.max_threshold, new_thr))
            )

    # ------------------------------------------------------------------
    @property
    def ema_rate(self) -> float:  # noqa: D401
        """Return the current exponential moving average of spike rate."""
        return self._ema_rate
