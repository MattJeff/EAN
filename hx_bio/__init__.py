"""Biologically inspired extensions for HyperCubeX.

This package contains optional advanced components (neurons, synapses, etc.)
that can be plugged into the task-agnostic *hx_core* engine when richer
biophysical realism is required.  Importing these modules is **optional** and
introduces **no** additional dependency for the basic system.
"""
from __future__ import annotations

from .neuron_adv import NeuronAdvanced  # noqa: F401
from .homeostasis import SpikeRateController  # noqa: F401
from .inhibitory import NeuronInhibitory  # noqa: F401

__all__ = ["NeuronAdvanced", "SpikeRateController", "NeuronInhibitory"]
