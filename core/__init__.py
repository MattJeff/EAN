"""
Core module for the Emergent Assembly Network (EAN) architecture.
"""

from .neuron import NeuronPMV
from .assembly import EmergentAssemblyEAN
from .network import IntegratedEAN

__all__ = [
    'NeuronPMV',
    'EmergentAssemblyEAN',
    'IntegratedEAN'
]

__version__ = '1.0.0'