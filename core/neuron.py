"""
Neuron implementation for the EAN architecture.
This represents the most advanced version (PMV - Peak Performance Version).
"""

import math
import random
from typing import Set, Tuple, Optional, Dict
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NeuronPMV:
    """
    Advanced neuron implementation with enhanced capabilities.
    
    PMV (Peak Performance Version) includes:
    - Spatial positioning for local connectivity
    - Energy-based activation
    - Assembly membership tracking
    - Synaptic connections with plasticity
    - Spike timing for STDP
    """
    
    id: int
    position: Tuple[float, float]
    energy: float = field(default_factory=lambda: random.uniform(40.0, 60.0))
    assembly_membership: Optional[str] = None
    neighboring_neurons: Set[int] = field(default_factory=set)
    
    # Advanced features
    synapses: Dict[int, float] = field(default_factory=dict)  # neuron_id -> weight
    spike_history: list = field(default_factory=list)  # List of spike times
    activation_threshold: float = 50.0
    refractory_period: float = 0.5
    last_spike_time: float = -1.0
    neuron_type: str = 'excitatory'  # 'excitatory' or 'inhibitory'
    
    # Plasticity parameters
    tau_plus: float = 20.0  # STDP time constant for potentiation
    tau_minus: float = 20.0  # STDP time constant for depression
    a_plus: float = 0.005  # Learning rate for potentiation
    a_minus: float = 0.00525  # Learning rate for depression
    
    def __post_init__(self):
        """Initialize synaptic weights for neighbors."""
        for neighbor_id in self.neighboring_neurons:
            if neighbor_id not in self.synapses:
                # Initialize with small random weights
                self.synapses[neighbor_id] = random.uniform(0.1, 0.3)
    
    def activate(self, input_energy: float, current_time: float) -> bool:
        """
        Activate the neuron with given energy input.
        
        Args:
            input_energy: Energy received from inputs
            current_time: Current simulation time
            
        Returns:
            bool: True if neuron fired (spike), False otherwise
        """
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return False
        
        # Accumulate energy
        self.energy = min(100.0, self.energy + input_energy)
        
        # Fire if above threshold
        if self.energy >= self.activation_threshold:
            self.fire(current_time)
            return True
        
        return False
    
    def fire(self, current_time: float):
        """
        Fire the neuron (generate spike).
        
        Args:
            current_time: Current simulation time
        """
        self.last_spike_time = current_time
        self.spike_history.append(current_time)
        
        # Keep only recent spike history (last 100 spikes)
        if len(self.spike_history) > 100:
            self.spike_history = self.spike_history[-100:]
        
        # Reset energy after firing
        self.energy = 20.0
    
    def decay(self, decay_rate: float):
        """
        Apply energy decay.
        
        Args:
            decay_rate: Rate of energy decay
        """
        self.energy = max(0.0, self.energy - decay_rate)
    
    def is_available_for_assembly(self) -> bool:
        """
        Check if neuron is available to join a new assembly.
        
        Returns:
            bool: True if available, False otherwise
        """
        return (self.energy > 30.0 and 
                self.assembly_membership is None)
    
    def calculate_distance_to(self, other_position: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance to another position.
        
        Args:
            other_position: Target position (x, y)
            
        Returns:
            float: Euclidean distance
        """
        return math.sqrt(
            (self.position[0] - other_position[0])**2 + 
            (self.position[1] - other_position[1])**2
        )
    
    def update_synapse_stdp(self, pre_neuron_id: int, pre_spike_time: float, 
                           post_spike_time: float):
        """
        Update synapse weight using Spike-Timing Dependent Plasticity (STDP).
        
        Args:
            pre_neuron_id: ID of presynaptic neuron
            pre_spike_time: Time of presynaptic spike
            post_spike_time: Time of postsynaptic spike
        """
        if pre_neuron_id not in self.synapses:
            return
        
        delta_t = post_spike_time - pre_spike_time
        
        if delta_t > 0:
            # Pre before post: potentiation
            delta_w = self.a_plus * math.exp(-delta_t / self.tau_plus)
        else:
            # Post before pre: depression
            delta_w = -self.a_minus * math.exp(delta_t / self.tau_minus)
        
        # Update weight with bounds
        old_weight = self.synapses[pre_neuron_id]
        new_weight = np.clip(old_weight + delta_w, 0.0, 1.0)
        self.synapses[pre_neuron_id] = new_weight
    
    def propagate_signal(self, target_neurons: Dict[int, 'NeuronPMV'], 
                        current_time: float) -> Dict[int, float]:
        """
        Propagate signal to connected neurons.
        
        Args:
            target_neurons: Dictionary of potential target neurons
            current_time: Current simulation time
            
        Returns:
            Dict[int, float]: Neuron IDs and energy transmitted
        """
        propagated_signals = {}
        
        for neuron_id, weight in self.synapses.items():
            if neuron_id in target_neurons:
                # Calculate signal strength based on weight and neuron type
                if self.neuron_type == 'excitatory':
                    signal_strength = self.energy * weight * 0.5
                else:  # inhibitory
                    signal_strength = -self.energy * weight * 0.3
                
                propagated_signals[neuron_id] = signal_strength
        
        return propagated_signals
    
    def join_assembly(self, assembly_id: str):
        """
        Join an assembly.
        
        Args:
            assembly_id: ID of the assembly to join
        """
        self.assembly_membership = assembly_id
        # Boost energy when joining assembly
        self.energy = min(100.0, self.energy + 15.0)
    
    def leave_assembly(self):
        """Leave current assembly."""
        self.assembly_membership = None
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state as a vector for analysis.
        
        Returns:
            np.ndarray: State vector
        """
        return np.array([
            self.energy,
            self.position[0],
            self.position[1],
            float(self.assembly_membership is not None),
            len(self.spike_history),
            self.last_spike_time
        ])
    
    def __repr__(self) -> str:
        return (f"NeuronPMV(id={self.id}, pos={self.position}, "
                f"energy={self.energy:.1f}, assembly={self.assembly_membership})")