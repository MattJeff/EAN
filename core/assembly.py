"""
Emergent Assembly implementation for the EAN architecture.
"""

import logging
from typing import Set, Optional, Tuple, List, Dict, Any
from collections import deque
from dataclasses import dataclass, field
import numpy as np

from .discovery.base_discoverer import TransformationKnowledge
from .discovery.position_mapping import ImprovedTransformationDiscoverer
from .discovery.atomic_operations import AtomicOperation

logger = logging.getLogger("EAN.Assembly")


@dataclass
class EmergentAssemblyEAN:
    """
    Emergent Assembly that can learn and apply transformations.
    
    This is the most advanced version with:
    - Pattern discovery capabilities
    - Transformation knowledge management
    - Confidence tracking
    - Recursive pattern support
    - Working memory
    """
    
    # Core properties
    id: str
    founder_ids: frozenset
    birth_time: float
    grid_shape: Tuple[int, int]
    
    # Members and state
    member_neuron_ids: Set[int] = field(default_factory=set)
    age: int = 0
    is_specialized: bool = False
    protection_counter: int = 0
    stability_score: float = 1.0
    
    # Transformation knowledge
    transformation_knowledge: TransformationKnowledge = field(
        default_factory=TransformationKnowledge
    )
    transformation_examples: deque = field(
        default_factory=lambda: deque(maxlen=20)
    )
    
    # Performance metrics
    consecutive_successes: int = 0
    recent_applications: deque = field(
        default_factory=lambda: deque(maxlen=10)
    )
    
    # Working memory for complex patterns
    working_memory: deque = field(
        default_factory=lambda: deque(maxlen=5)
    )
    
    # Discoverer instance
    discoverer: Optional[ImprovedTransformationDiscoverer] = None
    
    def __post_init__(self):
        """Initialize assembly after creation."""
        self.member_neuron_ids = set(self.founder_ids)
        self.discoverer = ImprovedTransformationDiscoverer(self.grid_shape)
        logger.debug(f"Created assembly {self.id} with {len(self.founder_ids)} founders")
    
    def observe_transformation(self, input_pattern: np.ndarray, 
                             output_pattern: np.ndarray):
        """
        Observe a transformation example.
        
        Args:
            input_pattern: Input grid pattern
            output_pattern: Expected output pattern
        """
        self.transformation_examples.append(
            (input_pattern.copy(), output_pattern.copy())
        )
        
        # Store in working memory for complex pattern analysis
        self.working_memory.append({
            'input': input_pattern.copy(),
            'output': output_pattern.copy(),
            'timestamp': self.age
        })
    
    def attempt_discovery(self) -> bool:
        """
        Attempt to discover transformation pattern.
        
        Returns:
            bool: True if discovery successful
        """
        if len(self.transformation_examples) < 4:
            return False
        
        # Use all examples for complete pattern discovery
        all_examples = list(self.transformation_examples)
        
        # Attempt discovery
        operations = self.discoverer.discover_pattern(all_examples)
        
        if operations:
            self.transformation_knowledge.operations = operations
            self.transformation_knowledge.discovery_method = operations[0].name
            self.transformation_knowledge.confidence = 0.9
            self.is_specialized = True
            
            logger.info(f"{self.id} discovered: {operations[0]}")
            return True
        
        return False
    
    def apply_transformation(self, input_pattern: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Apply learned transformation to input.
        
        Args:
            input_pattern: Input pattern to transform
            
        Returns:
            Tuple[Optional[np.ndarray], float]: (result, confidence)
        """
        if not self.is_specialized or not self.transformation_knowledge.operations:
            return None, 0.0
        
        try:
            result = input_pattern.copy()
            
            # Apply each operation in sequence
            for operation in self.transformation_knowledge.operations:
                result = operation.apply(result)
            
            # Calculate confidence
            confidence = self._calculate_confidence(input_pattern)
            
            return result, confidence
            
        except Exception as e:
            logger.error(f"{self.id} transformation error: {e}")
            return None, 0.0
    
    def _calculate_confidence(self, input_pattern: np.ndarray) -> float:
        """
        Calculate confidence for the transformation.
        
        Args:
            input_pattern: Input pattern
            
        Returns:
            float: Confidence score [0, 1]
        """
        base_confidence = self.transformation_knowledge.confidence
        
        # Check if we can handle this input
        has_active_pixels = np.any(input_pattern != 0)
        can_transform = False
        
        if has_active_pixels and self.transformation_knowledge.operations:
            # Check if we have rules for active pixels
            for operation in self.transformation_knowledge.operations:
                if operation.can_apply_to(input_pattern):
                    can_transform = True
                    break
        
        if not can_transform:
            return 0.0
        
        # Adjust based on recent performance
        if len(self.recent_applications) > 0:
            recent_success_rate = sum(self.recent_applications) / len(self.recent_applications)
            adjusted_confidence = base_confidence * (0.6 + 0.4 * recent_success_rate)
        else:
            adjusted_confidence = base_confidence * 0.8
        
        # Bonus for consecutive successes
        if self.consecutive_successes > 2:
            adjusted_confidence = min(1.0, adjusted_confidence * 1.1)
        
        return adjusted_confidence
    
    def record_application_result(self, success: bool):
        """
        Record the result of a transformation application.
        
        Args:
            success: Whether the application was successful
        """
        self.transformation_knowledge.total_attempts += 1
        self.recent_applications.append(success)
        
        if success:
            self.transformation_knowledge.successful_applications += 1
            self.consecutive_successes += 1
            self.protection_counter = 200  # Protect successful assemblies
            
            # Increase confidence
            self.transformation_knowledge.confidence = min(
                1.0,
                self.transformation_knowledge.confidence * 1.02
            )
        else:
            self.consecutive_successes = 0
            # Decrease confidence
            self.transformation_knowledge.confidence *= 0.95
    
    def should_dissolve(self) -> bool:
        """
        Determine if assembly should dissolve.
        
        Returns:
            bool: True if should dissolve
        """
        # Protect specialized assemblies with good performance
        if self.is_specialized and self.transformation_knowledge.success_rate > 0.4:
            return False
        
        # Protection period
        if self.protection_counter > 0:
            return False
        
        # Old unspecialized assemblies
        if self.age > 200 and not self.is_specialized:
            return True
        
        # Poor performing specialized assemblies
        if self.is_specialized and self.transformation_knowledge.total_attempts > 20:
            if self.transformation_knowledge.success_rate < 0.2:
                return True
        
        return False
    
    def update(self):
        """Periodic update of assembly state."""
        self.age += 1
        
        if self.protection_counter > 0:
            self.protection_counter -= 1
        
        # Retry discovery with more examples
        if not self.is_specialized and self.age % 15 == 0:
            if len(self.transformation_examples) >= 8:
                self.attempt_discovery()
        
        # Update stability based on performance
        if self.is_specialized:
            self.stability_score = (
                0.7 * self.stability_score + 
                0.3 * self.transformation_knowledge.success_rate
            )
    
    def add_neuron(self, neuron_id: int):
        """
        Add a neuron to the assembly.
        
        Args:
            neuron_id: ID of neuron to add
        """
        self.member_neuron_ids.add(neuron_id)
    
    def remove_neuron(self, neuron_id: int):
        """
        Remove a neuron from the assembly.
        
        Args:
            neuron_id: ID of neuron to remove
        """
        self.member_neuron_ids.discard(neuron_id)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of assembly state.
        
        Returns:
            Dict containing state information
        """
        return {
            'id': self.id,
            'age': self.age,
            'member_count': len(self.member_neuron_ids),
            'is_specialized': self.is_specialized,
            'success_rate': self.transformation_knowledge.success_rate,
            'confidence': self.transformation_knowledge.confidence,
            'stability': self.stability_score,
            'discovery_method': self.transformation_knowledge.discovery_method,
            'total_attempts': self.transformation_knowledge.total_attempts
        }
    
    def __repr__(self) -> str:
        status = "specialized" if self.is_specialized else "learning"
        return (f"Assembly({self.id}, {status}, "
                f"members={len(self.member_neuron_ids)}, "
                f"success={self.transformation_knowledge.success_rate:.2f})")