"""
Base classes for transformation discovery.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import logging

from .atomic_operations import AtomicOperation

logger = logging.getLogger("EAN.Discovery")


@dataclass
class TransformationKnowledge:
    """
    Knowledge about discovered transformations.
    
    Tracks:
    - The operations that implement the transformation
    - Confidence in the transformation
    - Performance metrics
    """
    
    operations: Optional[List[AtomicOperation]] = None
    confidence: float = 0.0
    discovery_method: str = ""
    successful_applications: int = 0
    total_attempts: int = 0
    
    # Additional metadata
    complexity: float = 0.0
    generalization_score: float = 0.0
    last_update_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of applications."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_applications / self.total_attempts
    
    @property
    def is_valid(self) -> bool:
        """Check if knowledge contains valid operations."""
        return self.operations is not None and len(self.operations) > 0
    
    def update_metrics(self, success: bool):
        """Update performance metrics."""
        self.total_attempts += 1
        if success:
            self.successful_applications += 1
            self.confidence = min(1.0, self.confidence * 1.05)
        else:
            self.confidence *= 0.95
    
    def calculate_complexity(self):
        """Calculate complexity of the transformation."""
        if not self.operations:
            return 0.0
        
        # Base complexity on number and type of operations
        complexity = 0.0
        for op in self.operations:
            if op.name == "position_mapping":
                complexity += len(op.params[0]) * 0.1
            elif op.name in ["rotation", "reflection"]:
                complexity += 0.3
            elif op.name == "translation":
                complexity += 0.2
            elif op.name == "composite":
                complexity += 0.5
            else:
                complexity += 0.4
        
        self.complexity = min(1.0, complexity)
        return self.complexity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operations': [op.to_dict() for op in self.operations] if self.operations else [],
            'confidence': self.confidence,
            'discovery_method': self.discovery_method,
            'successful_applications': self.successful_applications,
            'total_attempts': self.total_attempts,
            'complexity': self.complexity,
            'generalization_score': self.generalization_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransformationKnowledge':
        """Create from dictionary."""
        knowledge = cls()
        knowledge.operations = [
            AtomicOperation.from_dict(op_data) 
            for op_data in data.get('operations', [])
        ]
        knowledge.confidence = data.get('confidence', 0.0)
        knowledge.discovery_method = data.get('discovery_method', '')
        knowledge.successful_applications = data.get('successful_applications', 0)
        knowledge.total_attempts = data.get('total_attempts', 0)
        knowledge.complexity = data.get('complexity', 0.0)
        knowledge.generalization_score = data.get('generalization_score', 0.0)
        return knowledge


class TransformationDiscoverer(ABC):
    """
    Abstract base class for transformation discovery.
    
    Subclasses implement specific discovery strategies.
    """
    
    def __init__(self, grid_shape: Tuple[int, int]):
        """
        Initialize discoverer.
        
        Args:
            grid_shape: Shape of the grid (height, width)
        """
        self.grid_shape = grid_shape
        self.discovery_history = []
        
    @abstractmethod
    def discover_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[List[AtomicOperation]]:
        """
        Discover transformation pattern from examples.
        
        Args:
            examples: List of (input, output) pairs
            
        Returns:
            List of operations if pattern found, None otherwise
        """
        pass
    
    def analyze_transformation_type(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> str:
        """
        Analyze the type of transformation in examples.
        
        Args:
            examples: List of (input, output) pairs
            
        Returns:
            String describing transformation type
        """
        if not examples:
            return "unknown"
        
        # Check various transformation characteristics
        characteristics = []
        
        # Check if objects preserve values
        values_preserved = True
        positions_changed = False
        
        for inp, out in examples:
            inp_values = set(inp[inp != 0])
            out_values = set(out[out != 0])
            
            if inp_values != out_values:
                values_preserved = False
            
            # Check if positions changed
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if inp[i, j] != out[i, j]:
                        positions_changed = True
                        break
        
        if values_preserved and positions_changed:
            characteristics.append("spatial")
        elif not values_preserved:
            characteristics.append("value-changing")
        
        # Check for patterns
        if self._check_rotation(examples):
            characteristics.append("rotation")
        if self._check_reflection(examples):
            characteristics.append("reflection")
        if self._check_translation(examples):
            characteristics.append("translation")
        
        return "-".join(characteristics) if characteristics else "complex"
    
    def _check_rotation(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if examples represent rotation."""
        for inp, out in examples:
            # Try different rotation angles
            for k in [1, 2, 3]:  # 90, 180, 270 degrees
                if np.array_equal(np.rot90(inp, k=k), out):
                    return True
        return False
    
    def _check_reflection(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if examples represent reflection."""
        for inp, out in examples:
            if np.array_equal(np.flipud(inp), out) or np.array_equal(np.fliplr(inp), out):
                return True
        return False
    
    def _check_translation(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if examples represent translation."""
        for inp, out in examples:
            # Simple check: same values but different positions
            if np.sum(inp) == np.sum(out) and not np.array_equal(inp, out):
                # More detailed check would verify consistent displacement
                return True
        return False
    
    def validate_pattern(self, operations: List[AtomicOperation], 
                        examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Validate discovered pattern against examples.
        
        Args:
            operations: List of operations to validate
            examples: Test examples
            
        Returns:
            Validation score [0, 1]
        """
        if not operations or not examples:
            return 0.0
        
        correct = 0
        for inp, expected_out in examples:
            result = inp.copy()
            
            # Apply operations
            for op in operations:
                result = op.apply(result)
            
            if np.array_equal(result, expected_out):
                correct += 1
        
        return correct / len(examples)
    
    def find_minimal_operations(self, operations: List[AtomicOperation]) -> List[AtomicOperation]:
        """
        Find minimal set of operations that achieve the same result.
        
        Args:
            operations: List of operations
            
        Returns:
            Minimal list of operations
        """
        if len(operations) <= 1:
            return operations
        
        # Try to combine operations
        minimal = []
        i = 0
        
        while i < len(operations):
            if i + 1 < len(operations):
                # Try to compose adjacent operations
                composed = operations[i].compose_with(operations[i + 1])
                if composed.name != "composite":
                    # Successfully composed
                    minimal.append(composed)
                    i += 2
                    continue
            
            minimal.append(operations[i])
            i += 1
        
        # Recursively minimize if we made progress
        if len(minimal) < len(operations):
            return self.find_minimal_operations(minimal)
        
        return minimal