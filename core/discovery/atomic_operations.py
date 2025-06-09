"""
Atomic operations for grid transformations.
"""

import numpy as np
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("EAN.Operations")


@dataclass
class AtomicOperation:
    """
    Represents an atomic transformation operation on a grid.
    
    Supported operations:
    - position_mapping: Maps positions to new positions
    - rotation: Rotates the grid
    - reflection: Reflects the grid
    - translation: Translates objects
    - color_change: Changes object values
    """
    
    name: str
    params: Tuple[Any, ...]
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply the operation to a grid.
        
        Args:
            grid: Input grid
            
        Returns:
            Transformed grid
        """
        if self.name == "position_mapping":
            return self._apply_position_mapping(grid)
        elif self.name == "rotation":
            return self._apply_rotation(grid)
        elif self.name == "reflection":
            return self._apply_reflection(grid)
        elif self.name == "translation":
            return self._apply_translation(grid)
        elif self.name == "color_change":
            return self._apply_color_change(grid)
        else:
            logger.warning(f"Unknown operation: {self.name}")
            return grid.copy()
    
    def _apply_position_mapping(self, grid: np.ndarray) -> np.ndarray:
        """Apply position mapping transformation."""
        mappings = self.params[0]  # Dict[(i,j)] -> List[(i,j)]
        result = np.zeros_like(grid)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0:
                    source = (i, j)
                    if source in mappings:
                        for target in mappings[source]:
                            if (0 <= target[0] < grid.shape[0] and 
                                0 <= target[1] < grid.shape[1]):
                                result[target] = grid[i, j]
        
        return result
    
    def _apply_rotation(self, grid: np.ndarray) -> np.ndarray:
        """Apply rotation transformation."""
        angle = self.params[0]  # 90, 180, 270 degrees
        k = angle // 90
        return np.rot90(grid, k=k)
    
    def _apply_reflection(self, grid: np.ndarray) -> np.ndarray:
        """Apply reflection transformation."""
        axis = self.params[0]  # 'horizontal' or 'vertical'
        if axis == 'horizontal':
            return np.flipud(grid)
        elif axis == 'vertical':
            return np.fliplr(grid)
        else:
            return grid.copy()
    
    def _apply_translation(self, grid: np.ndarray) -> np.ndarray:
        """Apply translation transformation."""
        dx, dy = self.params[0], self.params[1]
        result = np.zeros_like(grid)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0:
                    new_i, new_j = i + dx, j + dy
                    if (0 <= new_i < grid.shape[0] and 
                        0 <= new_j < grid.shape[1]):
                        result[new_i, new_j] = grid[i, j]
        
        return result
    
    def _apply_color_change(self, grid: np.ndarray) -> np.ndarray:
        """Apply color/value change transformation."""
        color_map = self.params[0]  # Dict[old_value] -> new_value
        result = grid.copy()
        
        for old_val, new_val in color_map.items():
            result[grid == old_val] = new_val
        
        return result
    
    def can_apply_to(self, grid: np.ndarray) -> bool:
        """
        Check if this operation can be applied to the given grid.
        
        Args:
            grid: Input grid
            
        Returns:
            bool: True if operation can be applied
        """
        if self.name == "position_mapping":
            mappings = self.params[0]
            # Check if any active pixels have mapping rules
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i, j] != 0 and (i, j) in mappings:
                        return True
            return False
        
        # Most operations can always be applied
        return True
    
    def inverse(self) -> 'AtomicOperation':
        """
        Get the inverse operation if possible.
        
        Returns:
            AtomicOperation: Inverse operation
        """
        if self.name == "rotation":
            angle = self.params[0]
            inverse_angle = (360 - angle) % 360
            return AtomicOperation("rotation", (inverse_angle,))
        
        elif self.name == "reflection":
            # Reflection is its own inverse
            return AtomicOperation(self.name, self.params)
        
        elif self.name == "translation":
            dx, dy = self.params[0], self.params[1]
            return AtomicOperation("translation", (-dx, -dy))
        
        elif self.name == "position_mapping":
            # Create inverse mapping
            forward_mappings = self.params[0]
            inverse_mappings = {}
            for source, targets in forward_mappings.items():
                for target in targets:
                    if target not in inverse_mappings:
                        inverse_mappings[target] = []
                    inverse_mappings[target].append(source)
            return AtomicOperation("position_mapping", (inverse_mappings,))
        
        elif self.name == "color_change":
            # Create inverse color mapping
            forward_map = self.params[0]
            inverse_map = {v: k for k, v in forward_map.items()}
            return AtomicOperation("color_change", (inverse_map,))
        
        # If no inverse exists, return identity
        return AtomicOperation("identity", ())
    
    def compose_with(self, other: 'AtomicOperation') -> 'AtomicOperation':
        """
        Compose this operation with another.
        
        Args:
            other: Operation to compose with
            
        Returns:
            AtomicOperation: Composed operation
        """
        # Simple composition for same type operations
        if self.name == other.name == "rotation":
            total_angle = (self.params[0] + other.params[0]) % 360
            return AtomicOperation("rotation", (total_angle,))
        
        if self.name == other.name == "translation":
            total_dx = self.params[0] + other.params[0]
            total_dy = self.params[1] + other.params[1]
            return AtomicOperation("translation", (total_dx, total_dy))
        
        # For different operations, return a composite operation
        return AtomicOperation("composite", (self, other))
    
    def __repr__(self) -> str:
        if self.name == "position_mapping":
            return f"PositionMapping({len(self.params[0])} rules)"
        elif self.name == "rotation":
            return f"Rotation({self.params[0]}°)"
        elif self.name == "reflection":
            return f"Reflection({self.params[0]})"
        elif self.name == "translation":
            return f"Translation(dx={self.params[0]}, dy={self.params[1]})"
        elif self.name == "color_change":
            return f"ColorChange({len(self.params[0])} mappings)"
        else:
            return f"{self.name}{self.params}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'params': self.params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AtomicOperation':
        """Create from dictionary."""
        return cls(name=data['name'], params=tuple(data['params']))