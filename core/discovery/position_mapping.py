"""
Position mapping discoverer for spatial transformations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging

from .base_discoverer import TransformationDiscoverer
from .atomic_operations import AtomicOperation

logger = logging.getLogger("EAN.PositionMapping")


class ImprovedTransformationDiscoverer(TransformationDiscoverer):
    """
    Discoverer specialized in position mapping transformations.
    
    This discoverer:
    - Extracts all position mappings from examples
    - Validates consistency across examples
    - Handles partial patterns
    - Supports multi-object transformations
    """
    
    def __init__(self, grid_shape: Tuple[int, int]):
        """Initialize the position mapping discoverer."""
        super().__init__(grid_shape)
        self.discovered_mappings = {}
        self.mapping_confidence = {}
        
    def discover_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[List[AtomicOperation]]:
        """
        Discover position mapping pattern from examples.
        
        Args:
            examples: List of (input, output) pairs
            
        Returns:
            List containing position mapping operation if found
        """
        position_mappings = self.extract_position_mappings(examples)
        
        if position_mappings:
            operation = AtomicOperation("position_mapping", (position_mappings,))
            
            # Validate the pattern
            validation_score = self.validate_pattern([operation], examples)
            if validation_score > 0.8:  # High confidence threshold
                logger.info(f"Discovered position mapping with {len(position_mappings)} rules")
                return [operation]
            else:
                logger.debug(f"Position mapping validation failed: {validation_score:.2f}")
        
        return None
    
    def extract_position_mappings(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict[Tuple[int, int], List[Tuple[int, int]]]]:
        """
        Extract ALL position mappings from examples.
        
        Args:
            examples: List of (input, output) pairs
            
        Returns:
            Dictionary mapping source positions to target positions
        """
        # Collect all observed mappings
        active_pixel_mappings = defaultdict(set)
        
        for inp, out in examples:
            # Find all active pixels and their mappings
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if inp[i, j] != 0:
                        source = (i, j)
                        value = inp[i, j]
                        
                        # Find ALL pixels with same value in output
                        targets = []
                        for ii in range(self.grid_shape[0]):
                            for jj in range(self.grid_shape[1]):
                                if out[ii, jj] == value:
                                    targets.append((ii, jj))
                        
                        if targets:
                            active_pixel_mappings[source].update(targets)
        
        # Validate consistency
        final_mappings = self._validate_mappings(active_pixel_mappings, examples)
        
        if len(final_mappings) > 0:
            logger.info(f"Extracted {len(final_mappings)} consistent mappings")
            self.discovered_mappings = final_mappings
            return final_mappings
        
        return None
    
    def _validate_mappings(self, raw_mappings: Dict[Tuple[int, int], Set[Tuple[int, int]]], 
                          examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Validate and filter mappings for consistency.
        
        Args:
            raw_mappings: Raw collected mappings
            examples: Original examples
            
        Returns:
            Validated mappings
        """
        final_mappings = {}
        
        for source, targets in raw_mappings.items():
            target_list = list(targets)
            is_consistent = True
            
            # Check consistency across all examples
            for inp, out in examples:
                if inp[source] != 0:
                    expected_value = inp[source]
                    
                    # Verify all targets have expected value
                    for target in target_list:
                        if out[target] != expected_value:
                            is_consistent = False
                            break
                    
                    if not is_consistent:
                        break
                    
                    # Verify no unexpected positions have this value
                    actual_targets = []
                    for ii in range(self.grid_shape[0]):
                        for jj in range(self.grid_shape[1]):
                            if out[ii, jj] == expected_value:
                                actual_targets.append((ii, jj))
                    
                    if set(actual_targets) != set(target_list):
                        is_consistent = False
                        break
            
            if is_consistent:
                final_mappings[source] = target_list
                self.mapping_confidence[source] = 1.0
            else:
                # Track partial consistency
                self.mapping_confidence[source] = self._calculate_partial_consistency(
                    source, target_list, examples
                )
        
        return final_mappings
    
    def _calculate_partial_consistency(self, source: Tuple[int, int], 
                                     targets: List[Tuple[int, int]], 
                                     examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Calculate consistency score for partial mappings."""
        consistent_count = 0
        total_count = 0
        
        for inp, out in examples:
            if inp[source] != 0:
                total_count += 1
                expected_value = inp[source]
                
                # Check if mapping is correct for this example
                all_correct = True
                for target in targets:
                    if out[target] != expected_value:
                        all_correct = False
                        break
                
                if all_correct:
                    consistent_count += 1
        
        return consistent_count / total_count if total_count > 0 else 0.0
    
    def discover_partial_patterns(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict[Tuple[int, int], List[Tuple[int, int]]]]:
        """
        Discover partial patterns that may not cover all positions.
        
        Args:
            examples: List of (input, output) pairs
            
        Returns:
            List of partial mapping patterns
        """
        partial_patterns = []
        
        # Group examples by similar characteristics
        example_groups = self._group_similar_examples(examples)
        
        for group in example_groups:
            if len(group) >= 2:  # Need at least 2 examples
                mappings = self.extract_position_mappings(group)
                if mappings:
                    partial_patterns.append(mappings)
        
        return partial_patterns
    
    def _group_similar_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """Group examples by similarity."""
        groups = []
        used = set()
        
        for i, (inp1, out1) in enumerate(examples):
            if i in used:
                continue
            
            group = [(inp1, out1)]
            used.add(i)
            
            # Find similar examples
            for j, (inp2, out2) in enumerate(examples):
                if j in used:
                    continue
                
                # Check similarity (same active positions)
                if self._are_similar(inp1, inp2):
                    group.append((inp2, out2))
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _are_similar(self, grid1: np.ndarray, grid2: np.ndarray) -> bool:
        """Check if two grids have similar structure."""
        # Simple similarity: same number of active pixels
        return np.count_nonzero(grid1) == np.count_nonzero(grid2)
    
    def combine_partial_mappings(self, partial_mappings: List[Dict[Tuple[int, int], List[Tuple[int, int]]]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Combine multiple partial mappings into a complete mapping.
        
        Args:
            partial_mappings: List of partial mappings
            
        Returns:
            Combined mapping
        """
        combined = {}
        
        for mapping in partial_mappings:
            for source, targets in mapping.items():
                if source not in combined:
                    combined[source] = targets
                else:
                    # Merge targets, ensuring consistency
                    combined[source] = list(set(combined[source]) | set(targets))
        
        return combined
    
    def analyze_mapping_properties(self, mappings: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> Dict[str, any]:
        """
        Analyze properties of discovered mappings.
        
        Args:
            mappings: Position mappings
            
        Returns:
            Dictionary of properties
        """
        properties = {
            'total_rules': len(mappings),
            'is_rotation': False,
            'is_reflection': False,
            'is_translation': False,
            'is_complex': False,
            'pattern_type': 'unknown'
        }
        
        if not mappings:
            return properties
        
        # Check for rotation pattern
        if self._is_rotation_mapping(mappings):
            properties['is_rotation'] = True
            properties['pattern_type'] = 'rotation'
        
        # Check for reflection pattern
        elif self._is_reflection_mapping(mappings):
            properties['is_reflection'] = True
            properties['pattern_type'] = 'reflection'
        
        # Check for translation pattern
        elif self._is_translation_mapping(mappings):
            properties['is_translation'] = True
            properties['pattern_type'] = 'translation'
        
        else:
            properties['is_complex'] = True
            properties['pattern_type'] = 'complex'
        
        return properties
    
    def _is_rotation_mapping(self, mappings: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> bool:
        """Check if mappings represent a rotation."""
        # For 2x2 grid, check if it matches 90-degree rotation pattern
        if self.grid_shape == (2, 2):
            rotation_90 = {
                (0, 0): [(0, 1)],
                (0, 1): [(1, 1)],
                (1, 1): [(1, 0)],
                (1, 0): [(0, 0)]
            }
            
            # Check if mappings match rotation pattern
            for source, targets in mappings.items():
                if source in rotation_90:
                    if set(targets) != set(rotation_90[source]):
                        return False
                else:
                    return False
            
            return True
        
        return False
    
    def _is_reflection_mapping(self, mappings: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> bool:
        """Check if mappings represent a reflection."""
        # Simplified check for now
        return False
    
    def _is_translation_mapping(self, mappings: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> bool:
        """Check if mappings represent a translation."""
        if not mappings:
            return False
        
        # Check if all mappings have same displacement
        displacements = []
        for source, targets in mappings.items():
            if len(targets) == 1:
                target = targets[0]
                dx = target[0] - source[0]
                dy = target[1] - source[1]
                displacements.append((dx, dy))
        
        # All displacements should be the same
        if displacements and all(d == displacements[0] for d in displacements):
            return True
        
        return False