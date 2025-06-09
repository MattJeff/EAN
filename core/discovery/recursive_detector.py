"""
Recursive pattern detection and handling.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
import logging

from .base_discoverer import TransformationDiscoverer, TransformationKnowledge
from .atomic_operations import AtomicOperation

logger = logging.getLogger("EAN.RecursiveDetector")


class RecursivePatternDetector(TransformationDiscoverer):
    """
    Detector specialized for recursive and fractal patterns.
    
    Handles:
    - Self-similar patterns
    - Iterative transformations
    - Fractal-like structures
    - Multi-step recursive processes
    """
    
    def __init__(self, grid_shape: Tuple[int, int]):
        """Initialize recursive pattern detector."""
        super().__init__(grid_shape)
        self.pattern_memory = deque(maxlen=10)
        self.recursion_depth_limit = 5
        
    def discover_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[List[AtomicOperation]]:
        """
        Discover recursive pattern from examples.
        
        Args:
            examples: List of (input, output) pairs
            
        Returns:
            List of operations implementing recursive pattern
        """
        rules = self.detect_recursive_rule(examples)
        
        if rules and rules['recursion_type']:
            # Create appropriate operations based on recursion type
            if rules['recursion_type'] == 'self-similar':
                return self._create_self_similar_operations(rules)
            elif rules['recursion_type'] == 'iterative':
                return self._create_iterative_operations(rules)
            elif rules['recursion_type'] == 'fractal':
                return self._create_fractal_operations(rules)
        
        return None
    
    def detect_recursive_rule(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """
        Detect recursive rules in examples.
        
        Args:
            examples: List of (input, output) pairs
            
        Returns:
            Dictionary describing recursive pattern
        """
        rules = {
            'base_pattern': None,
            'recursive_positions': [],
            'scale_factor': 1.0,
            'recursion_type': None,
            'recursion_depth': 1,
            'transformation_rule': None
        }
        
        for inp, out in examples:
            # Extract objects and analyze structure
            input_objects = self._extract_objects(inp)
            output_objects = self._extract_objects(out)
            
            # Check different types of recursion
            if self._is_self_similar(input_objects, output_objects):
                rules['recursion_type'] = 'self-similar'
                rules['base_pattern'] = input_objects
                rules['recursive_positions'] = self._find_recursive_positions(
                    input_objects, output_objects
                )
                rules['recursion_depth'] = self._estimate_recursion_depth(
                    input_objects, output_objects
                )
            
            elif self._is_iterative(inp, out, examples):
                rules['recursion_type'] = 'iterative'
                rules['transformation_rule'] = self._extract_iterative_rule(examples)
            
            elif self._is_fractal_like(inp, out):
                rules['recursion_type'] = 'fractal'
                rules['scale_factor'] = self._calculate_scale_factor(inp, out)
        
        return rules
    
    def _extract_objects(self, grid: np.ndarray) -> List[Dict]:
        """Extract objects with their properties."""
        objects = []
        visited = set()
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0 and (i, j) not in visited:
                    # Extract connected component
                    obj = self._extract_connected_component(grid, i, j, visited)
                    objects.append(obj)
        
        return objects
    
    def _extract_connected_component(self, grid: np.ndarray, start_i: int, 
                                   start_j: int, visited: Set[Tuple[int, int]]) -> Dict:
        """Extract a connected component starting from given position."""
        value = grid[start_i, start_j]
        component = {
            'value': value,
            'positions': [],
            'bounds': {'min_i': start_i, 'max_i': start_i, 
                      'min_j': start_j, 'max_j': start_j},
            'center': None
        }
        
        # BFS to find all connected positions
        queue = [(start_i, start_j)]
        visited.add((start_i, start_j))
        
        while queue:
            i, j = queue.pop(0)
            component['positions'].append((i, j))
            
            # Update bounds
            component['bounds']['min_i'] = min(component['bounds']['min_i'], i)
            component['bounds']['max_i'] = max(component['bounds']['max_i'], i)
            component['bounds']['min_j'] = min(component['bounds']['min_j'], j)
            component['bounds']['max_j'] = max(component['bounds']['max_j'], j)
            
            # Check neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < grid.shape[0] and 
                    0 <= nj < grid.shape[1] and 
                    (ni, nj) not in visited and 
                    grid[ni, nj] == value):
                    queue.append((ni, nj))
                    visited.add((ni, nj))
        
        # Calculate center
        component['center'] = (
            sum(p[0] for p in component['positions']) / len(component['positions']),
            sum(p[1] for p in component['positions']) / len(component['positions'])
        )
        
        return component
    
    def _is_self_similar(self, input_objects: List[Dict], 
                        output_objects: List[Dict]) -> bool:
        """Check if pattern exhibits self-similarity."""
        if not input_objects or not output_objects:
            return False
        
        # Check if input pattern appears in output
        input_positions = set()
        for obj in input_objects:
            input_positions.update(obj['positions'])
        
        output_positions = set()
        for obj in output_objects:
            output_positions.update(obj['positions'])
        
        # Original pattern should be preserved
        if not input_positions.issubset(output_positions):
            return False
        
        # Should have additional copies
        return len(output_objects) > len(input_objects)
    
    def _is_iterative(self, inp: np.ndarray, out: np.ndarray, 
                     examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if pattern represents iterative transformation."""
        # Look for patterns that could be intermediate steps
        if len(examples) < 2:
            return False
        
        # Check if outputs of some examples match inputs of others
        for i, (inp1, out1) in enumerate(examples):
            for j, (inp2, out2) in enumerate(examples):
                if i != j and np.array_equal(out1, inp2):
                    return True
        
        return False
    
    def _is_fractal_like(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if pattern exhibits fractal-like properties."""
        # Simplified check: repeated patterns at different scales
        input_objects = self._extract_objects(inp)
        output_objects = self._extract_objects(out)
        
        if len(output_objects) <= len(input_objects):
            return False
        
        # Check for scaled copies
        scales_found = set()
        for out_obj in output_objects:
            for in_obj in input_objects:
                scale = self._calculate_object_scale(in_obj, out_obj)
                if scale and scale != 1.0:
                    scales_found.add(scale)
        
        return len(scales_found) > 0
    
    def _find_recursive_positions(self, base_objects: List[Dict], 
                                 output_objects: List[Dict]) -> List[Tuple[int, int]]:
        """Find positions where pattern recurses."""
        base_positions = set()
        for obj in base_objects:
            base_positions.update(obj['positions'])
        
        output_positions = set()
        for obj in output_objects:
            output_positions.update(obj['positions'])
        
        # New positions are potential recursive positions
        new_positions = output_positions - base_positions
        
        # Analyze relationships
        recursive_positions = []
        for new_pos in new_positions:
            if self._is_recursive_position(new_pos, base_objects):
                recursive_positions.append(new_pos)
        
        return recursive_positions
    
    def _is_recursive_position(self, position: Tuple[int, int], 
                              base_objects: List[Dict]) -> bool:
        """Determine if position follows recursive pattern."""
        # Check common recursive patterns
        i, j = position
        h, w = self.grid_shape
        
        # Opposite corners
        if (i, j) in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            return True
        
        # Check relationship to base object centers
        for obj in base_objects:
            center = obj['center']
            # Opposite position
            opposite_i = h - 1 - int(center[0])
            opposite_j = w - 1 - int(center[1])
            if abs(i - opposite_i) <= 1 and abs(j - opposite_j) <= 1:
                return True
        
        return False
    
    def _estimate_recursion_depth(self, input_objects: List[Dict], 
                                 output_objects: List[Dict]) -> int:
        """Estimate the depth of recursion."""
        if not input_objects:
            return 0
        
        # Simple estimate based on object count ratio
        ratio = len(output_objects) / len(input_objects)
        
        if ratio <= 2:
            return 1
        elif ratio <= 4:
            return 2
        else:
            return 3
    
    def _calculate_scale_factor(self, inp: np.ndarray, out: np.ndarray) -> float:
        """Calculate scale factor for fractal patterns."""
        input_size = np.count_nonzero(inp)
        output_size = np.count_nonzero(out)
        
        if input_size == 0:
            return 1.0
        
        return output_size / input_size
    
    def _calculate_object_scale(self, obj1: Dict, obj2: Dict) -> Optional[float]:
        """Calculate scale between two objects."""
        # Compare bounding box sizes
        size1 = (obj1['bounds']['max_i'] - obj1['bounds']['min_i'] + 1,
                obj1['bounds']['max_j'] - obj1['bounds']['min_j'] + 1)
        size2 = (obj2['bounds']['max_i'] - obj2['bounds']['min_i'] + 1,
                obj2['bounds']['max_j'] - obj2['bounds']['min_j'] + 1)
        
        if size1[0] == 0 or size1[1] == 0:
            return None
        
        scale_i = size2[0] / size1[0]
        scale_j = size2[1] / size1[1]
        
        # Check if scales are consistent
        if abs(scale_i - scale_j) < 0.1:
            return (scale_i + scale_j) / 2
        
        return None
    
    def _extract_iterative_rule(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Extract rule for iterative transformations."""
        rule = {
            'type': 'iterative',
            'steps': []
        }
        
        # Find sequence of transformations
        for i, (inp1, out1) in enumerate(examples):
            for j, (inp2, out2) in enumerate(examples):
                if np.array_equal(out1, inp2):
                    # Found a sequence
                    rule['steps'].append({
                        'from': i,
                        'to': j,
                        'transformation': self._analyze_single_transformation(inp1, out1)
                    })
        
        return rule
    
    def _analyze_single_transformation(self, inp: np.ndarray, out: np.ndarray) -> str:
        """Analyze a single transformation step."""
        # Simple analysis for now
        if np.sum(out) > np.sum(inp):
            return "expansion"
        elif np.sum(out) < np.sum(inp):
            return "contraction"
        else:
            return "rearrangement"
    
    def _create_self_similar_operations(self, rules: Dict) -> List[AtomicOperation]:
        """Create operations for self-similar patterns."""
        operations = []
        
        # Create position mappings for recursive copies
        mappings = {}
        
        if rules['base_pattern'] and rules['recursive_positions']:
            for obj in rules['base_pattern']:
                for pos in obj['positions']:
                    # Map to original position plus recursive positions
                    targets = [pos]  # Keep original
                    
                    # Add recursive copies
                    for rec_pos in rules['recursive_positions']:
                        targets.append(rec_pos)
                    
                    mappings[pos] = targets
            
            operations.append(AtomicOperation("position_mapping", (mappings,)))
        
        return operations
    
    def _create_iterative_operations(self, rules: Dict) -> List[AtomicOperation]:
        """Create operations for iterative patterns."""
        # For now, return empty - would need more complex handling
        return []
    
    def _create_fractal_operations(self, rules: Dict) -> List[AtomicOperation]:
        """Create operations for fractal patterns."""
        # For now, return empty - would need more complex handling
        return []


class ImprovedRecursiveAssembly:
    """
    Assembly specialized for recursive pattern handling.
    
    Features:
    - Working memory for intermediate states
    - Recursive transformation application
    - Depth analysis
    - Pattern composition
    """
    
    def __init__(self, assembly_id: str, grid_shape: Tuple[int, int]):
        """Initialize recursive assembly."""
        self.id = assembly_id
        self.grid_shape = grid_shape
        self.recursive_detector = RecursivePatternDetector(grid_shape)
        self.recursive_rules = None
        self.working_memory = deque(maxlen=5)
        self.is_recursive_specialized = False
        self.max_recursion_depth = 3
        
        # Transformation knowledge
        self.transformation_knowledge = TransformationKnowledge()
        
        logger.debug(f"Created recursive assembly {assembly_id}")
    
    def learn_recursive_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """
        Learn recursive pattern from examples.
        
        Args:
            examples: List of (input, output) pairs
            
        Returns:
            True if pattern learned successfully
        """
        # D'abord essayer la détection standard
        rules = self.recursive_detector.detect_recursive_rule(examples)
        
        # Si échec, essayer la détection améliorée pour patterns spécifiques
        if not rules or not rules['recursion_type']:
            rules = self._detect_enhanced_recursive_pattern(examples)
        
        if rules and rules['recursion_type']:
            self.recursive_rules = rules
            self.is_recursive_specialized = True
            
            # Create transformation knowledge
            operations = self.recursive_detector.discover_pattern(examples)
            if operations:
                self.transformation_knowledge.operations = operations
                self.transformation_knowledge.discovery_method = f"recursive_{rules['recursion_type']}"
                self.transformation_knowledge.confidence = 0.8
            else:
                # Créer des opérations basées sur les patterns détectés
                self._create_operations_from_rules(rules)
                
            logger.info(f"{self.id} learned recursive pattern: {rules['recursion_type']}")
            return True
        
        return False
    
    def _detect_enhanced_recursive_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Détection améliorée pour patterns récursifs spécifiques"""
        rules = {
            'base_pattern': None,
            'recursive_positions': [],
            'scale_factor': 1.0,
            'recursion_type': None,
            'recursion_depth': 1,
            'transformation_rule': None
        }
        
        # Analyser les exemples pour détecter des patterns spécifiques
        diagonal_count = 0
        center_to_corners_count = 0
        
        for inp, out in examples:
            # Détecter pattern diagonal
            if self._is_diagonal_pattern(inp, out):
                diagonal_count += 1
            
            # Détecter pattern centre vers coins
            if self._is_center_to_corners_pattern(inp, out):
                center_to_corners_count += 1
        
        # Déterminer le type dominant
        if diagonal_count > len(examples) / 2:
            rules['recursion_type'] = 'diagonal'
            rules['transformation_rule'] = 'diagonal_copy'
        elif center_to_corners_count > len(examples) / 2:
            rules['recursion_type'] = 'center_to_corners'
            rules['transformation_rule'] = 'center_expansion'
        else:
            # Essayer de détecter d'autres patterns
            rules = self._detect_general_recursive_pattern(examples)
        
        return rules
    
    def _is_diagonal_pattern(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Vérifie si c'est un pattern diagonal"""
        h, w = inp.shape
        
        # Pour une grille 3x3, vérifier les diagonales
        if h == 3 and w == 3:
            # Coins et leurs opposés
            diagonals = [
                ((0, 0), (2, 2)),
                ((0, 2), (2, 0)),
                ((2, 0), (0, 2)),
                ((2, 2), (0, 0))
            ]
            
            for (si, sj), (ti, tj) in diagonals:
                if inp[si, sj] != 0 and out[ti, tj] == inp[si, sj] and out[si, sj] == inp[si, sj]:
                    return True
        
        return False
    
    def _is_center_to_corners_pattern(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Vérifie si c'est un pattern centre vers coins"""
        h, w = inp.shape
        
        if h == 3 and w == 3:
            center = inp[1, 1]
            if center != 0:
                # Vérifier si le centre est copié aux coins dans l'output
                corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
                corners_match = all(out[i, j] == center for i, j in corners)
                center_preserved = out[1, 1] == center
                
                return corners_match and center_preserved
        
        return False
    
    def _detect_general_recursive_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Détection générale de patterns récursifs"""
        rules = {
            'recursion_type': 'self-similar',
            'transformation_rule': 'position_copy',
            'recursive_positions': []
        }
        
        # Analyser les positions qui changent
        for inp, out in examples:
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    if inp[i, j] != 0:
                        # Trouver où cette valeur apparaît dans l'output
                        value = inp[i, j]
                        for ii in range(out.shape[0]):
                            for jj in range(out.shape[1]):
                                if out[ii, jj] == value and (ii, jj) != (i, j):
                                    rules['recursive_positions'].append(((i, j), (ii, jj)))
        
        return rules if rules['recursive_positions'] else {'recursion_type': None}
    
    def _create_operations_from_rules(self, rules: Dict):
        """Crée des opérations à partir des règles détectées"""
        if rules['recursion_type'] in ['diagonal', 'center_to_corners', 'self-similar']:
            # Créer une opération personnalisée
            from .atomic_operations import AtomicOperation
            
            # Créer un mapping basé sur le type
            if rules['recursion_type'] == 'diagonal':
                mappings = {
                    (0, 0): [(0, 0), (2, 2)],
                    (0, 2): [(0, 2), (2, 0)],
                    (2, 0): [(2, 0), (0, 2)],
                    (2, 2): [(2, 2), (0, 0)],
                    (1, 1): [(1, 1)]  # Centre reste
                }
            elif rules['recursion_type'] == 'center_to_corners':
                mappings = {
                    (1, 1): [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)]
                }
            else:
                # Utiliser les positions détectées
                mappings = {}
                for (src, tgt) in rules.get('recursive_positions', []):
                    if src not in mappings:
                        mappings[src] = [src]
                    mappings[src].append(tgt)
            
            operation = AtomicOperation("position_mapping", (mappings,))
            self.transformation_knowledge.operations = [operation]
            self.transformation_knowledge.confidence = 0.9
    
    def apply_recursive_transformation(self, input_pattern: np.ndarray, 
                                     depth: int = 0) -> np.ndarray:
        """
        Apply recursive transformation with depth control.
        
        Args:
            input_pattern: Input grid
            depth: Current recursion depth
            
        Returns:
            Transformed grid
        """
        if not self.is_recursive_specialized or depth > self.max_recursion_depth:
            return input_pattern
        
        # Save to working memory
        self.working_memory.append({
            'pattern': input_pattern.copy(),
            'depth': depth
        })
        
        # Détection améliorée du type de pattern
        pattern_type = self._detect_specific_pattern_type(input_pattern)
        
        if pattern_type == "diagonal":
            return self._apply_diagonal_pattern(input_pattern)
        elif pattern_type == "center_to_corners":
            return self._apply_center_to_corners_pattern(input_pattern)
        
        # Fallback sur les méthodes existantes
        result = input_pattern.copy()
        
        if self.recursive_rules['recursion_type'] == 'self-similar':
            result = self._apply_self_similar(input_pattern)
        elif self.recursive_rules['recursion_type'] == 'iterative':
            result = self._apply_iterative(input_pattern, depth)
        elif self.recursive_rules['recursion_type'] == 'fractal':
            result = self._apply_fractal(input_pattern, depth)
        
        # Check if further recursion needed
        if self._needs_further_recursion(result, depth):
            result = self.apply_recursive_transformation(result, depth + 1)
        
        return result
    
    def _detect_specific_pattern_type(self, pattern: np.ndarray) -> str:
        """Détecte le type spécifique de pattern récursif"""
        h, w = pattern.shape
        
        if h == 3 and w == 3:
            # Vérifier les coins
            corners = [
                (0, 0, pattern[0, 0]), (0, w-1, pattern[0, w-1]),
                (h-1, 0, pattern[h-1, 0]), (h-1, w-1, pattern[h-1, w-1])
            ]
            center = pattern[h//2, w//2]
            
            # Un seul coin actif -> diagonal
            active_corners = [(i, j, v) for i, j, v in corners if v != 0]
            if len(active_corners) == 1 and center == 0:
                return "diagonal"
            
            # Centre actif, coins vides -> center to corners
            elif center != 0 and all(v == 0 for _, _, v in corners):
                return "center_to_corners"
        
        return "unknown"
    
    def _apply_diagonal_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Applique le pattern diagonal (coin -> coin opposé)"""
        result = pattern.copy()
        h, w = pattern.shape
        
        # Mapping diagonal
        diagonal_map = {
            (0, 0): (h-1, w-1),
            (0, w-1): (h-1, 0),
            (h-1, 0): (0, w-1),
            (h-1, w-1): (0, 0)
        }
        
        for (si, sj), (ti, tj) in diagonal_map.items():
            if 0 <= si < h and 0 <= sj < w and 0 <= ti < h and 0 <= tj < w:
                if pattern[si, sj] != 0:
                    result[ti, tj] = pattern[si, sj]
        
        return result
    
    def _apply_center_to_corners_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Applique le pattern centre vers coins"""
        result = pattern.copy()
        h, w = pattern.shape
        
        center_i, center_j = h // 2, w // 2
        center_val = pattern[center_i, center_j]
        
        if center_val != 0:
            # Copier aux 4 coins
            corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
            for ci, cj in corners:
                if 0 <= ci < h and 0 <= cj < w:
                    result[ci, cj] = center_val
        
        return result
    
    def _apply_self_similar(self, pattern: np.ndarray) -> np.ndarray:
        """Apply self-similar transformation."""
        if not self.transformation_knowledge.operations:
            return pattern
        
        result = pattern.copy()
        for op in self.transformation_knowledge.operations:
            result = op.apply(result)
        
        return result
    
    def _apply_iterative(self, pattern: np.ndarray, depth: int) -> np.ndarray:
        """Apply iterative transformation."""
        result = pattern.copy()
        
        # Apply transformation based on depth
        if self.recursive_rules.get('transformation_rule'):
            rule = self.recursive_rules['transformation_rule']
            
            # Simple iterative application
            for _ in range(min(depth + 1, self.max_recursion_depth)):
                # Apply basic transformation
                if rule['type'] == 'iterative' and rule['steps']:
                    # Apply first step transformation
                    result = self._apply_basic_transformation(result, rule['steps'][0]['transformation'])
        
        return result
    
    def _apply_fractal(self, pattern: np.ndarray, depth: int) -> np.ndarray:
        """Apply fractal transformation."""
        result = pattern.copy()
        scale = self.recursive_rules.get('scale_factor', 1.0)
        
        # For each object, create scaled copies
        objects = self.recursive_detector._extract_objects(pattern)
        
        for obj in objects:
            # Create copies at corners (simplified fractal)
            for corner in [(0, 0), (0, self.grid_shape[1]-1), 
                          (self.grid_shape[0]-1, 0), 
                          (self.grid_shape[0]-1, self.grid_shape[1]-1)]:
                if self._can_place_at(result, obj, corner):
                    self._place_object_at(result, obj, corner)
        
        return result
    
    def _apply_basic_transformation(self, pattern: np.ndarray, 
                                   transformation_type: str) -> np.ndarray:
        """Apply basic transformation based on type."""
        if transformation_type == "expansion":
            # Add pixels around existing ones
            result = pattern.copy()
            for i in range(pattern.shape[0]):
                for j in range(pattern.shape[1]):
                    if pattern[i, j] != 0:
                        # Add to adjacent empty cells
                        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < pattern.shape[0] and 
                                0 <= nj < pattern.shape[1] and 
                                result[ni, nj] == 0):
                                result[ni, nj] = pattern[i, j]
                                break  # Only add one
            return result
        
        return pattern.copy()
    
    def _needs_further_recursion(self, pattern: np.ndarray, depth: int) -> bool:
        """Check if pattern needs further recursion."""
        if depth >= self.max_recursion_depth - 1:
            return False
        
        # Check if pattern is "complete"
        if self.recursive_rules and self.recursive_rules.get('recursion_type') == 'self-similar':
            # Check if expected positions are filled
            expected_positions = self.recursive_rules.get('recursive_positions', [])
            
            # Handle different formats of expected_positions
            for item in expected_positions:
                # Si c'est un tuple de (source, target)
                if isinstance(item, tuple) and len(item) == 2:
                    if isinstance(item[0], tuple) and isinstance(item[1], tuple):
                        # Format ((src_i, src_j), (tgt_i, tgt_j))
                        _, tgt_pos = item
                        if 0 <= tgt_pos[0] < pattern.shape[0] and 0 <= tgt_pos[1] < pattern.shape[1]:
                            if pattern[tgt_pos] == 0:
                                return True
                    else:
                        # Format simple (i, j)
                        if 0 <= item[0] < pattern.shape[0] and 0 <= item[1] < pattern.shape[1]:
                            if pattern[item] == 0:
                                return True
                # Si c'est une position simple
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    if 0 <= item[0] < pattern.shape[0] and 0 <= item[1] < pattern.shape[1]:
                        if pattern[item[0], item[1]] == 0:
                            return True
        
        return False
        

    
    def _can_place_at(self, grid: np.ndarray, obj: Dict, 
                     position: Tuple[int, int]) -> bool:
        """Check if object can be placed at position."""
        for pos in obj['positions']:
            new_i = position[0] + pos[0] - obj['bounds']['min_i']
            new_j = position[1] + pos[1] - obj['bounds']['min_j']
            
            if (new_i < 0 or new_i >= grid.shape[0] or 
                new_j < 0 or new_j >= grid.shape[1] or 
                grid[new_i, new_j] != 0):
                return False
        
        return True
    
    def _place_object_at(self, grid: np.ndarray, obj: Dict, 
                        position: Tuple[int, int]):
        """Place object at given position."""
        for pos in obj['positions']:
            new_i = position[0] + pos[0] - obj['bounds']['min_i']
            new_j = position[1] + pos[1] - obj['bounds']['min_j']
            
            if (0 <= new_i < grid.shape[0] and 
                0 <= new_j < grid.shape[1]):
                grid[new_i, new_j] = obj['value']
    
    def analyze_recursion_depth(self, pattern: np.ndarray) -> int:
        """
        Analyze optimal recursion depth for pattern.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Recommended recursion depth
        """
        num_objects = np.count_nonzero(pattern)
        
        if num_objects == 0:
            return 0
        elif num_objects == 1:
            return 1  # Simple copy
        elif num_objects <= 4:
            return 2  # Moderate recursion
        else:
            return 3  # Deep recursion
    
    def get_transformation_confidence(self, pattern: np.ndarray) -> float:
        """Get confidence for transforming given pattern."""
        if not self.is_recursive_specialized:
            return 0.0
        
        # Base confidence
        confidence = self.transformation_knowledge.confidence
        
        # Adjust based on pattern complexity
        complexity = np.count_nonzero(pattern) / pattern.size
        if complexity > 0.5:  # Very complex
            confidence *= 0.8
        elif complexity < 0.1:  # Very simple
            confidence *= 0.9
        
        return confidence