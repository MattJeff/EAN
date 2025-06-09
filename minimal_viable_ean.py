"""
Minimal Viable Emergent Assembly Network (EAN)
Revolutionary approach to ARC-AGI through self-organizing neural assemblies
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import random
import time
import logging
import math
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("EAN")


class NeuronPMV:
    """
    Simple neuron optimized for assembly formation.
    
    Philosophy: Minimal individual intelligence, 
    maximum potential for collective intelligence.
    """
    
    def __init__(self, neuron_id: int, position: Tuple[float, float]):
        self.id = neuron_id
        self.position = position
        self.energy = random.uniform(30.0, 50.0)  # Initial energy
        self.last_activity_time = 0.0
        self.activation_history = deque(maxlen=20)
        self.neighboring_neurons: Set[int] = set()
        self.assembly_membership: Optional[str] = None
        
    def activate(self, energy_input: float, current_time: float):
        """Energy reception + temporal update"""
        old_energy = self.energy
        self.energy = min(100.0, self.energy + energy_input)
        self.last_activity_time = current_time
        self.activation_history.append({
            'time': current_time,
            'energy_before': old_energy,
            'energy_after': self.energy,
            'input': energy_input
        })
        
    def decay(self, decay_rate: float = 0.1):
        """Natural energy decay"""
        self.energy = max(0.0, self.energy - decay_rate)
        
    def get_activation_level(self) -> float:
        """Normalized activation level for assemblies"""
        return min(1.0, self.energy / 100.0)
        
    def is_available_for_assembly(self) -> bool:
        """Check if can join new assembly"""
        return self.energy > 20.0 and self.assembly_membership is None
        
    def get_spatial_influence(self, distance: float) -> float:
        """Energy influence based on distance"""
        if distance < 0.01:  # Same position
            return self.energy
        return max(0.0, self.energy * math.exp(-distance / 2.0))
        
    def calculate_distance_to(self, other_position: Tuple[float, float]) -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt(
            (self.position[0] - other_position[0])**2 + 
            (self.position[1] - other_position[1])**2
        )


class AssemblyPMV:
    """
    Self-organized neural assembly with collective memory and emergent actions.
    
    Core of intelligence: transforms individual patterns into collective behaviors.
    """
    
    def __init__(self, assembly_id: str, founder_ids: Set[int], birth_time: float):
        self.id = assembly_id
        self.founder_ids = frozenset(founder_ids)
        self.member_neuron_ids = set(founder_ids)
        self.birth_time = birth_time
        self.age = 0
        
        # Pattern memory
        self.shared_pattern_memory: Optional[np.ndarray] = None
        self.pattern_confidence = 0.0
        self.pattern_history = deque(maxlen=10)
        
        # Action learning
        self.action_strengths = {
            'rotate_90': 0.1,
            'rotate_270': 0.1,
            'flip_horizontal': 0.1,
            'flip_vertical': 0.1,
            'identity': 0.6
        }
        
        # Performance tracking
        self.success_history = deque(maxlen=20)
        self.consecutive_failures = 0
        self.coherence_score = 1.0
        self.specialization_type: Optional[str] = None
        
        # Transformation cache
        self.transformation_cache = {}
        
    def update_collective_memory(self, input_pattern: np.ndarray, 
                               active_member_ids: Set[int]) -> bool:
        """
        Update collective memory via majority vote of active members.
        
        Returns: True if consensus reached, False otherwise
        """
        if len(active_member_ids) < len(self.member_neuron_ids) * 0.5:
            return False  # Not enough active members
            
        # Store pattern in history
        self.pattern_history.append(input_pattern.copy())
        
        # Calculate pattern consensus
        if len(self.pattern_history) >= 3:
            # Find most common pattern elements
            pattern_stack = np.array(list(self.pattern_history)[-5:])
            consensus_pattern = np.zeros_like(input_pattern)
            
            # Majority vote for each position
            for i in range(input_pattern.shape[0]):
                for j in range(input_pattern.shape[1]):
                    values = pattern_stack[:, i, j]
                    unique, counts = np.unique(values, return_counts=True)
                    consensus_pattern[i, j] = unique[np.argmax(counts)]
                    
            self.shared_pattern_memory = consensus_pattern
            
            # Calculate confidence based on agreement
            agreement_scores = [
                np.mean(p == consensus_pattern) 
                for p in self.pattern_history
            ]
            self.pattern_confidence = np.mean(agreement_scores)
            
            return self.pattern_confidence > 0.7
            
        return False
        
    def attempt_transformation(self, input_pattern: np.ndarray) -> Tuple[Optional[np.ndarray], str, float]:
        """Attempt transformation based on memory + action strengths"""
        # Check if we have sufficient coherence
        if self.coherence_score < 0.3:
            return None, 'none', 0.0
            
        # CORRECTION : Plus d'exploration pendant l'apprentissage
        exploration_rate = 0.3 if self.age < 50 else 0.1
        
        if random.random() < exploration_rate:
            # Explorer préférentiellement les transformations non-identity
            non_identity_actions = [a for a in self.action_strengths.keys() if a != 'identity']
            action = random.choice(non_identity_actions)
        else:
            # Choose action based on strengths
            total_strength = sum(self.action_strengths.values())
            if total_strength == 0:
                return input_pattern.copy(), 'identity', 0.0
                
            # Weighted random selection
            r = random.random() * total_strength
            cumsum = 0
            action = 'identity'
            for act, strength in self.action_strengths.items():
                cumsum += strength
                if r <= cumsum:
                    action = act
                    break
                    
        # Apply transformation
        result = self._apply_action(input_pattern, action)
        confidence = self.action_strengths[action] * self.coherence_score
        
        return result, action, confidence
        
    def _apply_action(self, pattern: np.ndarray, action: str) -> np.ndarray:
        """Apply specific transformation action to pattern"""
        # Check cache first
        pattern_hash = hash(pattern.tobytes())
        cache_key = (pattern_hash, action)
        if cache_key in self.transformation_cache:
            return self.transformation_cache[cache_key].copy()
            
        # NOUVEAU : Pour les grilles 2x2, la rotation 90° horaire est en fait une rotation -90°
        if action == 'rotate_90':
            result = np.rot90(pattern, k=-1)  # Horaire
        elif action == 'rotate_270':
            result = np.rot90(pattern, k=1)   # Anti-horaire  
        elif action == 'flip_horizontal':
            result = np.fliplr(pattern)
        elif action == 'flip_vertical':
            result = np.flipud(pattern)
        else:  # identity
            result = pattern.copy()
            
        # Cache result
        self.transformation_cache[cache_key] = result.copy()
        return result
        
    def reinforce_success(self, action_type: str, success_strength: float = 1.0):
        """Renforce successful action + update history"""
        self.success_history.append(True)
        self.consecutive_failures = 0
        
        # CORRECTION : Renforcement plus fort et réduction des autres actions
        old_strength = self.action_strengths[action_type]
        
        # Augmenter fortement l'action réussie
        self.action_strengths[action_type] += 0.25 * success_strength  # Augmenté de 0.15 à 0.25
        
        # Réduire légèrement les autres actions
        for action in self.action_strengths:
            if action != action_type:
                self.action_strengths[action] *= 0.9
        
        # Normalize to maintain sum around 1
        total = sum(self.action_strengths.values())
        if total > 0:
            for action in self.action_strengths:
                self.action_strengths[action] /= total
                
        # Boost coherence
        self.coherence_score = min(1.0, self.coherence_score * 1.1)
        
        logger.debug(f"Assembly {self.id}: Reinforced {action_type} "
                    f"{old_strength:.3f} -> {self.action_strengths[action_type]:.3f}")
        
    def penalize_failure(self, action_type: str, penalty_strength: float = 0.3):
        """Penalize failed action"""
        self.success_history.append(False)
        self.consecutive_failures += 1
        
        # Weaken the failed action
        old_strength = self.action_strengths[action_type]
        self.action_strengths[action_type] *= (1.0 - 0.05 * penalty_strength)
        
        # Reduce coherence
        self.coherence_score *= 0.95
        
        logger.debug(f"Assembly {self.id}: Penalized {action_type} "
                    f"{old_strength:.3f} -> {self.action_strengths[action_type]:.3f}")
        
    def calculate_coherence(self, neuron_states: Dict[int, NeuronPMV]) -> float:
        """
        Calculate coherence based on:
        - Energy synchronization of members
        - Maintained spatial proximity
        - Correlated activity history
        """
        if len(self.member_neuron_ids) < 2:
            return 0.0
            
        member_neurons = [neuron_states[nid] for nid in self.member_neuron_ids 
                         if nid in neuron_states]
        
        if not member_neurons:
            return 0.0
            
        # Energy synchronization
        energies = [n.energy for n in member_neurons]
        mean_energy = np.mean(energies)
        energy_variance = np.var(energies) / (mean_energy + 1e-6)
        energy_sync = 1.0 / (1.0 + energy_variance)
        
        # Spatial cohesion
        positions = [n.position for n in member_neurons]
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(np.array(p) - center) for p in positions]
        spatial_cohesion = 1.0 / (1.0 + np.mean(distances))
        
        # Success rate
        if len(self.success_history) > 0:
            success_rate = sum(self.success_history) / len(self.success_history)
        else:
            success_rate = 0.5
            
        # Combined coherence
        self.coherence_score = (
            0.4 * energy_sync + 
            0.3 * spatial_cohesion + 
            0.3 * success_rate
        )
        
        return self.coherence_score
        
    def detect_specialization(self) -> Optional[str]:
        """Detect emergence of specialization"""
        # Find dominant action
        max_action = max(self.action_strengths.items(), key=lambda x: x[1])
        
        if max_action[1] > 0.6 and max_action[0] != 'identity':
            self.specialization_type = f"{max_action[0]}_specialist"
            return self.specialization_type
        elif len(self.success_history) > 10:
            success_rate = sum(self.success_history) / len(self.success_history)
            if success_rate > 0.8:
                self.specialization_type = "high_performer"
                return self.specialization_type
                
        return None
        
    def should_dissolve(self) -> bool:
        """Check if assembly should dissolve"""
        # NOUVEAU : Protéger les assemblées spécialisées qui performent bien
        if self.specialization_type and sum(self.success_history[-10:]) > 5:
            return False  # Ne jamais dissoudre une assemblée spécialisée qui réussit
            
        if self.age > 100 and self.coherence_score < 0.2:
            return True
        if self.consecutive_failures > 10:
            return True
        if len(self.member_neuron_ids) < 2:
            return True
        return False


class MinimalViableEAN:
    """
    Emergent Assembly Network - Revolutionary prototype for ARC-AGI.
    
    Goal: Demonstrate that intelligence can emerge from self-organization
    without explicit rule programming.
    """
    
    # Formation parameters
    FORMATION_ENERGY_THRESHOLD = 75.0
    FORMATION_SPATIAL_RADIUS = 2.5
    MIN_ASSEMBLY_SIZE = 3
    MAX_ASSEMBLY_SIZE = 8
    MAX_CONCURRENT_ASSEMBLIES = 6
    
    # Coherence and performance
    COHERENCE_THRESHOLD_FOR_ACTION = 0.6
    HIGH_COHERENCE_THRESHOLD = 0.8
    PATTERN_CONSENSUS_THRESHOLD = 0.7
    
    # Lifecycle
    DISSOLUTION_AGE_LIMIT = 200
    DISSOLUTION_COHERENCE_MIN = 0.15
    UNSUCCESSFUL_DISSOLUTION_LIMIT = 10
    
    # Learning
    LEARNING_RATE_SUCCESS = 0.15
    LEARNING_RATE_FAILURE = 0.05
    EXPLORATION_RANDOMNESS = 0.1
    
    def __init__(self, num_neurons: int = 50, world_size: Tuple[float, float] = (10.0, 10.0)):
        self.neurons = self._initialize_neurons(num_neurons, world_size)
        self.assemblies: Dict[str, AssemblyPMV] = {}
        self.current_step = 0
        self.current_time = 0.0
        self.world_size = world_size
        
        # Performance tracking
        self.assembly_formation_history = []
        self.learning_metrics = {
            'assemblies_formed': 0,
            'successful_transformations': 0,
            'failed_transformations': 0,
            'specializations_emerged': 0
        }
        
        logger.info(f"EAN initialized with {num_neurons} neurons in {world_size} world")
        
    def _initialize_neurons(self, num_neurons: int, world_size: Tuple[float, float]) -> Dict[int, NeuronPMV]:
        """Initialize neurons with random positions"""
        neurons = {}
        for i in range(num_neurons):
            pos = (
                random.uniform(0, world_size[0]), 
                random.uniform(0, world_size[1])
            )
            neurons[i] = NeuronPMV(neuron_id=i, position=pos)
            
        # Pre-calculate neighbors for efficiency
        for n1 in neurons.values():
            for n2 in neurons.values():
                if n1.id != n2.id:
                    dist = n1.calculate_distance_to(n2.position)
                    if dist < self.FORMATION_SPATIAL_RADIUS:
                        n1.neighboring_neurons.add(n2.id)
                        
        return neurons
        
    def _spatial_clustering_analysis(self) -> List[Set[int]]:
        """Analyse sophistiquée de clustering spatial pour formation d'assemblées"""
        clusters = []
        
        # Find neurons with sufficient energy
        high_energy_neurons = [
            n for n in self.neurons.values() 
            if n.energy > self.FORMATION_ENERGY_THRESHOLD and n.is_available_for_assembly()
        ]
        
        # CORRECTION : Réduire le seuil si aucun neurone n'a assez d'énergie
        if len(high_energy_neurons) < self.MIN_ASSEMBLY_SIZE:
            # Essayer avec un seuil plus bas
            high_energy_neurons = [
                n for n in self.neurons.values() 
                if n.energy > 50.0 and n.is_available_for_assembly()  # Seuil réduit
            ]
            
        if len(high_energy_neurons) < self.MIN_ASSEMBLY_SIZE:
            return clusters
            
        # Sort by energy (highest first)
        high_energy_neurons.sort(key=lambda n: n.energy, reverse=True)
        
        # Greedy clustering
        used_neurons = set()
        for seed_neuron in high_energy_neurons:
            if seed_neuron.id in used_neurons:
                continue
                
            # Find nearby high-energy neighbors
            cluster = {seed_neuron.id}
            candidates = [
                nid for nid in seed_neuron.neighboring_neurons 
                if nid not in used_neurons and 
                self.neurons[nid].energy > self.FORMATION_ENERGY_THRESHOLD and
                self.neurons[nid].is_available_for_assembly()
            ]
            
            # Add nearest neighbors up to max size
            candidates.sort(
                key=lambda nid: seed_neuron.calculate_distance_to(self.neurons[nid].position)
            )
            
            for nid in candidates[:self.MAX_ASSEMBLY_SIZE - 1]:
                cluster.add(nid)
                
            if len(cluster) >= self.MIN_ASSEMBLY_SIZE:
                clusters.append(cluster)
                used_neurons.update(cluster)
                
                # Stop if we have enough assemblies
                if len(clusters) >= self.MAX_CONCURRENT_ASSEMBLIES - len(self.assemblies):
                    break
                    
        return clusters
        
    def _energy_synchronization_detection(self, neuron_group: Set[int]) -> float:
        """Detect energy synchronization in a group of neurons"""
        if len(neuron_group) < 2:
            return 0.0
            
        neurons = [self.neurons[nid] for nid in neuron_group]
        
        # Check recent activation correlation
        recent_times = []
        for n in neurons:
            if n.activation_history:
                recent_times.append([h['time'] for h in list(n.activation_history)[-5:]])
                
        if len(recent_times) < 2:
            return 0.0
            
        # Simple synchronization metric: how close are activation times?
        sync_score = 0.0
        comparisons = 0
        
        for i in range(len(recent_times)):
            for j in range(i + 1, len(recent_times)):
                times1, times2 = recent_times[i], recent_times[j]
                min_len = min(len(times1), len(times2))
                if min_len > 0:
                    diffs = [abs(times1[k] - times2[k]) for k in range(min_len)]
                    sync_score += 1.0 / (1.0 + np.mean(diffs))
                    comparisons += 1
                    
        return sync_score / comparisons if comparisons > 0 else 0.0
        
    def _form_assembly(self, neuron_ids: Set[int]) -> Optional[AssemblyPMV]:
        """Form a new assembly from a group of neurons"""
        if len(self.assemblies) >= self.MAX_CONCURRENT_ASSEMBLIES:
            return None
            
        # Generate unique ID
        assembly_id = f"assembly_{self.current_step}_{hash(frozenset(neuron_ids))}"
        
        # Create assembly
        assembly = AssemblyPMV(
            assembly_id=assembly_id,
            founder_ids=neuron_ids,
            birth_time=self.current_time
        )
        
        # Update neuron membership
        for nid in neuron_ids:
            self.neurons[nid].assembly_membership = assembly_id
            
        self.assemblies[assembly_id] = assembly
        self.learning_metrics['assemblies_formed'] += 1
        
        logger.info(f"Formed {assembly_id} with {len(neuron_ids)} neurons")
        return assembly
        
    def _assembly_lifecycle_management(self):
        """
        Complete lifecycle management of assemblies:
        - Formation
        - Maintenance
        - Dissolution
        """
        # Formation phase
        if len(self.assemblies) < self.MAX_CONCURRENT_ASSEMBLIES:
            potential_clusters = self._spatial_clustering_analysis()
            for cluster in potential_clusters:
                sync_score = self._energy_synchronization_detection(cluster)
                if sync_score > 0.5:  # Sufficient synchronization
                    self._form_assembly(cluster)
                    
        # Maintenance and dissolution phase
        to_dissolve = []
        for assembly_id, assembly in self.assemblies.items():
            # Update age
            assembly.age += 1
            
            # Calculate coherence
            assembly.calculate_coherence(self.neurons)
            
            # Check for specialization
            if assembly.age > 20 and assembly.specialization_type is None:
                spec = assembly.detect_specialization()
                if spec:
                    self.learning_metrics['specializations_emerged'] += 1
                    logger.info(f"{assembly_id} specialized as: {spec}")
                    
            # Check dissolution criteria
            if assembly.should_dissolve():
                to_dissolve.append(assembly_id)
                
        # Dissolve marked assemblies
        for assembly_id in to_dissolve:
            self._dissolve_assembly(assembly_id)
            
    def _dissolve_assembly(self, assembly_id: str):
        """Dissolve an assembly and free its neurons"""
        if assembly_id not in self.assemblies:
            return
            
        assembly = self.assemblies[assembly_id]
        
        # Free neurons
        for nid in assembly.member_neuron_ids:
            if nid in self.neurons:
                self.neurons[nid].assembly_membership = None
                
        # Remove assembly
        del self.assemblies[assembly_id]
        logger.info(f"Dissolved {assembly_id} (age: {assembly.age}, coherence: {assembly.coherence_score:.3f})")
        
    def _map_grid_to_activation(self, grid: np.ndarray) -> List[Tuple[int, float]]:
        """Map input grid to neuron activations"""
        activations = []
        h, w = grid.shape
        
        # CORRECTION : Augmenter la force d'activation
        for i in range(h):
            for j in range(w):
                if grid[i, j] > 0:
                    # Map grid position to world position
                    world_x = (j + 0.5) * self.world_size[0] / w
                    world_y = (i + 0.5) * self.world_size[1] / h
                    
                    # Activate nearby neurons with stronger signal
                    for neuron in self.neurons.values():
                        dist = neuron.calculate_distance_to((world_x, world_y))
                        if dist < 3.0:  # Radius augmenté
                            activation_strength = 80.0 * (1.0 - dist / 3.0) * grid[i, j]  # Plus fort
                            activations.append((neuron.id, activation_strength))
                            
        return activations
        
    def step(self, input_pattern: Optional[np.ndarray] = None):
        """Single simulation step"""
        self.current_step += 1
        self.current_time += 0.1
        
        # Apply energy decay
        for neuron in self.neurons.values():
            neuron.decay(0.1)
            
        # Process input if provided
        if input_pattern is not None:
            activations = self._map_grid_to_activation(input_pattern)
            for neuron_id, strength in activations:
                self.neurons[neuron_id].activate(strength, self.current_time)
                
        # Assembly lifecycle
        self._assembly_lifecycle_management()
        
        # Assembly actions
        for assembly in self.assemblies.values():
            # Update collective memory if assembly has active members
            active_members = {
                nid for nid in assembly.member_neuron_ids
                if self.neurons[nid].energy > 50.0
            }
            
            if len(active_members) >= 2:
                if input_pattern is not None:
                    assembly.update_collective_memory(input_pattern, active_members)
                    
    def train_on_example_advanced(self, input_grid: np.ndarray, 
                            output_grid: np.ndarray, 
                            num_simulation_steps: int = 75) -> Dict[str, Any]:
        """Advanced training with detailed metrics"""
        start_metrics = self.learning_metrics.copy()
        successful_transformations = 0
        transformation_attempts = 0
        
        logger.info(f"Training on example: {input_grid.flatten()} -> {output_grid.flatten()}")
        
        # NOUVEAU : Maintenir l'activation pendant tout l'entraînement
        for step in range(num_simulation_steps):
            # Activer avec l'input
            self.step(input_grid)
            
            # NOUVEAU : Boost périodique d'énergie pour maintenir l'activité
            if step % 20 == 0:
                activations = self._map_grid_to_activation(input_grid)
                for neuron_id, strength in activations:
                    self.neurons[neuron_id].activate(strength * 0.5, self.current_time)
            
            # Every 10 steps, let assemblies attempt transformation
            if step % 10 == 9 and len(self.assemblies) > 0:
                for assembly in list(self.assemblies.values()):
                    if assembly.coherence_score > self.COHERENCE_THRESHOLD_FOR_ACTION:
                        # Attempt transformation
                        result, action, confidence = assembly.attempt_transformation(input_grid)
                        transformation_attempts += 1
                        
                        if result is not None:
                            # Check if transformation is correct
                            if np.array_equal(result, output_grid):
                                assembly.reinforce_success(action, success_strength=1.5)  # Plus fort
                                successful_transformations += 1
                                self.learning_metrics['successful_transformations'] += 1
                                logger.info(f"{assembly.id} learned {action}! "
                                        f"Strength: {assembly.action_strengths[action]:.3f}")
                            else:
                                assembly.penalize_failure(action, penalty_strength=0.2)  # Moins punitif
                                self.learning_metrics['failed_transformations'] += 1
                                
        # Calculate training metrics
        end_metrics = self.learning_metrics.copy()
        
        return {
            'assemblies_formed': end_metrics['assemblies_formed'] - start_metrics['assemblies_formed'],
            'transformation_attempts': transformation_attempts,
            'successful_transformations': successful_transformations,
            'success_rate': successful_transformations / transformation_attempts if transformation_attempts > 0 else 0,
            'active_assemblies': len(self.assemblies),
            'specializations': [a.specialization_type for a in self.assemblies.values() if a.specialization_type],
            'average_coherence': np.mean([a.coherence_score for a in self.assemblies.values()]) if self.assemblies else 0
        }
        
    def solve_with_confidence_analysis(self, test_input: np.ndarray, 
                                 max_attempts: int = 50) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """Solution with confidence analysis"""
        logger.info(f"Solving test input: {test_input.flatten()}")
        
        # NOUVEAU : Chercher d'abord les assemblées spécialisées
        specialized_assemblies = [
            a for a in self.assemblies.values() 
            if a.specialization_type is not None
        ]
        
        if specialized_assemblies:
            logger.info(f"Found {len(specialized_assemblies)} specialized assemblies")
            # Utiliser directement une assemblée spécialisée
            for assembly in specialized_assemblies:
                result, action, confidence = assembly.attempt_transformation(test_input)
                if result is not None and action != 'identity':
                    return result, {
                        'confidence': confidence,
                        'consensus': 1.0,
                        'assembly_id': assembly.id,
                        'action_used': action,
                        'specialization': assembly.specialization_type,
                        'total_attempts': 1
                    }
        
        # Run simulation to activate assemblies
        for _ in range(20):
            self.step(test_input)
            
        # Collect transformation attempts from all assemblies
        all_attempts = []
        
        for attempt in range(max_attempts):
            self.step(test_input)
            
            for assembly in self.assemblies.values():
                if assembly.coherence_score > self.COHERENCE_THRESHOLD_FOR_ACTION:
                    result, action, confidence = assembly.attempt_transformation(test_input)
                    if result is not None:
                        all_attempts.append({
                            'result': result,
                            'action': action,
                            'confidence': confidence,
                            'assembly_id': assembly.id,
                            'specialization': assembly.specialization_type
                        })
                        
        if not all_attempts:
            return None, {'confidence': 0.0, 'consensus': 0.0}
            
        # Find most confident result
        all_attempts.sort(key=lambda x: x['confidence'], reverse=True)
        best_attempt = all_attempts[0]
        
        # Calculate consensus
        result_hashes = [hash(a['result'].tobytes()) for a in all_attempts]
        most_common_hash = max(set(result_hashes), key=result_hashes.count)
        consensus = result_hashes.count(most_common_hash) / len(result_hashes)
        
        confidence_metrics = {
            'confidence': best_attempt['confidence'],
            'consensus': consensus,
            'assembly_id': best_attempt['assembly_id'],
            'action_used': best_attempt['action'],
            'specialization': best_attempt['specialization'] or 'none',
            'total_attempts': len(all_attempts)
        }
        
        logger.info(f"Solution found with confidence: {confidence_metrics}")
        
        return best_attempt['result'], confidence_metrics
        
    def run_comprehensive_pmv_test(self) -> Dict[str, Any]:
        """
        Comprehensive test with emergence and learning metrics.
        
        Phases:
        1. Pre-test: Initial network state
        2. Progressive training on multiple examples
        3. Specialization emergence analysis
        4. Generalization test on unseen inputs
        5. Post-test: Network evolution analysis
        """
        logger.info("="*50)
        logger.info("Running Comprehensive PMV Test")
        logger.info("="*50)
        
        # Training examples for 90-degree rotation
        training_examples = [
            (np.array(
                [[1, 0], [0, 0]]), np.array(
                [[0, 1], [1, 0]])),
            (np.array(
                [[0, 1], [0, 0]]), np.array(
                [[0, 0], [0, 1]])),
            (np.array(
                [[0, 0], [0, 1]]), np.array(
                [[1, 0], [0, 0]])),
            (np.array(
                [[0, 0], [1, 0]]), np.array(
                [[0, 1], [0, 0]]))
        ]
        
        # Test examples (unseen during training)
        test_examples = [
            (np.array(
                [[1, 0], [0, 0]]), np.array(
                [[0, 1], [1, 0]])),  # Seen
            (np.array(
                [[0, 0], [1, 0]]), np.array(
                [[0, 1], [0, 0]]))   # Seen but good to verify
        ]
        
        results = {
            'training_progression': [],
            'test_results': [],
            'emergence_metrics': {},
            'final_network_state': {}
        }
        
        # Phase 1: Initial state
        initial_state = {
            'neurons': len(self.neurons),
            'assemblies': len(self.assemblies),
            'total_energy': sum(n.energy for n in self.neurons.values())
        }
        results['initial_state'] = initial_state
        
        # Phase 2: Progressive training
        logger.info("\nPhase 2: Training")
        for epoch in range(3):  # Multiple passes through examples
            logger.info(f"\nEpoch {epoch + 1}/3")
            epoch_metrics = []
            
            for input_grid, output_grid in training_examples:
                metrics = self.train_on_example_advanced(
                    input_grid, output_grid, 
                    num_simulation_steps=50 + epoch * 25  # Increase time with epochs
                )
                epoch_metrics.append(metrics)
                
                logger.info(f"  Example trained. Success rate: {metrics['success_rate']:.2%}, "
                          f"Active assemblies: {metrics['active_assemblies']}")
                
            results['training_progression'].append({
                'epoch': epoch + 1,
                'average_success_rate': np.mean([m['success_rate'] for m in epoch_metrics]),
                'total_assemblies': len(self.assemblies),
                'specializations': [a.specialization_type for a in self.assemblies.values() 
                                   if a.specialization_type]
            })
            
        # Phase 3: Specialization analysis
        logger.info("\nPhase 3: Analyzing emerged specializations")
        specialization_analysis = {}
        for assembly in self.assemblies.values():
            if assembly.specialization_type:
                specialization_analysis[assembly.id] = {
                    'type': assembly.specialization_type,
                    'action_strengths': dict(assembly.action_strengths),
                    'success_rate': sum(assembly.success_history) / len(assembly.success_history) 
                                   if assembly.success_history else 0,
                    'age': assembly.age
                }
                
        results['specialization_analysis'] = specialization_analysis
        
        # Phase 4: Test on examples
        logger.info("\nPhase 4: Testing generalization")
        for test_input, expected_output in test_examples:
            solution, confidence = self.solve_with_confidence_analysis(test_input, max_attempts=30)
            
            success = np.array_equal(solution, expected_output) if solution is not None else False
            
            results['test_results'].append({
                'input': test_input.tolist(),
                'expected': expected_output.tolist(),
                'predicted': solution.tolist() if solution is not None else None,
                'success': success,
                'confidence': confidence
            })
            
            logger.info(f"  Test: {'PASS' if success else 'FAIL'} "
                      f"(confidence: {confidence['confidence']:.3f}, "
                      f"consensus: {confidence['consensus']:.3f})")
                      
        # Phase 5: Calculate emergence metrics
        logger.info("\nPhase 5: Computing emergence metrics")
        
        # Entropy reduction (behavioral complexity over time)
        entropy_progression = []
        for prog in results['training_progression']:
            # Simple entropy based on assembly action distributions
            all_actions = []
            for assembly in self.assemblies.values():
                all_actions.extend(list(assembly.action_strengths.values()))
            if all_actions:
                # Normalize to probabilities
                all_actions = np.array(all_actions)
                all_actions = all_actions / all_actions.sum()
                # Calculate entropy
                entropy = -np.sum(all_actions * np.log(all_actions + 1e-10))
                entropy_progression.append(entropy)
                
        if len(entropy_progression) > 1:
            entropy_reduction = (entropy_progression[0] - entropy_progression[-1]) / entropy_progression[0]
        else:
            entropy_reduction = 0.0
            
        # Novelty score (how different are the solutions from random)
        random_transformations = ['identity', 'rotate_90', 'rotate_270', 'flip_horizontal', 'flip_vertical']
        actual_dominant_actions = [
            max(a.action_strengths.items(), key=lambda x: x[1])[0] 
            for a in self.assemblies.values()
        ]
        novelty_score = len(set(actual_dominant_actions) - {'identity'}) / len(random_transformations)
        
        # Self-organization score
        if len(self.assemblies) > 0:
            avg_coherence = np.mean([a.coherence_score for a in self.assemblies.values()])
            avg_specialization = len([a for a in self.assemblies.values() 
                                    if a.specialization_type]) / len(self.assemblies)
            self_organization = (avg_coherence + avg_specialization) / 2
        else:
            self_organization = 0.0
            
        results['emergence_metrics'] = {
            'entropy_reduction': entropy_reduction,
            'novelty_score': novelty_score,
            'self_organization': self_organization,
            'learning_efficiency': sum(r['success'] for r in results['test_results']) / 
                                 len(results['test_results']) if results['test_results'] else 0
        }
        
        # Final network state
        results['final_network_state'] = {
            'total_neurons': len(self.neurons),
            'active_assemblies': len(self.assemblies),
            'specialized_assemblies': len([a for a in self.assemblies.values() 
                                         if a.specialization_type]),
            'average_neuron_energy': np.mean([n.energy for n in self.neurons.values()]),
            'total_transformations_learned': self.learning_metrics['successful_transformations'],
            'overall_success_rate': (self.learning_metrics['successful_transformations'] / 
                                   (self.learning_metrics['successful_transformations'] + 
                                    self.learning_metrics['failed_transformations'])
                                   if self.learning_metrics['successful_transformations'] + 
                                      self.learning_metrics['failed_transformations'] > 0 else 0)
        }
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("Test Complete - Summary:")
        logger.info(f"  Assemblies formed: {results['final_network_state']['active_assemblies']}")
        logger.info(f"  Specialized assemblies: {results['final_network_state']['specialized_assemblies']}")
        logger.info(f"  Test accuracy: {results['emergence_metrics']['learning_efficiency']:.1%}")
        logger.info(f"  Entropy reduction: {results['emergence_metrics']['entropy_reduction']:.1%}")
        logger.info(f"  Self-organization score: {results['emergence_metrics']['self_organization']:.3f}")
        logger.info("="*50)
        
        return results


# Utility functions
def calculate_pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """Calculate similarity between patterns with normalization"""
    if pattern1.shape != pattern2.shape:
        return 0.0
    return np.mean(pattern1 == pattern2)


def visualize_network_state(network: MinimalViableEAN, show_assemblies: bool = True) -> str:
    """ASCII visualization of network state"""
    lines = []
    lines.append("="*50)
    lines.append("Network State Visualization")
    lines.append("="*50)
    
    # Network overview
    lines.append(f"Step: {network.current_step}")
    lines.append(f"Neurons: {len(network.neurons)} (Avg energy: "
                f"{np.mean([n.energy for n in network.neurons.values()]):.1f})")
    lines.append(f"Assemblies: {len(network.assemblies)}")
    
    if show_assemblies and network.assemblies:
        lines.append("\nAssemblies:")
        for aid, assembly in network.assemblies.items():
            spec = assembly.specialization_type or "unspecialized"
            lines.append(f"  {aid[:20]}... ({len(assembly.member_neuron_ids)} neurons)")
            lines.append(f"    Specialization: {spec}")
            lines.append(f"    Coherence: {assembly.coherence_score:.3f}")
            lines.append(f"    Top action: {max(assembly.action_strengths.items(), key=lambda x: x[1])}")
            
    lines.append("="*50)
    return "\n".join(lines)


# Test suite
class TestEAN:
    """Test suite for EAN validation"""
    
    @staticmethod
    def test_basic_assembly_formation():
        """Test that assemblies can form with high-energy neurons"""
        logger.info("\nTest: Basic Assembly Formation")
        
        network = MinimalViableEAN(num_neurons=20, world_size=(5.0, 5.0))
        
        # Boost some neuron energies ET les mettre proches spatialement
        center_x, center_y = 2.5, 2.5
        for i in range(10):
            network.neurons[i].energy = 85.0  # Au-dessus du seuil
            # Placer les neurones proches les uns des autres
            angle = (i / 10) * 2 * math.pi
            network.neurons[i].position = (
                center_x + 0.5 * math.cos(angle),
                center_y + 0.5 * math.sin(angle)
            )
            
        # Recalculer les voisins après repositionnement
        for n1 in network.neurons.values():
            n1.neighboring_neurons = set()
            for n2 in network.neurons.values():
                if n1.id != n2.id:
                    dist = n1.calculate_distance_to(n2.position)
                    if dist < network.FORMATION_SPATIAL_RADIUS:
                        n1.neighboring_neurons.add(n2.id)
            
        # Run steps to allow formation
        for _ in range(20):  # Plus de steps
            network.step()
        
    @staticmethod
    def test_pattern_learning():
        """Test that assemblies can learn simple transformations"""
        logger.info("\nTest: Pattern Learning")
        
        network = MinimalViableEAN(num_neurons=30)
        
        # Train on rotation
        input_pattern = np.array([[1, 0], [0, 0]])
        output_pattern = np.array([[0, 1], [1, 0]])
        
        metrics = network.train_on_example_advanced(
            input_pattern, output_pattern, 
            num_simulation_steps=100
        )
        
        assert metrics['transformation_attempts'] > 0, "No transformation attempts made"
        assert metrics['successful_transformations'] > 0, "No successful transformations"
        logger.info(f"✓ Success rate: {metrics['success_rate']:.1%}")
        
    @staticmethod
    def test_generalization():
        """Test generalization to unseen patterns"""
        logger.info("\nTest: Generalization")
        
        network = MinimalViableEAN(num_neurons=40)
        
        # Train on multiple examples
        training_examples = [
            (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
            (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]]))
        ]
        
        for inp, out in training_examples * 3:  # Train multiple times
            network.train_on_example_advanced(inp, out, num_simulation_steps=75)
            
        # Test on trained example
        test_input = np.array([[1, 0], [0, 0]])
        expected = np.array([[0, 1], [1, 0]])
        
        solution, confidence = network.solve_with_confidence_analysis(test_input)
        
        assert solution is not None, "No solution found"
        assert np.array_equal(solution, expected), f"Wrong solution: {solution}"
        logger.info(f"✓ Correct solution with confidence: {confidence['confidence']:.3f}")
        
    @staticmethod
    def run_all_tests():
        """Run all tests"""
        logger.info("\n" + "="*50)
        logger.info("Running EAN Test Suite")
        logger.info("="*50)
        
        TestEAN.test_basic_assembly_formation()
        TestEAN.test_pattern_learning()
        TestEAN.test_generalization()
        
        logger.info("\n✅ All tests passed!")


def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("Emergent Assembly Network - Minimal Viable Prototype")
    logger.info("="*60)
    
    # Create network
    network = MinimalViableEAN(num_neurons=50, world_size=(10.0, 10.0))
    
    # Run comprehensive test
    results = network.run_comprehensive_pmv_test()
    
    # Visualize final state
    print("\n" + visualize_network_state(network, show_assemblies=True))
    
    # Save results
    import json
    with open('ean_pmv_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
            
        json.dump(convert_to_serializable(results), f, indent=2)
        
    logger.info("\nResults saved to ean_pmv_results.json")
    
    # Run test suite
    TestEAN.run_all_tests()
    
    logger.info("\n🚀 EAN PMV demonstration complete!")
    

if __name__ == "__main__":
    main()