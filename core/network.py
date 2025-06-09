"""
Main EAN network implementation integrating all components.
"""

import logging
import math
import random
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
import numpy as np

from .neuron import NeuronPMV
from .assembly import EmergentAssemblyEAN
from .discovery.position_mapping import ImprovedTransformationDiscoverer
from .discovery.recursive_detector import RecursivePatternDetector, ImprovedRecursiveAssembly
from .discovery.atomic_operations import AtomicOperation

logger = logging.getLogger("EAN.Network")


class PatternIntegrator:
    """Integrates partial patterns from different assemblies."""

    def __init__(self, grid_shape: Tuple[int, int]):
        """Initialize pattern integrator."""
        self.grid_shape = grid_shape

    def combine_partial_patterns(self, specialized_assemblies: List[EmergentAssemblyEAN]) -> Optional[Dict]:
        """
        Combine partial patterns into complete pattern.

        Args:
            specialized_assemblies: List of specialized assemblies

        Returns:
            Combined mappings or None
        """
        if not specialized_assemblies:
            return None

        # Collect all mappings
        all_mappings = {}
        conflicts = []

        for assembly in specialized_assemblies:
            if assembly.transformation_knowledge.operations:
                for operation in assembly.transformation_knowledge.operations:
                    if operation.name == "position_mapping":
                        partial_mappings = operation.params[0]

                        for source, targets in partial_mappings.items():
                            if source not in all_mappings:
                                all_mappings[source] = targets
                            else:
                                # Check for conflicts
                                if all_mappings[source] != targets:
                                    conflicts.append({
                                        'source': source,
                                        'existing': all_mappings[source],
                                        'new': targets,
                                        'assembly': assembly.id
                                    })

        # Handle conflicts
        if conflicts:
            logger.warning(f"Found {len(conflicts)} mapping conflicts")
            # For now, keep first mapping
            # Could implement more sophisticated conflict resolution

        if all_mappings:
            logger.info(f"Integrated pattern: {len(all_mappings)} rules combined")
            return all_mappings

        return None

    def analyze_pattern_coverage(self, mappings: Dict, grid_shape: Tuple[int, int]) -> float:
        """
        Analyze how much of the grid is covered by mappings.

        Args:
            mappings: Position mappings
            grid_shape: Shape of grid

        Returns:
            Coverage percentage [0, 1]
        """
        total_positions = grid_shape[0] * grid_shape[1]
        covered_positions = len(mappings)

        return covered_positions / total_positions if total_positions > 0 else 0.0


class IntegratedEAN:
    """
    Integrated Emergent Assembly Network.

    Main network class that:
    - Manages neurons and assemblies
    - Coordinates learning and inference
    - Integrates patterns from multiple assemblies
    - Handles complex transformations
    """

    def __init__(self, num_neurons: int = 100,
                 world_size: Tuple[float, float] = (10.0, 10.0),
                 grid_shape: Tuple[int, int] = (2, 2)):
        """
        Initialize the EAN network.

        Args:
            num_neurons: Number of neurons
            world_size: Size of the spatial world
            grid_shape: Shape of transformation grids
        """
        self.num_neurons = num_neurons
        self.world_size = world_size
        self.grid_shape = grid_shape

        # Parameters - MOVED UP
        self.max_assemblies = 15
        self.max_recursive_assemblies = 5
        self.min_confidence_threshold = 0.25
        self.energy_decay_rate = 0.015
        self.activation_radius = 4.0

        # Initialize components
        self.neurons = self._initialize_neurons() # Now self.activation_radius is defined
        self.assemblies: Dict[str, EmergentAssemblyEAN] = {}
        self.recursive_assemblies: Dict[str, ImprovedRecursiveAssembly] = {}

        # Pattern integration
        self.pattern_integrator = PatternIntegrator(grid_shape)

        # Network state
        self.current_step = 0
        self.current_time = 0.0
        self.assembly_formation_cooldown = 0

        # Metrics
        self.total_discoveries = 0
        self.total_predictions = 0
        self.total_correct_predictions = 0

        logger.info(f"Initialized IntegratedEAN with {num_neurons} neurons, "
                   f"grid shape {grid_shape}")

    def _initialize_neurons(self) -> Dict[int, NeuronPMV]:
        """Initialize neurons with spatial distribution."""
        neurons = {}

        # Create neurons with random positions
        for i in range(self.num_neurons):
            position = (
                random.uniform(0, self.world_size[0]),
                random.uniform(0, self.world_size[1])
            )
            neurons[i] = NeuronPMV(i, position)

        # Establish neighbor relationships
        for n1 in neurons.values():
            for n2 in neurons.values():
                if n1.id != n2.id:
                    dist = n1.calculate_distance_to(n2.position)
                    if dist < self.activation_radius: # This line caused the error
                        n1.neighboring_neurons.add(n2.id)
                        # Initialize synaptic connection
                        n1.synapses[n2.id] = random.uniform(0.1, 0.3)

        logger.info(f"Initialized {len(neurons)} neurons with spatial connectivity")
        return neurons

    def _activate_for_pattern(self, pattern: np.ndarray):
        """
        Activate neurons based on input pattern.

        Args:
            pattern: Grid pattern to encode
        """
        h, w = pattern.shape

        for i in range(h):
            for j in range(w):
                if pattern[i, j] > 0:
                    # Convert grid position to world coordinates
                    world_x = (j + 0.5) * self.world_size[0] / w
                    world_y = (i + 0.5) * self.world_size[1] / h

                    # Activate nearby neurons
                    for neuron in self.neurons.values():
                        dist = neuron.calculate_distance_to((world_x, world_y))
                        if dist < self.activation_radius:
                            # Activation strength decreases with distance
                            strength = 80.0 * (1.0 - dist / self.activation_radius) * pattern[i, j]

                            # Activate neuron
                            if neuron.activate(strength, self.current_time):
                                # Neuron fired - propagate to neighbors
                                self._propagate_activation(neuron)

                    # Consider forming assemblies near high activation
                    if pattern[i, j] > 1 or np.sum(pattern) > 2:
                        self._form_assemblies_near_activation((world_x, world_y))

    def _propagate_activation(self, source_neuron: NeuronPMV):
        """
        Propagate activation from fired neuron.

        Args:
            source_neuron: Neuron that fired
        """
        signals = source_neuron.propagate_signal(self.neurons, self.current_time)

        for target_id, signal_strength in signals.items():
            target_neuron = self.neurons[target_id]

            # Apply signal
            if signal_strength > 0:
                target_neuron.activate(signal_strength, self.current_time)
            else:  # Inhibitory signal
                target_neuron.energy = max(0, target_neuron.energy + signal_strength)

            # Update synapse with STDP
            if source_neuron.last_spike_time > 0 and target_neuron.last_spike_time > 0:
                target_neuron.update_synapse_stdp(
                    source_neuron.id,
                    source_neuron.last_spike_time,
                    target_neuron.last_spike_time
                )

    def _form_assemblies_near_activation(self, activation_center: Tuple[float, float]):
        """
        Form new assemblies near activation center.

        Args:
            activation_center: World coordinates of activation
        """
        if self.assembly_formation_cooldown > 0:
            return

        if len(self.assemblies) >= self.max_assemblies:
            # Remove poorest performing assembly
            self._remove_weakest_assembly()

        # Find available neurons near activation
        nearby_neurons = []
        for neuron in self.neurons.values():
            if neuron.is_available_for_assembly():
                dist = neuron.calculate_distance_to(activation_center)
                if dist < self.activation_radius and neuron.energy > 45.0:
                    nearby_neurons.append((neuron.id, dist))

        # Need minimum neurons for assembly
        if len(nearby_neurons) >= 5:
            # Sort by distance and take closest
            nearby_neurons.sort(key=lambda x: x[1])
            founder_ids = {n[0] for n in nearby_neurons[:8]}

            # Create assembly
            assembly_id = f"assembly_{self.current_step}_{len(self.assemblies)}"
            assembly = EmergentAssemblyEAN(
                assembly_id,
                frozenset(founder_ids),
                self.current_time,
                self.grid_shape
            )

            # Update neurons
            for nid in founder_ids:
                self.neurons[nid].join_assembly(assembly_id)

            self.assemblies[assembly_id] = assembly
            self.assembly_formation_cooldown = 3

            logger.debug(f"Formed {assembly_id} with {len(founder_ids)} neurons")

    def _remove_weakest_assembly(self):
        """Remove the weakest performing assembly."""
        if not self.assemblies:
            return

        # Find assembly with worst performance
        worst_assembly = None
        worst_score = float('inf')

        for assembly in self.assemblies.values():
            if assembly.protection_counter == 0:  # Not protected
                score = (assembly.transformation_knowledge.success_rate * 0.5 +
                        assembly.stability_score * 0.3 +
                        (1.0 if assembly.is_specialized else 0.0) * 0.2)

                if score < worst_score:
                    worst_score = score
                    worst_assembly = assembly

        if worst_assembly:
            self._dissolve_assembly(worst_assembly.id)

    def _dissolve_assembly(self, assembly_id: str):
        """
        Dissolve an assembly.

        Args:
            assembly_id: ID of assembly to dissolve
        """
        if assembly_id not in self.assemblies:
            return

        assembly = self.assemblies[assembly_id]

        # Free neurons
        for nid in assembly.member_neuron_ids:
            if nid in self.neurons:
                self.neurons[nid].leave_assembly()

        # Remove assembly
        del self.assemblies[assembly_id]
        logger.debug(f"Dissolved {assembly_id}")

    def train_step(self, input_pattern: np.ndarray, output_pattern: np.ndarray):
        """
        Single training step.

        Args:
            input_pattern: Input grid pattern
            output_pattern: Expected output pattern
        """
        self.current_step += 1
        self.current_time += 0.1

        # Update cooldowns
        if self.assembly_formation_cooldown > 0:
            self.assembly_formation_cooldown -= 1

        # Apply energy decay
        for neuron in self.neurons.values():
            neuron.decay(self.energy_decay_rate)

        # Activate based on input
        self._activate_for_pattern(input_pattern)

        # Let assemblies observe transformation
        for assembly in self.assemblies.values():
            assembly.observe_transformation(input_pattern, output_pattern)

        # Update assemblies
        assemblies_to_remove = []

        for assembly_id, assembly in self.assemblies.items():
            # Attempt discovery
            if not assembly.is_specialized and len(assembly.transformation_examples) >= 6:
                if assembly.attempt_discovery():
                    self.total_discoveries += 1

                    # Check if it's a recursive pattern
                    if self._is_recursive_pattern(assembly):
                        self._create_recursive_assembly(assembly)

            # Test specialized assemblies
            if assembly.is_specialized:
                result, confidence = assembly.apply_transformation(input_pattern)

                if result is not None:
                    is_correct = np.array_equal(result, output_pattern)
                    assembly.record_application_result(is_correct)

                    if is_correct:
                        # Reward successful transformation
                        self._reward_assembly(assembly)

            # Update assembly state
            assembly.update()

            # Check if should dissolve
            if assembly.should_dissolve():
                assemblies_to_remove.append(assembly_id)

        # Remove assemblies marked for dissolution
        for assembly_id in assemblies_to_remove:
            self._dissolve_assembly(assembly_id)

    def _is_recursive_pattern(self, assembly: EmergentAssemblyEAN) -> bool:
        """Check if assembly has discovered a recursive pattern."""
        if not assembly.transformation_knowledge.operations:
            return False

        # Simple heuristic: complex position mappings might be recursive
        for op in assembly.transformation_knowledge.operations:
            if op.name == "position_mapping":
                mappings = op.params[0]
                # Multiple targets per source suggests recursion
                for targets in mappings.values():
                    if len(targets) > 2:
                        return True

        return False

    def _create_recursive_assembly(self, source_assembly: EmergentAssemblyEAN):
        """Create specialized recursive assembly."""
        if len(self.recursive_assemblies) >= self.max_recursive_assemblies:
            return

        rec_id = f"recursive_{source_assembly.id}"
        rec_assembly = ImprovedRecursiveAssembly(rec_id, self.grid_shape)

        # Transfer examples
        examples = list(source_assembly.transformation_examples)
        if rec_assembly.learn_recursive_pattern(examples):
            self.recursive_assemblies[rec_id] = rec_assembly
            logger.info(f"Created recursive assembly {rec_id}")

    def _reward_assembly(self, assembly: EmergentAssemblyEAN):
        """Reward successful assembly."""
        # Boost energy of member neurons
        for nid in assembly.member_neuron_ids:
            if nid in self.neurons:
                self.neurons[nid].activate(20.0, self.current_time)


    def solve(self, test_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Version modifiée de solve() qui priorise correctement pour les grilles 4x4
        """
        # Activate network with test input
        self._activate_for_pattern(test_input)
        
        # Pour les grilles 4x4, éviter les assemblées récursives
        if test_input.shape == (4, 4):
            # Collecter UNIQUEMENT les assemblées non-récursives spécialisées
            specialized = [a for a in self.assemblies.values() if a.is_specialized]
            
            if specialized:
                # Essayer l'intégration de patterns en premier
                integrated_mappings = self.pattern_integrator.combine_partial_patterns(specialized)
                
                if integrated_mappings:
                    coverage = self.pattern_integrator.analyze_pattern_coverage(
                        integrated_mappings, self.grid_shape
                    )
                    
                    if coverage > 0.3:  # Seuil plus bas pour 4x4
                        logger.info(f"Using integrated pattern for 4x4 (coverage: {coverage:.2f})")
                        integrated_op = AtomicOperation("position_mapping", (integrated_mappings,))
                        return integrated_op.apply(test_input)
                
                # Sinon, voter entre les prédictions
                predictions = []
                
                for assembly in specialized:
                    result, confidence = assembly.apply_transformation(test_input)
                    
                    if result is not None and confidence >= 0.1:  # Seuil très bas
                        predictions.append({
                            'result': result,
                            'confidence': confidence,
                            'assembly': assembly,
                            'score': confidence * assembly.transformation_knowledge.success_rate
                        })
                
                if predictions:
                    best = max(predictions, key=lambda x: x['score'])
                    logger.info(f"Best 4x4 prediction from {best['assembly'].id}")
                    return best['result']
        
        # Pour les autres tailles, utiliser la logique standard
        else:
            # Try recursive assemblies first (pour 3x3, 2x2)
            for rec_assembly in self.recursive_assemblies.values():
                if rec_assembly.get_transformation_confidence(test_input) > 0.7:
                    result = rec_assembly.apply_recursive_transformation(test_input)
                    logger.info(f"Using recursive assembly {rec_assembly.id}")
                    return result
            
            # Puis la logique standard...
            specialized = [a for a in self.assemblies.values() if a.is_specialized]
            
            if not specialized:
                logger.warning("No specialized assemblies available")
                return None
            
            # Reste du code identique...
            integrated_mappings = self.pattern_integrator.combine_partial_patterns(specialized)
            
            if integrated_mappings:
                coverage = self.pattern_integrator.analyze_pattern_coverage(
                    integrated_mappings, self.grid_shape
                )
                
                if coverage > 0.6:
                    logger.info(f"Using integrated pattern (coverage: {coverage:.2f})")
                    integrated_op = AtomicOperation("position_mapping", (integrated_mappings,))
                    return integrated_op.apply(test_input)
            
            # Voting fallback
            predictions = []
            
            for assembly in specialized:
                result, confidence = assembly.apply_transformation(test_input)
                
                if result is not None and confidence >= self.min_confidence_threshold:
                    predictions.append({
                        'result': result,
                        'confidence': confidence,
                        'assembly': assembly,
                        'score': confidence * assembly.transformation_knowledge.success_rate
                    })
            
            self.total_predictions += 1
            
            if not predictions:
                logger.warning("No confident predictions")
                return None
            
            best = max(predictions, key=lambda x: x['score'])
            logger.info(f"Best prediction from {best['assembly'].id} "
                    f"(score: {best['score']:.3f})")
            
            return best['result']

    def train_on_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]],
                         epochs: int = 5, steps_per_example: int = 20):
        """
        Train network on examples.

        Args:
            examples: List of (input, output) pairs
            epochs: Number of training epochs
            steps_per_example: Training steps per example
        """
        logger.info(f"Training on {len(examples)} examples for {epochs} epochs")

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Shuffle examples for each epoch
            shuffled_examples = examples.copy()
            random.shuffle(shuffled_examples)

            for i, (inp, out) in enumerate(shuffled_examples):
                # Multiple training steps per example
                for step in range(steps_per_example):
                    self.train_step(inp, out)

                # Log progress
                if (i + 1) % max(1, len(examples) // 5) == 0:
                    stats = self.get_statistics()
                    logger.info(f"  Example {i+1}/{len(examples)}: "
                               f"{stats['specialized_assemblies']} specialized, "
                               f"{stats['total_discoveries']} discoveries")

            # End of epoch summary
            stats = self.get_statistics()
            logger.info(f"Epoch {epoch + 1} complete: "
                       f"{stats['total_assemblies']} assemblies, "
                       f"{stats['specialized_assemblies']} specialized, "
                       f"avg success rate: {stats['avg_success_rate']:.2f}")

    def test_on_examples(self, test_cases: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Test network on examples.

        Args:
            test_cases: List of (input, expected_output) pairs

        Returns:
            Accuracy percentage
        """
        logger.info(f"\nTesting on {len(test_cases)} examples")
        correct = 0

        for i, (inp, expected) in enumerate(test_cases):
            result = self.solve(inp)

            if result is not None and np.array_equal(result, expected):
                correct += 1
                self.total_correct_predictions += 1
                status = "✓ CORRECT"
            else:
                status = "✗ INCORRECT"

            logger.info(f"Test {i+1}: {status}")
            logger.debug(f"  Input: {inp.flatten()}")
            logger.debug(f"  Expected: {expected.flatten()}")
            if result is not None:
                logger.debug(f"  Got: {result.flatten()}")

        accuracy = 100 * correct / len(test_cases) if test_cases else 0
        logger.info(f"\nAccuracy: {correct}/{len(test_cases)} = {accuracy:.1f}%")

        return accuracy

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get network statistics.

        Returns:
            Dictionary of statistics
        """
        specialized = [a for a in self.assemblies.values() if a.is_specialized]

        # Calculate average success rate
        if specialized:
            success_rates = [a.transformation_knowledge.success_rate for a in specialized]
            avg_success_rate = sum(success_rates) / len(success_rates)
        else:
            avg_success_rate = 0.0

        # Calculate neuron utilization
        neurons_in_assemblies = sum(
            len(a.member_neuron_ids) for a in self.assemblies.values()
        )
        neuron_utilization = (neurons_in_assemblies / self.num_neurons
                              if self.num_neurons > 0 else 0.0) # Avoid division by zero

        return {
            'total_assemblies': len(self.assemblies),
            'specialized_assemblies': len(specialized),
            'recursive_assemblies': len(self.recursive_assemblies),
            'total_discoveries': self.total_discoveries,
            'avg_success_rate': avg_success_rate,
            'neuron_utilization': neuron_utilization,
            'prediction_accuracy': (self.total_correct_predictions / self.total_predictions
                                  if self.total_predictions > 0 else 0.0),
            'current_step': self.current_step
        }

    def visualize_state(self) -> str:
        """
        Get visual representation of network state.

        Returns:
            String visualization
        """
        lines = []
        lines.append("=" * 60)
        lines.append("EAN Network State")
        lines.append("=" * 60)

        # Network overview
        stats = self.get_statistics()
        lines.append(f"Steps: {stats['current_step']}")
        lines.append(f"Assemblies: {stats['total_assemblies']} "
                    f"({stats['specialized_assemblies']} specialized)")
        lines.append(f"Recursive Assemblies: {stats['recursive_assemblies']}")
        lines.append(f"Discoveries: {stats['total_discoveries']}")
        lines.append(f"Neuron Utilization: {stats['neuron_utilization']:.1%}")
        lines.append(f"Average Success Rate: {stats['avg_success_rate']:.1%}") # Changed to .1% for consistency

        # Assembly details
        lines.append("\nSpecialized Assemblies:")
        for assembly in self.assemblies.values():
            if assembly.is_specialized:
                summary = assembly.get_state_summary()
                lines.append(f"  - {summary['id']}: "
                            f"{summary['discovery_method']}, "
                            f"success={summary['success_rate']:.2f}, "
                            f"members={summary['member_count']}")

        # Recursive assembly details
        if self.recursive_assemblies:
            lines.append("\nRecursive Assemblies:")
            for rec_id, rec_assembly in self.recursive_assemblies.items():
                if rec_assembly.recursive_rules: # Check if recursive_rules is not None
                    # Ensure 'recursion_type' exists in recursive_rules
                    recursion_type_display = rec_assembly.recursive_rules.get('recursion_type', 'N/A')
                    lines.append(f"  - {rec_id}: "
                                f"{recursion_type_display}")
                else:
                    lines.append(f"  - {rec_id}: No recursive rules defined")


        return "\n".join(lines)

    def save_state(self) -> Dict[str, Any]:
        """
        Save network state for persistence.

        Returns:
            Serializable state dictionary
        """
        state = {
            'num_neurons': self.num_neurons,
            'world_size': self.world_size,
            'grid_shape': self.grid_shape,
            'current_step': self.current_step,
            'current_time': self.current_time,
            'total_discoveries': self.total_discoveries,
            'total_predictions': self.total_predictions,
            'total_correct_predictions': self.total_correct_predictions,

            # Assembly states
            'assemblies': {},
            'recursive_assemblies': {}, # Added for completeness, though not fully saved here

            # Neuron states (simplified)
            'neuron_states': {}
        }

        # Save assembly information
        for assembly_id, assembly in self.assemblies.items():
            state['assemblies'][assembly_id] = assembly.get_state_summary()

        # Save recursive assembly information (simplified)
        for rec_id, rec_assembly in self.recursive_assemblies.items():
            state['recursive_assemblies'][rec_id] = {
                'id': rec_assembly.id,
                'is_specialized': rec_assembly.is_recursive_specialized,
                'rules_type': rec_assembly.recursive_rules.get('recursion_type', None) if rec_assembly.recursive_rules else None
            }

        # Save neuron states
        for neuron_id, neuron in self.neurons.items():
            state['neuron_states'][neuron_id] = {
                'energy': neuron.energy,
                'assembly': neuron.assembly_membership,
                'position': neuron.position
            }

        return state

    def load_state(self, state: Dict[str, Any]):
        """
        Load network state.

        Args:
            state: State dictionary to load
        """
        # Restore basic properties
        self.num_neurons = state.get('num_neurons', self.num_neurons)
        self.world_size = state.get('world_size', self.world_size)
        self.grid_shape = state.get('grid_shape', self.grid_shape)

        self.current_step = state.get('current_step', self.current_step)
        self.current_time = state.get('current_time', self.current_time)
        self.total_discoveries = state.get('total_discoveries', self.total_discoveries)
        self.total_predictions = state.get('total_predictions', self.total_predictions)
        self.total_correct_predictions = state.get('total_correct_predictions', self.total_correct_predictions)
        
        # Re-initialize parameters if they were part of the saved state or use current
        self.max_assemblies = state.get('max_assemblies', self.max_assemblies)
        self.max_recursive_assemblies = state.get('max_recursive_assemblies', self.max_recursive_assemblies)
        self.min_confidence_threshold = state.get('min_confidence_threshold', self.min_confidence_threshold)
        self.energy_decay_rate = state.get('energy_decay_rate', self.energy_decay_rate)
        self.activation_radius = state.get('activation_radius', self.activation_radius)


        # Note: Full state restoration would require re-instantiating
        # complete assembly and neuron objects with their full state, which is complex.
        # This is a simplified version for demonstration.
        # For a more robust load, you'd iterate through saved assembly/neuron data
        # and reconstruct them. For now, we primarily restore high-level stats and params.

        # Potentially re-initialize neurons if their count or properties changed significantly
        # self.neurons = self._initialize_neurons() # Be careful with this, might lose learned state

        logger.info(f"Loaded state at step {self.current_step}")


def create_ean_network(num_neurons: int = 100,
                      grid_shape: Tuple[int, int] = (2, 2)) -> IntegratedEAN:
    """
    Factory function to create EAN network.

    Args:
        num_neurons: Number of neurons
        grid_shape: Shape of transformation grids

    Returns:
        Configured EAN network
    """
    # Configure logging - moved to be configured once if multiple networks are created
    # Or ensure it's idempotent. BasicConfig is safe if called once.
    # If called multiple times, handlers might be added repeatedly.
    # A common practice is to configure logging at the application entry point.
    # For now, keeping it here as in the original.
    if not logging.getLogger().hasHandlers(): # Configure only if no handlers exist
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
    else: # If already configured, ensure our desired level for EAN.Network
        logging.getLogger("EAN.Network").setLevel(logging.INFO)


    # Create network
    network = IntegratedEAN(
        num_neurons=num_neurons,
        world_size=(10.0, 10.0),
        grid_shape=grid_shape
    )

    return network


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test
    network = create_ean_network(num_neurons=60, grid_shape=(2, 2))

    # Simple rotation examples
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[1, 0], [0, 0]]))
    ]

    # Train
    network.train_on_examples(examples, epochs=5, steps_per_example=20)

    # Test
    accuracy = network.test_on_examples(examples)

    # Show state
    print(network.visualize_state())
    print(f"\nFinal accuracy: {accuracy:.1f}%")