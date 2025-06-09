"""
Version 6 du système EAN avec intégration améliorée des patterns partiels
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import random
import logging
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("EAN")

@dataclass
class AtomicOperation:
    """Une opération atomique sur une grille"""
    name: str
    params: Tuple
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Applique l'opération à la grille"""
        if self.name == "position_mapping":
            mappings = self.params[0]  # Dict[(i,j)] -> List[(i,j)]
            result = np.zeros_like(grid)
            
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i, j] != 0:
                        source = (i, j)
                        if source in mappings:
                            for target in mappings[source]:
                                if 0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]:
                                    result[target] = grid[i, j]
            return result
        else:
            return grid.copy()
    
    def __repr__(self):
        if self.name == "position_mapping":
            return f"PositionMapping({len(self.params[0])} rules)"
        return f"{self.name}{self.params}"


class ImprovedTransformationDiscoverer:
    """Découvreur qui collecte toutes les règles de mapping observées"""
    
    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape
        
    def extract_position_mappings(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict]:
        """Extrait TOUS les mappings de positions possibles"""
        active_pixel_mappings = defaultdict(set)
        
        for inp, out in examples:
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if inp[i, j] != 0:
                        source = (i, j)
                        value = inp[i, j]
                        
                        # Trouver TOUS les pixels de même valeur dans l'output
                        targets = []
                        for ii in range(self.grid_shape[0]):
                            for jj in range(self.grid_shape[1]):
                                if out[ii, jj] == value:
                                    targets.append((ii, jj))
                        
                        if targets:
                            active_pixel_mappings[source].update(targets)
        
        # Vérifier la cohérence pour chaque mapping potentiel
        final_mappings = {}
        
        for source, targets in active_pixel_mappings.items():
            target_list = list(targets)
            is_consistent = True
            
            for inp, out in examples:
                if inp[source] != 0:
                    expected_value = inp[source]
                    
                    # Vérifier cohérence
                    for target in target_list:
                        if out[target] != expected_value:
                            is_consistent = False
                            break
                    
                    if not is_consistent:
                        break
                    
                    # Vérifier qu'il n'y a pas d'autres positions avec cette valeur
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
        
        if len(final_mappings) > 0:
            logger.info(f"Mappings découverts: {final_mappings}")
            return final_mappings
        else:
            return None
    
    def discover_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[List[AtomicOperation]]:
        """Découvre le pattern de mapping"""
        position_mappings = self.extract_position_mappings(examples)
        
        if position_mappings:
            operation = AtomicOperation("position_mapping", (position_mappings,))
            return [operation]
        
        return None


@dataclass
class TransformationKnowledge:
    """Connaissance sur les transformations découvertes"""
    operations: Optional[List[AtomicOperation]] = None
    confidence: float = 0.0
    discovery_method: str = ""
    successful_applications: int = 0
    total_attempts: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_applications / self.total_attempts


class FixedEmergentAssemblyEAN:
    """Assemblée avec meilleure gestion de la confiance et prédictions partielles"""
    
    def __init__(self, assembly_id: str, founder_ids: Set[int], birth_time: float, grid_shape: Tuple[int, int]):
        self.id = assembly_id
        self.founder_ids = frozenset(founder_ids)
        self.member_neuron_ids = set(founder_ids)
        self.birth_time = birth_time
        self.age = 0
        self.grid_shape = grid_shape
        
        # Connaissances sur les transformations
        self.transformation_knowledge = TransformationKnowledge()
        self.transformation_examples = deque(maxlen=20)
        
        # Métriques
        self.consecutive_successes = 0
        self.recent_applications = deque(maxlen=10)
        
        # État
        self.is_specialized = False
        self.protection_counter = 0
        self.stability_score = 1.0
        
        # Découvreur
        self.discoverer = ImprovedTransformationDiscoverer(grid_shape)
        
    def observe_transformation(self, input_pattern: np.ndarray, output_pattern: np.ndarray):
        """Observe un exemple de transformation"""
        self.transformation_examples.append((input_pattern.copy(), output_pattern.copy()))
    
    def attempt_discovery(self) -> bool:
        """Tente de découvrir une transformation"""
        if len(self.transformation_examples) < 4:
            return False
        
        # Utiliser TOUS les exemples pour avoir le pattern complet
        all_examples = list(self.transformation_examples)
        
        # Tenter la découverte
        operations = self.discoverer.discover_pattern(all_examples)
        
        if operations:
            self.transformation_knowledge.operations = operations
            self.transformation_knowledge.discovery_method = operations[0].name
            self.transformation_knowledge.confidence = 0.9
            self.is_specialized = True
            
            logger.info(f"{self.id} découvert: {operations[0]}")
            return True
        
        return False
    
    def apply_transformation(self, input_pattern: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Applique la transformation avec gestion de confiance améliorée"""
        if not self.is_specialized or not self.transformation_knowledge.operations:
            return None, 0.0
        
        try:
            result = input_pattern.copy()
            for operation in self.transformation_knowledge.operations:
                result = operation.apply(result)
            
            # Calculer une confiance basée sur la complétude du pattern
            base_confidence = self.transformation_knowledge.confidence
            
            # Vérifier si notre pattern peut traiter ce input
            has_active_pixels = np.any(input_pattern != 0)
            can_transform = False
            
            if has_active_pixels and self.transformation_knowledge.operations:
                mappings = self.transformation_knowledge.operations[0].params[0]
                # Vérifier si on a des règles pour les pixels actifs de l'input
                for i in range(input_pattern.shape[0]):
                    for j in range(input_pattern.shape[1]):
                        if input_pattern[i, j] != 0:
                            if (i, j) in mappings:
                                can_transform = True
                                break
            
            if not can_transform:
                return None, 0.0
            
            # Ajuster la confiance selon l'historique
            if len(self.recent_applications) > 0:
                recent_success_rate = sum(self.recent_applications) / len(self.recent_applications)
                adjusted_confidence = base_confidence * (0.6 + 0.4 * recent_success_rate)
            else:
                adjusted_confidence = base_confidence * 0.8
            
            # Bonus pour succès consécutifs
            if self.consecutive_successes > 2:
                adjusted_confidence = min(1.0, adjusted_confidence * 1.1)
            
            return result, adjusted_confidence
            
        except Exception as e:
            logger.error(f"{self.id} erreur d'application: {e}")
            return None, 0.0
    
    def record_application_result(self, success: bool):
        """Enregistre le résultat d'une application"""
        self.transformation_knowledge.total_attempts += 1
        self.recent_applications.append(success)
        
        if success:
            self.transformation_knowledge.successful_applications += 1
            self.consecutive_successes += 1
            self.protection_counter = 200
            self.transformation_knowledge.confidence = min(1.0, 
                self.transformation_knowledge.confidence * 1.02)
        else:
            self.consecutive_successes = 0
            self.transformation_knowledge.confidence *= 0.95
    
    def should_dissolve(self) -> bool:
        """Critères de dissolution"""
        if self.is_specialized and self.transformation_knowledge.success_rate > 0.4:
            return False
        
        if self.protection_counter > 0:
            return False
        
        if self.age > 200 and not self.is_specialized:
            return True
        
        if self.is_specialized and self.transformation_knowledge.total_attempts > 20:
            if self.transformation_knowledge.success_rate < 0.2:
                return True
        
        return False
    
    def update(self):
        """Mise à jour périodique"""
        self.age += 1
        if self.protection_counter > 0:
            self.protection_counter -= 1
        
        # Retenter la découverte avec plus d'exemples
        if not self.is_specialized and self.age % 15 == 0 and len(self.transformation_examples) >= 8:
            self.attempt_discovery()


class PatternIntegrator:
    """Classe pour intégrer les patterns partiels de différentes assemblées"""
    
    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape
    
    def combine_partial_patterns(self, specialized_assemblies: List) -> Optional[Dict]:
        """Combine les patterns partiels en un pattern complet"""
        if not specialized_assemblies:
            return None
        
        # Collecter tous les mappings
        all_mappings = {}
        
        for assembly in specialized_assemblies:
            if assembly.transformation_knowledge.operations:
                operation = assembly.transformation_knowledge.operations[0]
                if operation.name == "position_mapping":
                    partial_mappings = operation.params[0]
                    for source, targets in partial_mappings.items():
                        if source not in all_mappings:
                            all_mappings[source] = targets
                        else:
                            # Vérifier la cohérence
                            if all_mappings[source] != targets:
                                logger.warning(f"Conflit de mapping pour {source}: "
                                             f"{all_mappings[source]} vs {targets}")
        
        if all_mappings:
            logger.info(f"Pattern intégré: {len(all_mappings)} règles combinées")
            return all_mappings
        
        return None


class FixedIntegratedEAN:
    """Réseau EAN avec intégration de patterns et vote amélioré"""
    
    def __init__(self, num_neurons: int = 100, world_size: Tuple[float, float] = (10.0, 10.0),
                 grid_shape: Tuple[int, int] = (2, 2)):
        self.neurons = self._initialize_neurons(num_neurons, world_size)
        self.assemblies: Dict[str, FixedEmergentAssemblyEAN] = {}
        self.current_step = 0
        self.current_time = 0.0
        self.world_size = world_size
        self.grid_shape = grid_shape
        
        # Intégrateur de patterns
        self.pattern_integrator = PatternIntegrator(grid_shape)
        
        # Métriques
        self.total_discoveries = 0
        self.total_correct_predictions = 0
        self.total_predictions = 0
        
        # Paramètres
        self.max_assemblies = 10
        self.assembly_formation_cooldown = 0
        self.min_confidence_threshold = 0.25  # Seuil encore plus bas
        
        logger.info(f"Fixed EAN V6 initialisé avec {num_neurons} neurones")
    
    def _initialize_neurons(self, num_neurons: int, world_size: Tuple[float, float]) -> Dict:
        """Initialise les neurones"""
        class SimpleNeuron:
            def __init__(self, neuron_id, position):
                self.id = neuron_id
                self.position = position
                self.energy = random.uniform(40.0, 60.0)
                self.assembly_membership = None
                self.neighboring_neurons = set()
                
            def activate(self, energy, time):
                self.energy = min(100.0, self.energy + energy)
                
            def decay(self, rate):
                self.energy = max(0.0, self.energy - rate)
                
            def is_available_for_assembly(self):
                return self.energy > 30.0 and self.assembly_membership is None
                
            def calculate_distance_to(self, other_pos):
                return math.sqrt((self.position[0] - other_pos[0])**2 + 
                               (self.position[1] - other_pos[1])**2)
        
        neurons = {}
        for i in range(num_neurons):
            pos = (random.uniform(0, world_size[0]), 
                   random.uniform(0, world_size[1]))
            neurons[i] = SimpleNeuron(i, pos)
        
        # Calculer les voisins
        for n1 in neurons.values():
            for n2 in neurons.values():
                if n1.id != n2.id:
                    dist = n1.calculate_distance_to(n2.position)
                    if dist < 4.0:
                        n1.neighboring_neurons.add(n2.id)
        
        return neurons
    
    def _form_assemblies_near_activation(self, activation_center: Tuple[float, float]):
        """Forme des assemblées près du centre d'activation"""
        if self.assembly_formation_cooldown > 0 or len(self.assemblies) >= self.max_assemblies:
            return
        
        # Trouver les neurones disponibles proches
        nearby_neurons = []
        for neuron in self.neurons.values():
            if neuron.is_available_for_assembly():
                dist = neuron.calculate_distance_to(activation_center)
                if dist < 4.0 and neuron.energy > 45.0:
                    nearby_neurons.append((neuron.id, dist))
        
        # Former une assemblée si assez de neurones
        if len(nearby_neurons) >= 5:
            nearby_neurons.sort(key=lambda x: x[1])
            founder_ids = {n[0] for n in nearby_neurons[:8]}
            
            assembly_id = f"assembly_{self.current_step}_{len(self.assemblies)}"
            assembly = FixedEmergentAssemblyEAN(
                assembly_id, founder_ids, self.current_time, self.grid_shape)
            
            # Marquer les neurones
            for nid in founder_ids:
                self.neurons[nid].assembly_membership = assembly_id
                self.neurons[nid].activate(15.0, self.current_time)
            
            self.assemblies[assembly_id] = assembly
            self.assembly_formation_cooldown = 3
            logger.debug(f"Formé {assembly_id}")
    
    def _activate_for_pattern(self, pattern: np.ndarray):
        """Active les neurones selon le pattern"""
        h, w = pattern.shape
        
        for i in range(h):
            for j in range(w):
                if pattern[i, j] > 0:
                    world_x = (j + 0.5) * self.world_size[0] / w
                    world_y = (i + 0.5) * self.world_size[1] / h
                    
                    for neuron in self.neurons.values():
                        dist = neuron.calculate_distance_to((world_x, world_y))
                        if dist < 4.0:
                            strength = 80.0 * (1.0 - dist / 4.0) * pattern[i, j]
                            neuron.activate(strength, self.current_time)
                    
                    self._form_assemblies_near_activation((world_x, world_y))
    
    def train_step(self, input_pattern: np.ndarray, output_pattern: np.ndarray):
        """Step d'entraînement avec focus sur la découverte complète"""
        self.current_step += 1
        self.current_time += 0.1
        
        if self.assembly_formation_cooldown > 0:
            self.assembly_formation_cooldown -= 1
        
        # Décroissance d'énergie
        for neuron in self.neurons.values():
            neuron.decay(0.015)
        
        # Activer selon le pattern
        self._activate_for_pattern(input_pattern)
        
        # Phase 1: Observation
        for assembly in self.assemblies.values():
            assembly.observe_transformation(input_pattern, output_pattern)
        
        # Phase 2: Découverte et test
        assemblies_to_remove = []
        
        for assembly_id, assembly in self.assemblies.items():
            # Tenter la découverte plus fréquemment
            if not assembly.is_specialized and len(assembly.transformation_examples) >= 6:
                if assembly.attempt_discovery():
                    self.total_discoveries += 1
            
            # Test immédiat des assemblées spécialisées
            if assembly.is_specialized:
                result, confidence = assembly.apply_transformation(input_pattern)
                
                if result is not None:
                    is_correct = np.array_equal(result, output_pattern)
                    assembly.record_application_result(is_correct)
                    
                    if is_correct:
                        # Récompenser
                        for nid in assembly.member_neuron_ids:
                            if nid in self.neurons:
                                self.neurons[nid].activate(20.0, self.current_time)
            
            assembly.update()
            
            if assembly.should_dissolve():
                assemblies_to_remove.append(assembly_id)
        
        # Retirer les assemblées à dissoudre
        for assembly_id in assemblies_to_remove:
            self._dissolve_assembly(assembly_id)
    
    def _dissolve_assembly(self, assembly_id: str):
        """Dissout une assemblée"""
        assembly = self.assemblies[assembly_id]
        for nid in assembly.member_neuron_ids:
            if nid in self.neurons:
                self.neurons[nid].assembly_membership = None
        del self.assemblies[assembly_id]
    
    def solve(self, test_input: np.ndarray) -> Optional[np.ndarray]:
        """Résout avec intégration de patterns et vote intelligent"""
        self._activate_for_pattern(test_input)
        
        # Collecter les assemblées spécialisées
        specialized_assemblies = [a for a in self.assemblies.values() if a.is_specialized]
        
        if not specialized_assemblies:
            logger.warning("Aucune assemblée spécialisée disponible")
            return None
        
        # Stratégie 1: Essayer l'intégration de patterns
        integrated_mappings = self.pattern_integrator.combine_partial_patterns(specialized_assemblies)
        
        if integrated_mappings:
            logger.info(f"Utilisation du pattern intégré avec {len(integrated_mappings)} règles")
            # Appliquer le pattern intégré
            integrated_op = AtomicOperation("position_mapping", (integrated_mappings,))
            result = integrated_op.apply(test_input)
            return result
        
        # Stratégie 2: Vote des prédictions individuelles
        predictions = []
        
        for assembly in specialized_assemblies:
            result, confidence = assembly.apply_transformation(test_input)
            
            if result is not None and confidence >= self.min_confidence_threshold:
                predictions.append({
                    'result': result,
                    'confidence': confidence,
                    'assembly': assembly.id,
                    'success_rate': assembly.transformation_knowledge.success_rate
                })
                logger.debug(f"Prédiction {assembly.id}: conf={confidence:.2f}")
        
        self.total_predictions += 1
        
        if not predictions:
            logger.warning("Aucune prédiction confiante")
            return None
        
        # Prendre la meilleure prédiction
        best_prediction = max(predictions, key=lambda x: x['confidence'] * (0.5 + 0.5 * x['success_rate']))
        logger.info(f"Meilleure prédiction: {best_prediction['assembly']} "
                   f"(conf: {best_prediction['confidence']:.2f})")
        
        return best_prediction['result']
    
    def train_on_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                         epochs: int = 5, steps_per_example: int = 20):
        """Entraîne le réseau"""
        logger.info(f"Entraînement: {len(examples)} exemples, {epochs} époques")
        
        for epoch in range(epochs):
            logger.info(f"\nÉpoque {epoch + 1}/{epochs}")
            
            shuffled_examples = examples.copy()
            random.shuffle(shuffled_examples)
            
            for i, (inp, out) in enumerate(shuffled_examples):
                for step in range(steps_per_example):
                    self.train_step(inp, out)
                
                # Afficher les stats
                stats = self.get_statistics()
                if stats['specialized_assemblies'] > 0:
                    logger.info(f"  Ex{i+1}: {stats['specialized_assemblies']} spécialisées, "
                               f"{stats['total_discoveries']} découvertes")
    
    def test_on_examples(self, test_cases: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Teste le réseau"""
        logger.info("\nPhase de test:")
        correct = 0
        
        for i, (inp, expected) in enumerate(test_cases):
            result = self.solve(inp)
            success = np.array_equal(result, expected) if result is not None else False
            
            if success:
                correct += 1
                self.total_correct_predictions += 1
            
            logger.info(f"Test {i+1}: {inp.flatten()} -> {expected.flatten()}: "
                       f"{'SUCCÈS' if success else 'ÉCHEC'}")
            
            if result is not None and not success:
                logger.info(f"  Obtenu: {result.flatten()}")
        
        accuracy = 100 * correct / len(test_cases) if test_cases else 0
        logger.info(f"\nPrécision: {correct}/{len(test_cases)} = {accuracy:.0f}%")
        
        return accuracy
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques"""
        specialized = [a for a in self.assemblies.values() if a.is_specialized]
        
        total_success_rate = 0.0
        if specialized:
            success_rates = [a.transformation_knowledge.success_rate for a in specialized]
            total_success_rate = sum(success_rates) / len(success_rates)
        
        return {
            'total_assemblies': len(self.assemblies),
            'specialized_assemblies': len(specialized),
            'total_discoveries': self.total_discoveries,
            'avg_success_rate': total_success_rate,
            'prediction_accuracy': self.total_correct_predictions / self.total_predictions 
                                  if self.total_predictions > 0 else 0.0
        }


def test_ean_v6():
    """Test du système EAN V6 avec intégration"""
    logger.info("=" * 60)
    logger.info("Test du Système EAN V6 - Intégration de Patterns")
    logger.info("=" * 60)
    
    # Créer le réseau
    network = FixedIntegratedEAN(num_neurons=60, grid_shape=(2, 2))
    
    # Exemples d'entraînement
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))
    ]
    
    # Entraînement
    network.train_on_examples(examples, epochs=6, steps_per_example=20)
    
    # Statistiques
    stats = network.get_statistics()
    logger.info(f"\nStatistiques finales:")
    logger.info(f"  Assemblées spécialisées: {stats['specialized_assemblies']}")
    logger.info(f"  Découvertes totales: {stats['total_discoveries']}")
    
    if stats['specialized_assemblies'] == 0:
        logger.error("Aucune assemblée spécialisée!")
        return False
    
    # Test
    accuracy = network.test_on_examples(examples)
    
    if accuracy > 0:
        # Test de généralisation
        logger.info("\nTest de généralisation:")
        gen_tests = [
            (np.array([[2, 0], [0, 0]]), np.array([[0, 2], [2, 0]])),
            (np.array([[0, 3], [0, 0]]), np.array([[0, 0], [0, 3]]))
        ]
        
        gen_accuracy = network.test_on_examples(gen_tests)
        return accuracy >= 50  # Au moins 50% de réussite
    
    return False


if __name__ == "__main__":
    success = test_ean_v6()
    logger.info(f"\nRésultat final: {'SUCCÈS' if success else 'ÉCHEC'}")
    exit(0 if success else 1)