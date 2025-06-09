"""
Version corrigée du système EAN intégré qui gère correctement la duplication
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

# Importer les classes corrigées
from fixed_discovery_system import AtomicOperation, ImprovedTransformationDiscoverer


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
    """Assemblée corrigée avec meilleure gestion de la confiance"""
    
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
        self.recent_applications = deque(maxlen=10)  # True/False pour succès/échec
        
        # État
        self.is_specialized = False
        self.protection_counter = 0
        self.stability_score = 1.0
        
        # Découvreur amélioré
        self.discoverer = ImprovedTransformationDiscoverer(grid_shape)
        
    def observe_transformation(self, input_pattern: np.ndarray, output_pattern: np.ndarray):
        """Observe un exemple de transformation"""
        self.transformation_examples.append((input_pattern.copy(), output_pattern.copy()))
    
    def attempt_discovery(self) -> bool:
        """Tente de découvrir une transformation"""
        if len(self.transformation_examples) < 4:
            return False
        
        # Utiliser les exemples récents
        recent_examples = list(self.transformation_examples)[-10:]
        
        # Tenter la découverte
        operations = self.discoverer.discover_pattern(recent_examples)
        
        if operations:
            self.transformation_knowledge.operations = operations
            self.transformation_knowledge.discovery_method = operations[0].name
            self.transformation_knowledge.confidence = 0.9  # Haute confiance initiale
            self.is_specialized = True
            
            logger.info(f"{self.id} discovered transformation: {operations[0]}")
            return True
        
        return False
    
    def apply_transformation(self, input_pattern: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Applique la transformation avec gestion améliorée de la confiance"""
        if not self.is_specialized or not self.transformation_knowledge.operations:
            return None, 0.0
        
        # Appliquer les opérations
        try:
            result = input_pattern.copy()
            for operation in self.transformation_knowledge.operations:
                result = operation.apply(result)
            
            # La confiance est basée sur l'historique récent ET le taux de succès global
            base_confidence = self.transformation_knowledge.confidence
            
            # Ajuster selon les applications récentes
            if len(self.recent_applications) > 0:
                recent_success_rate = sum(self.recent_applications) / len(self.recent_applications)
                adjusted_confidence = base_confidence * (0.5 + 0.5 * recent_success_rate)
            else:
                adjusted_confidence = base_confidence * 0.7  # Réduction initiale
            
            # Bonus pour les succès consécutifs
            if self.consecutive_successes > 3:
                adjusted_confidence = min(1.0, adjusted_confidence * 1.1)
            
            return result, adjusted_confidence
            
        except Exception as e:
            logger.error(f"{self.id} failed to apply transformation: {e}")
            return None, 0.0
    
    def record_application_result(self, success: bool):
        """Enregistre le résultat d'une application"""
        self.transformation_knowledge.total_attempts += 1
        self.recent_applications.append(success)
        
        if success:
            self.transformation_knowledge.successful_applications += 1
            self.consecutive_successes += 1
            self.protection_counter = 200
            self.stability_score = min(1.0, self.stability_score + 0.1)
            
            # Augmenter la confiance
            self.transformation_knowledge.confidence = min(1.0, 
                self.transformation_knowledge.confidence * 1.05)
        else:
            self.consecutive_successes = 0
            self.stability_score = max(0.0, self.stability_score - 0.05)
            
            # Réduire la confiance mais pas trop drastiquement
            self.transformation_knowledge.confidence *= 0.95
    
    def should_dissolve(self) -> bool:
        """Critères de dissolution adaptés"""
        # Protéger les assemblées spécialisées performantes
        if self.is_specialized and self.transformation_knowledge.success_rate > 0.5:
            return False
        
        if self.protection_counter > 0:
            return False
        
        # Dissoudre si trop vieille sans spécialisation
        if self.age > 150 and not self.is_specialized:
            return True
        
        # Dissoudre si spécialisée mais très mauvaise performance
        if self.is_specialized and self.transformation_knowledge.total_attempts > 50:
            if self.transformation_knowledge.success_rate < 0.1:
                return True
        
        return False
    
    def update(self):
        """Mise à jour périodique"""
        self.age += 1
        if self.protection_counter > 0:
            self.protection_counter -= 1
        
        # Tenter la découverte périodiquement si pas encore spécialisée
        if not self.is_specialized and self.age % 30 == 0:
            self.attempt_discovery()


class FixedIntegratedEAN:
    """Réseau EAN corrigé avec meilleure orchestration"""
    
    def __init__(self, num_neurons: int = 100, world_size: Tuple[float, float] = (10.0, 10.0),
                 grid_shape: Tuple[int, int] = (2, 2)):
        self.neurons = self._initialize_neurons(num_neurons, world_size)
        self.assemblies: Dict[str, FixedEmergentAssemblyEAN] = {}
        self.current_step = 0
        self.current_time = 0.0
        self.world_size = world_size
        self.grid_shape = grid_shape
        
        # Métriques
        self.total_discoveries = 0
        self.total_correct_predictions = 0
        self.total_predictions = 0
        
        # Paramètres
        self.max_assemblies = 15
        self.assembly_formation_cooldown = 0
        self.min_confidence_threshold = 0.4  # Seuil de confiance pour les prédictions
        
        logger.info(f"Fixed Integrated EAN initialized with {num_neurons} neurons")
    
    def _initialize_neurons(self, num_neurons: int, world_size: Tuple[float, float]) -> Dict:
        """Initialise les neurones"""
        # Classe simple de neurone pour les tests
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
            self.assembly_formation_cooldown = 5
            logger.debug(f"Formed {assembly_id} with {len(founder_ids)} neurons")
    
    def _activate_for_pattern(self, pattern: np.ndarray):
        """Active les neurones selon le pattern"""
        h, w = pattern.shape
        
        for i in range(h):
            for j in range(w):
                if pattern[i, j] > 0:
                    # Position dans le monde neuronal
                    world_x = (j + 0.5) * self.world_size[0] / w
                    world_y = (i + 0.5) * self.world_size[1] / h
                    
                    # Activer les neurones proches
                    for neuron in self.neurons.values():
                        dist = neuron.calculate_distance_to((world_x, world_y))
                        if dist < 4.0:
                            strength = 80.0 * (1.0 - dist / 4.0) * pattern[i, j]
                            neuron.activate(strength, self.current_time)
                    
                    # Former des assemblées si nécessaire
                    self._form_assemblies_near_activation((world_x, world_y))
    
    def train_step(self, input_pattern: np.ndarray, output_pattern: np.ndarray):
        """Step d'entraînement amélioré"""
        self.current_step += 1
        self.current_time += 0.1
        
        # Mise à jour du cooldown
        if self.assembly_formation_cooldown > 0:
            self.assembly_formation_cooldown -= 1
        
        # Décroissance d'énergie
        for neuron in self.neurons.values():
            neuron.decay(0.02)
        
        # Activer selon le pattern d'entrée
        self._activate_for_pattern(input_pattern)
        
        # Phase 1: Observation
        for assembly in self.assemblies.values():
            assembly.observe_transformation(input_pattern, output_pattern)
        
        # Phase 2: Découverte et Application
        assemblies_to_remove = []
        successful_assemblies = []
        
        for assembly_id, assembly in self.assemblies.items():
            # Tenter la découverte si nécessaire
            if not assembly.is_specialized and len(assembly.transformation_examples) >= 4:
                if assembly.attempt_discovery():
                    self.total_discoveries += 1
            
            # Appliquer la transformation si spécialisée
            if assembly.is_specialized:
                result, confidence = assembly.apply_transformation(input_pattern)
                
                if result is not None:
                    # Vérifier si c'est correct
                    is_correct = np.array_equal(result, output_pattern)
                    assembly.record_application_result(is_correct)
                    
                    if is_correct:
                        successful_assemblies.append((assembly, confidence))
                        # Récompenser énergétiquement
                        for nid in assembly.member_neuron_ids:
                            if nid in self.neurons:
                                self.neurons[nid].activate(30.0, self.current_time)
            
            # Mise à jour et vérification de dissolution
            assembly.update()
            
            if assembly.should_dissolve():
                assemblies_to_remove.append(assembly_id)
        
        # Retirer les assemblées à dissoudre
        for assembly_id in assemblies_to_remove:
            self._dissolve_assembly(assembly_id)
        
        # Logger les succès
        if successful_assemblies:
            best_assembly, best_conf = max(successful_assemblies, key=lambda x: x[1])
            logger.debug(f"Best result from {best_assembly.id} (conf: {best_conf:.2f})")
    
    def _dissolve_assembly(self, assembly_id: str):
        """Dissout une assemblée"""
        assembly = self.assemblies[assembly_id]
        for nid in assembly.member_neuron_ids:
            if nid in self.neurons:
                self.neurons[nid].assembly_membership = None
        del self.assemblies[assembly_id]
        logger.debug(f"Dissolved {assembly_id}")
    
    def solve(self, test_input: np.ndarray) -> Optional[np.ndarray]:
        """Résout un pattern avec vote pondéré"""
        # Activer le réseau
        self._activate_for_pattern(test_input)
        
        # Collecter toutes les prédictions
        predictions = []
        
        for assembly in self.assemblies.values():
            if assembly.is_specialized:
                result, confidence = assembly.apply_transformation(test_input)
                
                # Accepter les prédictions au-dessus du seuil
                if result is not None and confidence >= self.min_confidence_threshold:
                    predictions.append({
                        'result': result,
                        'confidence': confidence,
                        'assembly': assembly.id,
                        'success_rate': assembly.transformation_knowledge.success_rate
                    })
        
        self.total_predictions += 1
        
        if not predictions:
            logger.warning("No confident predictions from assemblies")
            return None
        
        # Stratégie de vote améliorée
        # Grouper les prédictions identiques
        result_groups = defaultdict(list)
        for pred in predictions:
            result_key = pred['result'].tobytes()
            result_groups[result_key].append(pred)
        
        # Calculer le score pour chaque résultat unique
        best_score = -1
        best_result = None
        
        for result_key, group in result_groups.items():
            # Score = somme des confidences * moyenne des taux de succès
            total_confidence = sum(p['confidence'] for p in group)
            avg_success_rate = sum(p['success_rate'] for p in group) / len(group)
            score = total_confidence * (0.5 + 0.5 * avg_success_rate)
            
            if score > best_score:
                best_score = score
                best_result = group[0]['result']
                best_assembly = group[0]['assembly']
        
        if best_result is not None:
            logger.info(f"Best prediction from {best_assembly} and {len(result_groups[best_result.tobytes()])-1} others "
                       f"(score: {best_score:.2f})")
        
        return best_result
    
    def train_on_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                         epochs: int = 3, steps_per_example: int = 30):
        """Entraîne le réseau sur des exemples"""
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Mélanger les exemples
            shuffled_examples = examples.copy()
            random.shuffle(shuffled_examples)
            
            for i, (inp, out) in enumerate(shuffled_examples):
                logger.info(f"Training example {i+1}: {inp.flatten()} -> {out.flatten()}")
                
                # Plusieurs steps par exemple
                for _ in range(steps_per_example):
                    self.train_step(inp, out)
                
                # Afficher les statistiques périodiquement
                if (i + 1) % 2 == 0 or i == len(shuffled_examples) - 1:
                    stats = self.get_statistics()
                    if stats['specialized_assemblies'] > 0:
                        logger.info(f"  Specialized: {stats['specialized_assemblies']}, "
                                   f"Discoveries: {stats['total_discoveries']}, "
                                   f"Avg success rate: {stats['avg_success_rate']:.2f}")
    
    def test_on_examples(self, test_cases: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Teste le réseau et retourne la précision"""
        logger.info("\nTesting phase:")
        correct = 0
        
        for inp, expected in test_cases:
            result = self.solve(inp)
            success = np.array_equal(result, expected) if result is not None else False
            
            if success:
                correct += 1
                self.total_correct_predictions += 1
            
            logger.info(f"Test {inp.flatten()} -> {expected.flatten()}: "
                       f"{'PASS' if success else 'FAIL'}")
            
            if result is not None and not success:
                logger.info(f"  Got: {result.flatten()}")
        
        accuracy = 100 * correct / len(test_cases) if test_cases else 0
        logger.info(f"\nAccuracy: {correct}/{len(test_cases)} = {accuracy:.0f}%")
        
        return accuracy
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques détaillées"""
        specialized = [a for a in self.assemblies.values() if a.is_specialized]
        
        # Calculer le taux de succès moyen
        total_success_rate = 0.0
        if specialized:
            success_rates = [a.transformation_knowledge.success_rate for a in specialized]
            total_success_rate = sum(success_rates) / len(success_rates)
        
        stats = {
            'total_assemblies': len(self.assemblies),
            'specialized_assemblies': len(specialized),
            'total_discoveries': self.total_discoveries,
            'avg_success_rate': total_success_rate,
            'prediction_accuracy': self.total_correct_predictions / self.total_predictions 
                                  if self.total_predictions > 0 else 0.0
        }
        
        return stats


def test_fixed_integrated_ean():
    """Test du système EAN corrigé"""
    logger.info("=" * 60)
    logger.info("Testing Fixed Integrated EAN")
    logger.info("=" * 60)
    
    # Créer le réseau
    network = FixedIntegratedEAN(num_neurons=100, grid_shape=(2, 2))
    
    # Exemples d'entraînement (rotation ARC avec duplication)
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))
    ]
    
    # Entraînement
    network.train_on_examples(examples, epochs=3, steps_per_example=30)
    
    # Test sur les mêmes exemples
    accuracy = network.test_on_examples(examples)
    
    # Test de généralisation
    logger.info("\n" + "=" * 60)
    logger.info("Generalization Test (different values):")
    
    generalization_tests = [
        (np.array([[2, 0], [0, 0]]), np.array([[0, 2], [2, 0]])),
        (np.array([[0, 3], [0, 0]]), np.array([[0, 0], [0, 3]]))
    ]
    
    gen_accuracy = network.test_on_examples(generalization_tests)
    
    # Statistiques finales
    final_stats = network.get_statistics()
    logger.info(f"\nFinal Network Statistics:")
    logger.info(f"  Total assemblies: {final_stats['total_assemblies']}")
    logger.info(f"  Specialized assemblies: {final_stats['specialized_assemblies']}")
    logger.info(f"  Total discoveries: {final_stats['total_discoveries']}")
    logger.info(f"  Average success rate: {final_stats['avg_success_rate']:.2f}")
    logger.info(f"  Overall prediction accuracy: {final_stats['prediction_accuracy']:.2f}")
    
    return accuracy > 0


def test_robustness():
    """Test de robustesse avec différentes configurations"""
    logger.info("\n" + "=" * 60)
    logger.info("Robustness Test")
    logger.info("=" * 60)
    
    # Test avec moins de neurones
    configs = [
        {'neurons': 50, 'epochs': 2, 'steps': 20},
        {'neurons': 100, 'epochs': 3, 'steps': 30},
        {'neurons': 150, 'epochs': 3, 'steps': 40}
    ]
    
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))
    ]
    
    for config in configs:
        logger.info(f"\nTesting with {config['neurons']} neurons...")
        network = FixedIntegratedEAN(num_neurons=config['neurons'])
        network.train_on_examples(examples, 
                                epochs=config['epochs'], 
                                steps_per_example=config['steps'])
        accuracy = network.test_on_examples(examples)
        logger.info(f"Configuration accuracy: {accuracy:.0f}%")


if __name__ == "__main__":
    # Import nécessaire si modules séparés
    import sys
    sys.path.append('.')
    
    # Test principal
    success = test_fixed_integrated_ean()
    
    # Test de robustesse
    if success:
        test_robustness()
    
    exit(0 if success else 1)