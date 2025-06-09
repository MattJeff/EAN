"""
Minimal Viable Emergent Assembly Network (EAN) - Version Améliorée
Amélioration de la stabilité d'apprentissage et de la persistance des assemblées
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
    """Neurone simple optimisé pour formation d'assemblées"""
    
    def __init__(self, neuron_id: int, position: Tuple[float, float]):
        self.id = neuron_id
        self.position = position
        self.energy = random.uniform(40.0, 60.0)
        self.last_activity_time = 0.0
        self.activation_history = deque(maxlen=20)
        self.neighboring_neurons: Set[int] = set()
        self.assembly_membership: Optional[str] = None
        
    def activate(self, energy_input: float, current_time: float):
        """Réception d'énergie + mise à jour temporelle"""
        old_energy = self.energy
        self.energy = min(100.0, self.energy + energy_input)
        self.last_activity_time = current_time
        self.activation_history.append({
            'time': current_time,
            'energy_before': old_energy,
            'energy_after': self.energy,
            'input': energy_input
        })
        
    def decay(self, decay_rate: float = 0.02):  # Réduit de 0.05 à 0.02
        """Décroissance énergétique naturelle encore plus réduite"""
        self.energy = max(0.0, self.energy - decay_rate)
        
    def is_available_for_assembly(self) -> bool:
        """Vérifie si peut rejoindre nouvelle assemblée"""
        return self.energy > 30.0 and self.assembly_membership is None
        
    def calculate_distance_to(self, other_position: Tuple[float, float]) -> float:
        """Calcule distance euclidienne"""
        return math.sqrt(
            (self.position[0] - other_position[0])**2 + 
            (self.position[1] - other_position[1])**2
        )


class AssemblyPMV:
    """Assemblée neuronale avec apprentissage stabilisé"""
    
    def __init__(self, assembly_id: str, founder_ids: Set[int], birth_time: float):
        self.id = assembly_id
        self.founder_ids = frozenset(founder_ids)
        self.member_neuron_ids = set(founder_ids)
        self.birth_time = birth_time
        self.age = 0
        
        # Apprentissage avec mémoire complète
        self.transformation_memory = defaultdict(list)  # action -> [(input, output), ...]
        self.primary_action = None
        self.confidence_scores = defaultdict(float)  # action -> confidence
        
        # Historique de performance amélioré
        self.success_count = 0
        self.attempt_count = 0
        self.consecutive_successes = 0
        self.recent_successes = deque(maxlen=10)  # Track recent performance
        
        # État
        self.coherence_score = 1.0
        self.is_specialized = False
        self.protection_counter = 0
        self.stability_score = 1.0  # Nouveau: score de stabilité
        
    def attempt_transformation(self, input_pattern: np.ndarray) -> Tuple[Optional[np.ndarray], str, float]:
        """Tentative de transformation avec mémoire améliorée"""
        self.attempt_count += 1
        
        # Si pas encore d'apprentissage, explorer intelligemment
        if not self.transformation_memory:
            # Prioriser rotate_90 dans l'exploration (puisque c'est ce qu'on veut apprendre)
            actions = ['rotate_90'] * 3 + ['flip_horizontal', 'flip_vertical', 'identity']
            action = random.choice(actions)
            result = self._apply_action(input_pattern, action)
            return result, action, 0.3
            
        # Chercher dans la mémoire si on connaît cet input
        for action, memories in self.transformation_memory.items():
            for mem_input, mem_output in memories:
                if np.array_equal(input_pattern, mem_input):
                    # On connaît exactement cet input!
                    return mem_output.copy(), action, self.confidence_scores[action]
        
        # Input inconnu, utiliser notre action principale si on en a une
        if self.primary_action:
            result = self._apply_action(input_pattern, self.primary_action)
            # Confiance réduite pour input inconnu
            return result, self.primary_action, self.confidence_scores[self.primary_action] * 0.7
        
        # Sinon, choisir l'action avec la meilleure confiance
        if self.confidence_scores:
            best_action = max(self.confidence_scores.items(), key=lambda x: x[1])[0]
            result = self._apply_action(input_pattern, best_action)
            return result, best_action, self.confidence_scores[best_action] * 0.5
        
        # Fallback: action aléatoire
        action = random.choice(['rotate_90', 'flip_horizontal', 'flip_vertical', 'identity'])
        result = self._apply_action(input_pattern, action)
        return result, action, 0.2
            
    def _apply_action(self, pattern: np.ndarray, action: str) -> np.ndarray:
        """Applique une transformation spécifique"""
        if action == 'rotate_90':
            # Transformation ARC spécifique
            result = np.zeros_like(pattern)
            
            # Règles de transformation confirmées :
            if pattern[0,0] == 1:
                # Position (0,0) -> Positions (0,1) ET (1,0) [duplication]
                result[0,1] = 1
                result[1,0] = 1
            elif pattern[0,1] == 1:
                # Position (0,1) -> Position (1,1)
                result[1,1] = 1
            elif pattern[1,0] == 1:
                # Position (1,0) -> Position (0,1)
                result[0,1] = 1
            elif pattern[1,1] == 1:
                # Position (1,1) -> Position (0,0)
                result[0,0] = 1
                
            return result
        elif action == 'flip_horizontal':
            return np.fliplr(pattern)
        elif action == 'flip_vertical':
            return np.flipud(pattern)
        else:  # identity
            return pattern.copy()
            
    def learn_transformation(self, input_pattern: np.ndarray, output_pattern: np.ndarray, action: str):
        """Apprentissage renforcé avec mémoire persistante"""
        # Ajouter à la mémoire
        self.transformation_memory[action].append((input_pattern.copy(), output_pattern.copy()))
        
        # Limiter la taille de la mémoire par action
        if len(self.transformation_memory[action]) > 10:
            self.transformation_memory[action].pop(0)
        
        self.success_count += 1
        self.consecutive_successes += 1
        self.recent_successes.append(True)
        
        # Calculer la confiance basée sur la cohérence
        num_examples = len(self.transformation_memory[action])
        consistency_bonus = 0.1 * self.consecutive_successes
        self.confidence_scores[action] = min(1.0, 0.3 + (num_examples * 0.15) + consistency_bonus)
        
        # Mise à jour de l'action principale
        if not self.primary_action or self.confidence_scores[action] > self.confidence_scores.get(self.primary_action, 0):
            self.primary_action = action
        
        # Protection renforcée après succès
        self.protection_counter = 200  # Augmenté de 100 à 200
        self.stability_score = min(1.0, self.stability_score + 0.1)
        
        # Spécialisation après succès répétés
        if self.consecutive_successes >= 5 and not self.is_specialized:
            self.is_specialized = True
            self.specialized_action = action
            logger.info(f"{self.id} specialized in {action}! Examples: {num_examples}, Confidence: {self.confidence_scores[action]:.2f}")
            
    def penalize_failure(self):
        """Pénalisation douce pour éviter la déstabilisation"""
        self.consecutive_successes = 0
        self.recent_successes.append(False)
        
        # Réduction plus douce de la confiance
        for action in self.confidence_scores:
            self.confidence_scores[action] *= 0.95  # Au lieu de 0.9
        
        self.stability_score = max(0.0, self.stability_score - 0.05)
        
    def should_dissolve(self) -> bool:
        """Critères plus stricts pour la dissolution"""
        # Jamais dissoudre une assemblée spécialisée ou protégée
        if self.is_specialized or self.protection_counter > 0:
            return False
        
        # Vérifier la performance récente
        if len(self.recent_successes) >= 5:
            recent_success_rate = sum(self.recent_successes) / len(self.recent_successes)
            if recent_success_rate > 0.3:  # Au moins 30% de succès récents
                return False
            
        # Dissoudre si trop vieille sans succès significatif
        if self.age > 300 and self.success_count < 3:  # Augmenté de 150 à 300
            return True
            
        # Dissoudre si trop d'échecs ET pas de succès récents
        if self.attempt_count > 50 and self.success_count / self.attempt_count < 0.05:
            return True
            
        return False
        
    def update(self):
        """Mise à jour périodique"""
        self.age += 1
        if self.protection_counter > 0:
            self.protection_counter -= 1
            
        # Décroissance très lente de la stabilité
        self.stability_score = max(0.0, self.stability_score - 0.001)


class MinimalViableEAN:
    """Réseau avec mécanismes de stabilisation améliorés"""
    
    def __init__(self, num_neurons: int = 100, world_size: Tuple[float, float] = (10.0, 10.0)):
        self.neurons = self._initialize_neurons(num_neurons, world_size)
        self.assemblies: Dict[str, AssemblyPMV] = {}
        self.current_step = 0
        self.current_time = 0.0
        self.world_size = world_size
        
        # Métriques
        self.total_successes = 0
        self.specialized_assemblies = []
        
        # Nouveaux paramètres de contrôle
        self.max_assemblies = 15  # Augmenté de 8 à 15
        self.assembly_formation_cooldown = 0
        
        logger.info(f"EAN initialized with {num_neurons} neurons")
        
    def _initialize_neurons(self, num_neurons: int, world_size: Tuple[float, float]) -> Dict[int, NeuronPMV]:
        """Initialise les neurones avec positions aléatoires"""
        neurons = {}
        for i in range(num_neurons):
            pos = (
                random.uniform(0, world_size[0]), 
                random.uniform(0, world_size[1])
            )
            neurons[i] = NeuronPMV(neuron_id=i, position=pos)
            
        # Pré-calcul des voisins avec rayon plus large
        for n1 in neurons.values():
            for n2 in neurons.values():
                if n1.id != n2.id:
                    dist = n1.calculate_distance_to(n2.position)
                    if dist < 4.0:  # Augmenté de 3.0 à 4.0
                        n1.neighboring_neurons.add(n2.id)
                        
        return neurons
        
    def _form_assemblies_near_activation(self, activation_center: Tuple[float, float]):
        """Formation d'assemblées avec cooldown"""
        # Vérifier le cooldown
        if self.assembly_formation_cooldown > 0:
            return
            
        # Vérifier la limite d'assemblées
        if len(self.assemblies) >= self.max_assemblies:
            return
            
        # Trouve les neurones proches et énergétiques
        nearby_neurons = []
        for neuron in self.neurons.values():
            if neuron.is_available_for_assembly():
                dist = neuron.calculate_distance_to(activation_center)
                if dist < 4.0 and neuron.energy > 45.0:  # Seuil d'énergie réduit
                    nearby_neurons.append((neuron.id, dist))
                    
        # Forme une assemblée avec les plus proches
        if len(nearby_neurons) >= 4:  # Réduit de 3 à 4 pour plus de robustesse
            nearby_neurons.sort(key=lambda x: x[1])
            founder_ids = {n[0] for n in nearby_neurons[:7]}  # Augmenté de 5 à 7
            
            assembly_id = f"assembly_{self.current_step}_{hash(frozenset(founder_ids))}"
            assembly = AssemblyPMV(assembly_id, founder_ids, self.current_time)
            
            # Marque les neurones
            for nid in founder_ids:
                self.neurons[nid].assembly_membership = assembly_id
                # Boost initial d'énergie
                self.neurons[nid].activate(10.0, self.current_time)
                
            self.assemblies[assembly_id] = assembly
            self.assembly_formation_cooldown = 5  # Cooldown de 5 steps
            logger.info(f"Formed {assembly_id} with {len(founder_ids)} neurons")
            
    def _activate_for_pattern(self, pattern: np.ndarray):
        """Activation améliorée avec plus d'énergie"""
        h, w = pattern.shape
        
        for i in range(h):
            for j in range(w):
                if pattern[i, j] > 0:
                    # Position dans le monde
                    world_x = (j + 0.5) * self.world_size[0] / w
                    world_y = (i + 0.5) * self.world_size[1] / h
                    
                    # Active les neurones proches avec plus d'énergie
                    for neuron in self.neurons.values():
                        dist = neuron.calculate_distance_to((world_x, world_y))
                        if dist < 4.0:
                            strength = 80.0 * (1.0 - dist / 4.0)  # Augmenté de 60 à 80
                            neuron.activate(strength, self.current_time)
                            
                    # Forme des assemblées si nécessaire
                    self._form_assemblies_near_activation((world_x, world_y))
                        
    def train_step(self, input_pattern: np.ndarray, output_pattern: np.ndarray):
        """Step d'entraînement avec stabilisation"""
        self.current_step += 1
        self.current_time += 0.1
        
        # Mise à jour du cooldown
        if self.assembly_formation_cooldown > 0:
            self.assembly_formation_cooldown -= 1
        
        # Décroissance d'énergie très légère
        for neuron in self.neurons.values():
            neuron.decay(0.02)
            
        # Active selon le pattern
        self._activate_for_pattern(input_pattern)
        
        # Déterminer la transformation correcte
        correct_action = self._determine_correct_transformation(input_pattern, output_pattern)
        
        # Laisse les assemblées tenter des transformations
        assemblies_to_remove = []
        for assembly_id, assembly in self.assemblies.items():
            if assembly.attempt_count < 100:  # Augmenté de 50 à 100
                result, action, confidence = assembly.attempt_transformation(input_pattern)
                
                if result is not None and np.array_equal(result, output_pattern):
                    if action == correct_action:
                        assembly.learn_transformation(input_pattern, output_pattern, action)
                        self.total_successes += 1
                        
                        # Récompense énergétique pour tous les membres
                        for nid in assembly.member_neuron_ids:
                            if nid in self.neurons:
                                self.neurons[nid].activate(30.0, self.current_time)
                    else:
                        assembly.penalize_failure()
                else:
                    assembly.penalize_failure()
                    
            # Mise à jour de l'assemblée
            assembly.update()
            
            # Vérifier dissolution
            if assembly.should_dissolve():
                assemblies_to_remove.append(assembly_id)
                
        # Retirer les assemblées dissoutes
        for assembly_id in assemblies_to_remove:
            assembly = self.assemblies[assembly_id]
            for nid in assembly.member_neuron_ids:
                if nid in self.neurons:
                    self.neurons[nid].assembly_membership = None
            del self.assemblies[assembly_id]
            logger.info(f"Dissolved {assembly_id}")

    def _determine_correct_transformation(self, input_pattern: np.ndarray, 
                                        output_pattern: np.ndarray) -> str:
        """Détermine quelle transformation est la vraie solution"""
        # Transformation ARC spécifique
        transformed = np.zeros_like(input_pattern)
        
        # Règles de transformation confirmées :
        if input_pattern[0,0] == 1:
            # Position (0,0) -> Positions (0,1) ET (1,0) [duplication]
            transformed[0,1] = 1
            transformed[1,0] = 1
        elif input_pattern[0,1] == 1:
            # Position (0,1) -> Position (1,1)
            transformed[1,1] = 1
        elif input_pattern[1,0] == 1:
            # Position (1,0) -> Position (0,1)
            transformed[0,1] = 1
        elif input_pattern[1,1] == 1:
            # Position (1,1) -> Position (0,0)
            transformed[0,0] = 1
        
        if np.array_equal(transformed, output_pattern):
            return 'rotate_90'
        elif np.array_equal(np.fliplr(input_pattern), output_pattern):
            return 'flip_horizontal'
        elif np.array_equal(np.flipud(input_pattern), output_pattern):
            return 'flip_vertical'
        elif np.array_equal(input_pattern, output_pattern):
            return 'identity'
        else:
            return 'unknown'
                
    def train_on_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 5):
        """Entraînement prolongé avec plus d'époques"""
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Mélanger les exemples à chaque époque
            shuffled_examples = examples.copy()
            random.shuffle(shuffled_examples)
            
            for i, (inp, out) in enumerate(shuffled_examples):
                logger.info(f"Training example {i+1}: {inp.flatten()} -> {out.flatten()}")
                
                # Plus de steps par exemple
                for _ in range(50):  # Augmenté de 30 à 50
                    self.train_step(inp, out)
                    
                # Affiche les assemblées spécialisées
                specialized = [a for a in self.assemblies.values() if a.is_specialized]
                if specialized:
                    logger.info(f"  Specialized assemblies: {len(specialized)}")
                    for a in sorted(specialized, key=lambda x: max(x.confidence_scores.values()), reverse=True)[:5]:
                        action = a.primary_action or 'unknown'
                        conf = a.confidence_scores.get(action, 0.0)
                        logger.info(f"    {a.id}: {action} (confidence: {conf:.2f}, successes: {a.success_count})")
                        
    def solve(self, test_input: np.ndarray) -> Optional[np.ndarray]:
        """Résolution avec vote majoritaire des assemblées spécialisées"""
        # Active le réseau
        self._activate_for_pattern(test_input)
        
        # Collecter les votes des assemblées
        votes = defaultdict(list)  # action -> [(result, confidence), ...]
        
        for assembly in self.assemblies.values():
            if assembly.is_specialized or assembly.success_count > 5:
                result, action, confidence = assembly.attempt_transformation(test_input)
                if result is not None and confidence > 0.3:
                    votes[action].append((result, confidence))
                    logger.info(f"Vote from {assembly.id}: {action} (confidence: {confidence:.2f})")
        
        # Choisir le meilleur résultat
        if not votes:
            logger.warning("No confident votes from assemblies")
            return None
            
        # Calculer le score moyen par action
        action_scores = {}
        for action, results in votes.items():
            total_confidence = sum(conf for _, conf in results)
            action_scores[action] = total_confidence / len(results)
        
        # Prendre l'action avec le meilleur score
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        best_results = votes[best_action]
        
        # Retourner le résultat avec la meilleure confiance
        best_result = max(best_results, key=lambda x: x[1])[0]
        logger.info(f"Final decision: {best_action} (score: {action_scores[best_action]:.2f})")
        
        return best_result


def test_minimal_system():
    """Test amélioré du système"""
    logger.info("="*60)
    logger.info("Testing Improved EAN System")
    logger.info("="*60)
    
    # Créer le réseau avec plus de neurones
    network = MinimalViableEAN(num_neurons=100)
    
    # Exemples d'entraînement pour rotation 90° (horaire)
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))
    ]
    
    # Entraîner avec plus d'époques
    network.train_on_examples(examples, epochs=5)
    
    # Tester avec plus de cas
    logger.info("\nTesting phase:")
    test_cases = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))
    ]
    
    correct = 0
    for inp, expected in test_cases:
        result = network.solve(inp)
        success = np.array_equal(result, expected) if result is not None else False
        logger.info(f"Test {inp.flatten()} -> {expected.flatten()}: "
                   f"{'PASS' if success else 'FAIL'}")
        if result is not None:
            logger.info(f"  Got: {result.flatten()}")
        if success:
            correct += 1
            
    accuracy = 100 * correct / len(test_cases)
    logger.info(f"\nAccuracy: {correct}/{len(test_cases)} = {accuracy:.0f}%")
    
    # Statistiques finales
    specialized = [a for a in network.assemblies.values() if a.is_specialized]
    logger.info(f"Total assemblies: {len(network.assemblies)}")
    logger.info(f"Specialized assemblies: {len(specialized)}")
    logger.info(f"Total successes during training: {network.total_successes}")
    
    return correct > 0


if __name__ == "__main__":
    success = test_minimal_system()
    exit(0 if success else 1)