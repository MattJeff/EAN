"""
Système de découverte corrigé - Version 2
Gère correctement la duplication et les pixels de fond
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger("Discovery")

@dataclass
class AtomicOperation:
    """Une opération atomique sur une grille"""
    name: str
    params: Tuple
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Applique l'opération à la grille"""
        if self.name == "position_mapping":
            # Nouvelle opération qui applique un mapping de positions complet
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
            
        elif self.name == "clear_and_set":
            positions_values = self.params
            result = np.zeros_like(grid)
            for pos, val in positions_values:
                if 0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1]:
                    result[pos] = val
            return result
            
        else:
            # Autres opérations standards
            return grid.copy()
    
    def __repr__(self):
        if self.name == "position_mapping":
            return f"PositionMapping({len(self.params[0])} rules)"
        return f"{self.name}{self.params}"


class ImprovedTransformationDiscoverer:
    """Version améliorée qui gère mieux la duplication et les pixels de fond"""
    
    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape
        self.discovered_sequences = defaultdict(list)
        
    def extract_position_mappings(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict]:
        """Version corrigée qui se concentre sur les pixels actifs (non-zéro)"""
        logger.info(f"Tentative de découverte de mappings sur {len(examples)} exemples")
        
        # Étape 1: Collecter tous les mappings pour les pixels ACTIFS uniquement
        active_pixel_mappings = defaultdict(set)
        
        for example_idx, (inp, out) in enumerate(examples):
            logger.debug(f"Exemple {example_idx}: {inp.flatten()} -> {out.flatten()}")
            
            # Pour chaque pixel ACTIF dans l'input
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if inp[i, j] != 0:  # Seulement les pixels actifs
                        source = (i, j)
                        value = inp[i, j]
                        
                        # Trouver TOUS les pixels de même valeur dans l'output
                        targets = []
                        for ii in range(self.grid_shape[0]):
                            for jj in range(self.grid_shape[1]):
                                if out[ii, jj] == value:
                                    targets.append((ii, jj))
                        
                        # Ajouter au mapping global
                        if targets:
                            active_pixel_mappings[source].update(targets)
                            logger.debug(f"  Pixel actif {source} (valeur {value}) -> {targets}")
        
        logger.info(f"Mappings de pixels actifs collectés: {dict(active_pixel_mappings)}")
        
        # Étape 2: Vérifier la cohérence pour les pixels actifs
        final_mappings = {}
        
        for source, targets in active_pixel_mappings.items():
            target_list = list(targets)
            is_consistent = True
            
            logger.debug(f"Vérification de cohérence pour {source} -> {target_list}")
            
            # Vérifier que ce mapping est cohérent à travers tous les exemples
            for example_idx, (inp, out) in enumerate(examples):
                # Vérifier seulement si ce pixel est actif dans cet exemple
                if inp[source] != 0:
                    expected_value = inp[source]
                    
                    # Vérifier que toutes les positions cibles ont la bonne valeur
                    for target in target_list:
                        if out[target] != expected_value:
                            logger.debug(f"  Incohérence dans exemple {example_idx}: "
                                       f"out[{target}] = {out[target]}, attendu {expected_value}")
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
                        logger.debug(f"  Incohérence dans exemple {example_idx}: "
                                   f"targets attendus {set(target_list)}, trouvés {set(actual_targets)}")
                        is_consistent = False
                        break
            
            if is_consistent:
                final_mappings[source] = target_list
                logger.info(f"Mapping validé: {source} -> {target_list}")
            else:
                logger.debug(f"Mapping rejeté pour incohérence: {source} -> {target_list}")
        
        # Étape 3: Vérifier qu'on a au moins un mapping valide
        if len(final_mappings) > 0:
            logger.info(f"Position mappings découverts avec succès!")
            logger.info(f"Nombre de règles de mapping: {len(final_mappings)}")
            for source, targets in final_mappings.items():
                logger.info(f"  {source} -> {targets}")
            return final_mappings
        else:
            logger.info("Aucun mapping de positions cohérent trouvé")
            return None
    
    def discover_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[List[AtomicOperation]]:
        """Découvre un pattern - d'abord essayer les mappings de positions"""
        logger.info(f"Découverte de pattern sur {len(examples)} exemples")
        
        # Essayer d'abord les mappings de positions (plus efficace)
        position_mappings = self.extract_position_mappings(examples)
        
        if position_mappings:
            # Retourner une seule opération qui encode le mapping complet
            operation = AtomicOperation("position_mapping", (position_mappings,))
            logger.info(f"Pattern découvert: {operation}")
            return [operation]
        
        # Sinon, essayer d'autres méthodes...
        logger.info("Pas de mapping de positions trouvé, essayer d'autres méthodes...")
        
        # Ici on pourrait ajouter d'autres méthodes de découverte
        # Pour l'instant, on retourne None
        return None


class RobustEmergentAssembly:
    """Assemblée avec découverte plus robuste"""
    
    def __init__(self, assembly_id: str, grid_shape: Tuple[int, int]):
        self.id = assembly_id
        self.grid_shape = grid_shape
        self.discoverer = ImprovedTransformationDiscoverer(grid_shape)
        self.transformation_knowledge = None
        self.transformation_examples = []
        self.success_count = 0
        self.attempt_count = 0
        self.confidence = 0.0
        self.is_specialized = False
        
    def observe_transformation(self, input_pattern: np.ndarray, output_pattern: np.ndarray):
        """Observe et stocke un exemple"""
        self.transformation_examples.append((input_pattern.copy(), output_pattern.copy()))
        if len(self.transformation_examples) > 20:
            self.transformation_examples.pop(0)
    
    def attempt_discovery(self) -> bool:
        """Tente de découvrir une transformation"""
        if len(self.transformation_examples) < 4:
            return False
        
        # Utiliser seulement les exemples récents pour la découverte
        recent_examples = self.transformation_examples[-8:]
        
        pattern = self.discoverer.discover_pattern(recent_examples)
        if pattern:
            self.transformation_knowledge = pattern
            self.confidence = 0.9  # Haute confiance initiale
            self.is_specialized = True
            logger.info(f"{self.id} découvert: {pattern}")
            return True
        
        return False
    
    def apply_transformation(self, input_pattern: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Applique la transformation découverte"""
        self.attempt_count += 1
        
        if not self.transformation_knowledge:
            return None, 0.0
        
        # Appliquer la transformation
        result = input_pattern.copy()
        for operation in self.transformation_knowledge:
            result = operation.apply(result)
        
        return result, self.confidence
    
    def update_confidence(self, success: bool):
        """Met à jour la confiance basée sur le succès"""
        if success:
            self.success_count += 1
            self.confidence = min(1.0, self.confidence * 1.05)
        else:
            self.confidence *= 0.9
        
        # Calculer le taux de succès global
        if self.attempt_count > 0:
            success_rate = self.success_count / self.attempt_count
            # Ajuster la confiance basée sur le taux de succès
            self.confidence = 0.3 + 0.7 * success_rate


def test_improved_discovery_v2():
    """Test de la découverte améliorée - Version 2"""
    print("Test de la découverte améliorée avec duplication - Version 2")
    print("=" * 60)
    
    # Exemples de rotation ARC avec duplication
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))
    ]
    
    # Créer une assemblée
    assembly = RobustEmergentAssembly("test_assembly_v2", (2, 2))
    
    # Observer tous les exemples
    for inp, out in examples:
        assembly.observe_transformation(inp, out)
    
    # Tenter la découverte
    if assembly.attempt_discovery():
        print("\nTransformation découverte avec succès!")
        
        # Tester sur tous les exemples
        print("\nTest sur les exemples d'entraînement:")
        all_correct = True
        for i, (inp, expected) in enumerate(examples):
            result, confidence = assembly.apply_transformation(inp)
            is_correct = np.array_equal(result, expected) if result is not None else False
            assembly.update_confidence(is_correct)
            
            print(f"Exemple {i+1}:")
            print(f"  Input: {inp.flatten()}")
            print(f"  Expected: {expected.flatten()}")
            print(f"  Got: {result.flatten() if result is not None else 'None'}")
            print(f"  Correct: {'✓' if is_correct else '✗'}")
            print(f"  Confidence: {confidence:.2f}")
            
            all_correct = all_correct and is_correct
        
        print(f"\nConfiance finale: {assembly.confidence:.2f}")
        
        
        # Test de généralisation
        print("\n" + "=" * 60)
        print("Test de généralisation avec d'autres valeurs:")
        
        generalization_tests = [
            (np.array([[2, 0], [0, 0]]), np.array([[0, 2], [2, 0]])),
            (np.array([[0, 3], [0, 0]]), np.array([[0, 0], [0, 3]])),
            (np.array([[0, 0], [0, 4]]), np.array([[4, 0], [0, 0]])),
            (np.array([[0, 0], [5, 0]]), np.array([[0, 5], [0, 0]]))
        ]
        
        for i, (inp, expected) in enumerate(generalization_tests):
            result, confidence = assembly.apply_transformation(inp)
            is_correct = np.array_equal(result, expected) if result is not None else False
            print(f"Test {i+1}: valeur={inp.max()}, correct={'✓' if is_correct else '✗'}")
            if not is_correct and result is not None:
                print(f"  Attendu: {expected.flatten()}, Obtenu: {result.flatten()}")
    
    else:
        print("Échec de la découverte")
        return False
    
    return True


def debug_discovery_process():
    """Debug détaillé du processus de découverte"""
    print("\n" + "=" * 60)
    print("DEBUG: Processus de découverte détaillé")
    print("=" * 60)
    
    # Configuration de logging pour debug
    logging.getLogger("Discovery").setLevel(logging.DEBUG)
    
    # Un seul exemple pour déboguer
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]]))
    ]
    
    discoverer = ImprovedTransformationDiscoverer((2, 2))
    
    print("Exemples utilisés pour la découverte:")
    for i, (inp, out) in enumerate(examples):
        print(f"  Exemple {i+1}: {inp.flatten()} -> {out.flatten()}")
    
    result = discoverer.extract_position_mappings(examples)
    
    if result:
        print(f"\nRésultat de la découverte: {result}")
        
        # Tester l'application
        print("\nTest d'application:")
        operation = AtomicOperation("position_mapping", (result,))
        
        for i, (inp, expected) in enumerate(examples):
            applied = operation.apply(inp)
            is_correct = np.array_equal(applied, expected)
            print(f"  Exemple {i+1}: {inp.flatten()} -> {applied.flatten()} "
                  f"(attendu: {expected.flatten()}) {'✓' if is_correct else '✗'}")
    else:
        print("\nÉchec de la découverte")
    
    # Restaurer le niveau de logging
    logging.getLogger("Discovery").setLevel(logging.INFO)


if __name__ == "__main__":
    # Test basique
    success = test_improved_discovery_v2()
    
    # Debug si nécessaire
    if not success:
        debug_discovery_process()
    
    print(f"\nRésultat final: {'SUCCÈS' if success else 'ÉCHEC'}")