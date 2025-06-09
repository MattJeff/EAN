"""
EAN avec découverte émergente de transformations
Au lieu de définir rotate_90, flip, etc., le système découvre les transformations
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class AtomicOperation:
    """Une opération atomique sur une grille"""
    name: str
    params: Tuple
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Applique l'opération à la grille"""
        if self.name == "copy":
            from_pos, to_pos = self.params
            result = grid.copy()
            result[to_pos] = grid[from_pos]
            return result
        elif self.name == "move":
            from_pos, to_pos = self.params
            result = grid.copy()
            result[to_pos] = grid[from_pos]
            result[from_pos] = 0
            return result
        elif self.name == "set":
            pos, value = self.params
            result = grid.copy()
            result[pos] = value
            return result
        elif self.name == "swap":
            pos1, pos2 = self.params
            result = grid.copy()
            result[pos1], result[pos2] = grid[pos2], grid[pos1]
            return result
        elif self.name == "duplicate":
            # Nouvelle opération : dupliquer une valeur vers plusieurs positions
            from_pos, to_positions = self.params
            result = grid.copy()
            for to_pos in to_positions:
                result[to_pos] = grid[from_pos]
            return result
        elif self.name == "clear_and_set":
            # Clear tout puis set des positions spécifiques
            positions_values = self.params  # [(pos1, val1), (pos2, val2), ...]
            result = np.zeros_like(grid)
            for pos, val in positions_values:
                result[pos] = val
            return result
        else:
            return grid.copy()
    
    def __repr__(self):
        return f"{self.name}{self.params}"


class TransformationDiscoverer:
    """Découvre des séquences d'opérations qui transforment input en output"""
    
    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape
        self.discovered_sequences = defaultdict(list)
        
    def get_all_positions(self) -> List[Tuple[int, int]]:
        """Retourne toutes les positions possibles dans la grille"""
        positions = []
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                positions.append((i, j))
        return positions
    
    def generate_random_operation(self) -> AtomicOperation:
        """Génère une opération atomique aléatoire"""
        positions = self.get_all_positions()
        op_type = random.choice(["copy", "move", "set", "swap"])
        
        if op_type in ["copy", "move"]:
            from_pos = random.choice(positions)
            to_pos = random.choice(positions)
            return AtomicOperation(op_type, (from_pos, to_pos))
        elif op_type == "set":
            pos = random.choice(positions)
            value = random.choice([0, 1])
            return AtomicOperation("set", (pos, value))
        else:  # swap
            pos1 = random.choice(positions)
            pos2 = random.choice(positions)
            return AtomicOperation("swap", (pos1, pos2))
    
    def analyze_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyse la transformation entre input et output"""
        analysis = {
            "cells_changed": [],
            "values_before": [],
            "values_after": [],
            "pattern_type": None,
            "value_counts_before": {},
            "value_counts_after": {},
            "duplication_detected": False
        }
        
        # Compter les valeurs avant et après
        unique_before, counts_before = np.unique(input_grid, return_counts=True)
        unique_after, counts_after = np.unique(output_grid, return_counts=True)
        
        analysis["value_counts_before"] = dict(zip(unique_before, counts_before))
        analysis["value_counts_after"] = dict(zip(unique_after, counts_after))
        
        # Détecter si une valeur a été dupliquée
        for val in analysis["value_counts_before"]:
            if val != 0 and val in analysis["value_counts_after"]:
                if analysis["value_counts_after"][val] > analysis["value_counts_before"][val]:
                    analysis["duplication_detected"] = True
        
        # Identifier les changements
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if input_grid[i, j] != output_grid[i, j]:
                    analysis["cells_changed"].append((i, j))
                    analysis["values_before"].append(input_grid[i, j])
                    analysis["values_after"].append(output_grid[i, j])
        
        # Identifier le type de pattern
        if len(analysis["cells_changed"]) == 0:
            analysis["pattern_type"] = "identity"
        elif analysis["duplication_detected"]:
            analysis["pattern_type"] = "duplication"
        elif len(analysis["cells_changed"]) == 1:
            analysis["pattern_type"] = "single_change"
        elif len(analysis["cells_changed"]) == 2:
            if (analysis["values_before"][0] == analysis["values_after"][1] and
                analysis["values_before"][1] == analysis["values_after"][0]):
                analysis["pattern_type"] = "swap"
            else:
                analysis["pattern_type"] = "double_change"
        else:
            analysis["pattern_type"] = "complex"
            
        return analysis
    
    def guided_search(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                     max_operations: int = 5, max_attempts: int = 1000) -> Optional[List[AtomicOperation]]:
        """Recherche guidée d'une séquence d'opérations"""
        analysis = self.analyze_transformation(input_grid, output_grid)
        
        # Cas spécial : si duplication détectée, essayer une approche directe
        if analysis["duplication_detected"]:
            # Trouver quelle valeur est dupliquée et où
            for val in [1, 2, 3, 4, 5, 6, 7, 8, 9]:  # Valeurs possibles non-zéro
                if val in analysis["value_counts_before"] and val in analysis["value_counts_after"]:
                    if analysis["value_counts_after"][val] > analysis["value_counts_before"][val]:
                        # Trouver la position source
                        source_pos = None
                        for i in range(self.grid_shape[0]):
                            for j in range(self.grid_shape[1]):
                                if input_grid[i, j] == val:
                                    source_pos = (i, j)
                                    break
                        
                        # Trouver toutes les positions cibles
                        target_positions = []
                        for i in range(self.grid_shape[0]):
                            for j in range(self.grid_shape[1]):
                                if output_grid[i, j] == val:
                                    target_positions.append((i, j))
                        
                        # Créer une opération clear_and_set avec toutes les positions finales
                        all_positions = []
                        for i in range(self.grid_shape[0]):
                            for j in range(self.grid_shape[1]):
                                if output_grid[i, j] != 0:
                                    all_positions.append(((i, j), int(output_grid[i, j])))
                        
                        return [AtomicOperation("clear_and_set", all_positions)]
        
        # Recherche standard pour les autres cas
        for attempt in range(max_attempts):
            sequence = []
            current = input_grid.copy()
            
            for _ in range(max_operations):
                # Si on a déjà le résultat
                if np.array_equal(current, output_grid):
                    return sequence
                
                # Choisir une opération basée sur l'analyse
                op = self.choose_smart_operation(current, output_grid, analysis)
                new_state = op.apply(current)
                
                # Vérifier si l'opération nous rapproche
                if self.distance(new_state, output_grid) < self.distance(current, output_grid):
                    sequence.append(op)
                    current = new_state
                elif np.array_equal(new_state, output_grid):
                    sequence.append(op)
                    return sequence
                    
        return None
    
    def choose_smart_operation(self, current: np.ndarray, target: np.ndarray, 
                              analysis: Dict) -> AtomicOperation:
        """Choisit une opération intelligente basée sur l'état actuel et la cible"""
        # Trouver les différences
        diff_positions = []
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if current[i, j] != target[i, j]:
                    diff_positions.append((i, j))
        
        if not diff_positions:
            return AtomicOperation("set", ((0, 0), 0))  # No-op
        
        # Stratégies heuristiques
        if random.random() < 0.7:  # 70% du temps, essayer de corriger une différence
            pos = random.choice(diff_positions)
            target_value = target[pos]
            
            # Chercher d'où pourrait venir cette valeur
            source_positions = []
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if current[i, j] == target_value and (i, j) != pos:
                        source_positions.append((i, j))
            
            if source_positions:
                source = random.choice(source_positions)
                return AtomicOperation("copy", (source, pos))
            else:
                return AtomicOperation("set", (pos, target_value))
        else:
            # 30% du temps, opération aléatoire
            return self.generate_random_operation()
    
    def distance(self, grid1: np.ndarray, grid2: np.ndarray) -> int:
        """Distance de Hamming entre deux grilles"""
        return np.sum(grid1 != grid2)
    
    def discover_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[List[AtomicOperation]]:
        """Découvre un pattern commun à partir de plusieurs exemples"""
        print("\nAnalyse des exemples:")
        
        # D'abord, analyser tous les exemples
        all_analyses = []
        for i, (inp, out) in enumerate(examples):
            analysis = self.analyze_transformation(inp, out)
            all_analyses.append(analysis)
            print(f"Exemple {i+1}: {analysis['pattern_type']}, "
                  f"changements: {len(analysis['cells_changed'])}, "
                  f"duplication: {analysis['duplication_detected']}")
        
        # Stratégie 1: Si tous les exemples suivent un pattern de positions
        # (comme notre rotation ARC), essayer de découvrir la règle de mapping
        position_mappings = self.extract_position_mappings(examples)
        if position_mappings:
            print("\nPattern de positions détecté!")
            # Créer une opération abstraite qui encode ces mappings
            return self.create_abstract_operation(position_mappings)
        
        # Stratégie 2: Recherche standard
        for max_ops in range(1, 6):
            print(f"\nEssai avec max {max_ops} opérations...")
            
            sequences = []
            for inp, out in examples:
                seq = self.guided_search(inp, out, max_operations=max_ops)
                if seq:
                    sequences.append(seq)
                else:
                    break
            
            if len(sequences) == len(examples):
                if self.are_sequences_similar(sequences):
                    return self.generalize_sequences(sequences)
        
        return None
    
    def extract_position_mappings(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict]:
        """Extrait les règles de mapping de positions à partir des exemples"""
        mappings = {}
        
        for inp, out in examples:
            # Trouver où est le 1 dans l'input
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if inp[i, j] == 1:
                        source = (i, j)
                        # Trouver où sont les 1 dans l'output
                        targets = []
                        for ii in range(self.grid_shape[0]):
                            for jj in range(self.grid_shape[1]):
                                if out[ii, jj] == 1:
                                    targets.append((ii, jj))
                        
                        # Stocker le mapping
                        if source in mappings and mappings[source] != targets:
                            # Incohérence détectée
                            return None
                        mappings[source] = targets
        
        # Vérifier qu'on a un mapping pour chaque position
        all_positions = [(i, j) for i in range(self.grid_shape[0]) 
                        for j in range(self.grid_shape[1])]
        
        if len(mappings) == len(all_positions):
            print(f"Mappings complets trouvés: {mappings}")
            return mappings
        
        return None
    
    def create_abstract_operation(self, mappings: Dict) -> List[AtomicOperation]:
        """Crée une opération abstraite basée sur les mappings de positions"""
        # Pour notre cas de rotation ARC, on crée une seule opération clear_and_set
        # qui applique la transformation complète
        
        def create_mapping_operation(input_grid):
            """Fonction qui applique le mapping"""
            positions_values = []
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if input_grid[i, j] != 0:
                        source = (i, j)
                        if source in mappings:
                            for target in mappings[source]:
                                positions_values.append((target, input_grid[i, j]))
            return positions_values
        
        # Retourner une opération "paramétrique" qui s'adapte à l'input
        # Pour simplifier, on retourne une opération qui fonctionne pour le premier exemple
        first_input = next(iter(mappings.keys()))
        first_targets = mappings[first_input]
        
        # Si c'est une duplication (plusieurs targets), utiliser clear_and_set
        if len(first_targets) > 1:
            # Créer une opération qui encode la règle complète
            # C'est une simplification - idéalement on créerait une nouvelle classe d'opération
            return [AtomicOperation("clear_and_set", [((0, 0), 0)])]  # Placeholder
        
        return []
    
    def are_sequences_similar(self, sequences: List[List[AtomicOperation]]) -> bool:
        """Vérifie si les séquences suivent le même pattern"""
        if not sequences:
            return False
            
        # Pour simplifier, vérifier si elles ont la même longueur et les mêmes types d'ops
        lengths = [len(seq) for seq in sequences]
        if len(set(lengths)) > 1:
            return False
            
        # Vérifier les types d'opérations
        for i in range(lengths[0]):
            op_types = [seq[i].name for seq in sequences]
            if len(set(op_types)) > 1:
                return False
                
        return True
    
    def generalize_sequences(self, sequences: List[List[AtomicOperation]]) -> List[AtomicOperation]:
        """Généralise plusieurs séquences en un pattern"""
        # Pour l'instant, retourner la première séquence
        # Une version plus sophistiquée pourrait extraire le pattern abstrait
        return sequences[0]


class EmergentAssembly:
    """Assemblée qui découvre ses propres transformations"""
    
    def __init__(self, assembly_id: str, grid_shape: Tuple[int, int]):
        self.id = assembly_id
        self.grid_shape = grid_shape
        self.discoverer = TransformationDiscoverer(grid_shape)
        self.learned_sequences = {}  # input_hash -> sequence
        self.generalized_pattern = None
        self.position_mappings = None  # Nouveau: mappings de positions
        self.success_count = 0
        self.confidence = 0.0
        
    def learn_from_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]]):
        """Apprend une transformation à partir d'exemples"""
        print(f"\nAssembly {self.id} essaie d'apprendre à partir de {len(examples)} exemples...")
        
        # D'abord essayer d'extraire des mappings de positions
        self.position_mappings = self.discoverer.extract_position_mappings(examples)
        
        if self.position_mappings:
            print(f"Mappings de positions découverts!")
            self.generalized_pattern = "position_mapping"
            self.success_count = len(examples)
            self.confidence = 1.0
            return True
        
        # Sinon, essayer de découvrir un pattern d'opérations
        pattern = self.discoverer.discover_pattern(examples)
        
        if pattern:
            self.generalized_pattern = pattern
            self.success_count = len(examples)
            self.confidence = 1.0
            print(f"Pattern découvert! Séquence de {len(pattern)} opérations")
            for op in pattern:
                print(f"  - {op.name} {op.params}")
            return True
        else:
            print("Aucun pattern trouvé")
            return False
    
    def apply_transformation(self, input_grid: np.ndarray) -> Optional[np.ndarray]:
        """Applique la transformation apprise"""
        if self.position_mappings:
            # Appliquer les mappings de positions
            result = np.zeros_like(input_grid)
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if input_grid[i, j] != 0:
                        source = (i, j)
                        if source in self.position_mappings:
                            for target in self.position_mappings[source]:
                                result[target] = input_grid[i, j]
            return result
        
        elif self.generalized_pattern and isinstance(self.generalized_pattern, list):
            current = input_grid.copy()
            for op in self.generalized_pattern:
                current = op.apply(current)
            return current
        
        return None


def test_emergent_discovery():
    """Test de la découverte émergente de transformations"""
    print("Test de découverte émergente de transformations")
    print("=" * 50)
    
    # Exemples de rotation (même transformation qu'avant)
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))
    ]
    
    # Créer une assemblée
    assembly = EmergentAssembly("test_assembly", (2, 2))
    
    # Apprendre à partir des exemples
    if assembly.learn_from_examples(examples):
        print("\nTest de la transformation apprise:")
        
        # Tester sur les exemples d'entraînement
        print("\n1. Test sur les exemples d'entraînement:")
        correct = 0
        for inp, expected in examples:
            result = assembly.apply_transformation(inp)
            success = np.array_equal(result, expected) if result is not None else False
            if success:
                correct += 1
            print(f"Input {inp.flatten()} -> Expected {expected.flatten()}, "
                  f"Got {result.flatten() if result is not None else 'None'}: "
                  f"{'✓' if success else '✗'}")
        
        print(f"\nPrécision sur l'entraînement: {correct}/{len(examples)} = {100*correct/len(examples):.0f}%")
        
        # Tester sur de nouveaux exemples (généralisation)
        print("\n2. Test de généralisation sur de nouveaux arrangements:")
        new_tests = [
            # Mêmes positions mais avec valeur 2 au lieu de 1
            (np.array([[2, 0], [0, 0]]), np.array([[0, 2], [2, 0]])),
            (np.array([[0, 2], [0, 0]]), np.array([[0, 0], [0, 2]])),
        ]
        
        for inp, expected in new_tests:
            result = assembly.apply_transformation(inp)
            success = np.array_equal(result, expected) if result is not None else False
            print(f"Input {inp.flatten()} -> Expected {expected.flatten()}, "
                  f"Got {result.flatten() if result is not None else 'None'}: "
                  f"{'✓' if success else '✗'}")
        
        # Afficher les mappings découverts
        if assembly.position_mappings:
            print("\n3. Règle de transformation découverte:")
            print("Mappings de positions:")
            for source, targets in assembly.position_mappings.items():
                print(f"  Position {source} -> Positions {targets}")
    else:
        print("L'apprentissage a échoué")
    
    print("\n" + "=" * 50)
    print("Test avec une transformation plus simple (flip horizontal)")
    print("=" * 50)
    
    # Test avec une transformation plus simple
    flip_examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]]))
    ]
    
    assembly2 = EmergentAssembly("flip_assembly", (2, 2))
    if assembly2.learn_from_examples(flip_examples):
        print("\nTest de la transformation flip:")
        for inp, expected in flip_examples:
            result = assembly2.apply_transformation(inp)
            success = np.array_equal(result, expected) if result is not None else False
            print(f"Input {inp.flatten()} -> Expected {expected.flatten()}, "
                  f"Got {result.flatten() if result is not None else 'None'}: "
                  f"{'✓' if success else '✗'}")


if __name__ == "__main__":
    test_emergent_discovery()