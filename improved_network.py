"""
Améliorations critiques pour le réseau EAN
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import random

logger = logging.getLogger("EAN.NetworkFix")

class NetworkEnhancements:
    """Améliorations pour la formation d'assemblées et découverte de patterns"""
    
    @staticmethod
    def enhanced_assembly_formation(network, activation_center: Tuple[float, float]):
        """Formation d'assemblées améliorée avec critères plus souples"""
        if network.assembly_formation_cooldown > 0:
            network.assembly_formation_cooldown -= 1
            return
        
        # Réduire les critères pour la formation
        MIN_NEURONS_FOR_ASSEMBLY = 3  # Réduit de 5
        MIN_ENERGY_THRESHOLD = 30.0   # Réduit de 45.0
        ACTIVATION_RADIUS = 6.0       # Augmenté de 4.0
        
        # Trouver les neurones disponibles
        nearby_neurons = []
        for neuron in network.neurons.values():
            if neuron.is_available_for_assembly() or neuron.energy > MIN_ENERGY_THRESHOLD:
                dist = neuron.calculate_distance_to(activation_center)
                if dist < ACTIVATION_RADIUS:
                    nearby_neurons.append((neuron.id, dist, neuron.energy))
        
        # Former une assemblée avec les meilleurs neurones
        if len(nearby_neurons) >= MIN_NEURONS_FOR_ASSEMBLY:
            # Trier par énergie décroissante, puis distance
            nearby_neurons.sort(key=lambda x: (-x[2], x[1]))
            
            # Prendre jusqu'à 10 neurones
            founder_ids = {n[0] for n in nearby_neurons[:min(10, len(nearby_neurons))]}
            
            # Créer l'assemblée
            assembly_id = f"assembly_{network.current_step}_{len(network.assemblies)}"
            from core.assembly import EmergentAssemblyEAN
            
            assembly = EmergentAssemblyEAN(
                assembly_id,
                frozenset(founder_ids),
                network.current_time,
                network.grid_shape
            )
            
            # Assigner les neurones
            for nid in founder_ids:
                if nid in network.neurons:
                    network.neurons[nid].join_assembly(assembly_id)
            
            network.assemblies[assembly_id] = assembly
            network.assembly_formation_cooldown = 1  # Réduit de 3
            
            logger.info(f"Formed {assembly_id} with {len(founder_ids)} neurons at step {network.current_step}")
            return True
        
        return False
    
    @staticmethod
    def enhanced_activation_pattern(network, pattern: np.ndarray):
        """Activation améliorée pour favoriser la formation d'assemblées"""
        h, w = pattern.shape
        activation_centers = []
        
        # Identifier tous les centres d'activation
        for i in range(h):
            for j in range(w):
                if pattern[i, j] > 0:
                    world_x = (j + 0.5) * network.world_size[0] / w
                    world_y = (i + 0.5) * network.world_size[1] / h
                    activation_centers.append((world_x, world_y, pattern[i, j]))
        
        # Activer les neurones avec plus d'intensité
        for world_x, world_y, value in activation_centers:
            for neuron in network.neurons.values():
                dist = neuron.calculate_distance_to((world_x, world_y))
                if dist < network.activation_radius * 1.5:  # Rayon augmenté
                    # Activation plus forte
                    strength = 100.0 * (1.0 - dist / (network.activation_radius * 1.5)) * value
                    neuron.activate(strength * 1.5, network.current_time)
        
        # Tenter de former des assemblées à chaque centre
        for world_x, world_y, _ in activation_centers:
            NetworkEnhancements.enhanced_assembly_formation(network, (world_x, world_y))
        
        # Si peu d'assemblées, forcer la création
        if len(network.assemblies) < 2 and network.current_step > 10:
            # Créer une assemblée avec des neurones aléatoires de haute énergie
            high_energy_neurons = [
                (n.id, n.energy) for n in network.neurons.values() 
                if n.energy > 40 and n.is_available_for_assembly()
            ]
            
            if len(high_energy_neurons) >= 3:
                high_energy_neurons.sort(key=lambda x: -x[1])
                founder_ids = {n[0] for n in high_energy_neurons[:6]}
                
                assembly_id = f"forced_assembly_{network.current_step}"
                from core.assembly import EmergentAssemblyEAN
                
                assembly = EmergentAssemblyEAN(
                    assembly_id,
                    frozenset(founder_ids),
                    network.current_time,
                    network.grid_shape
                )
                
                for nid in founder_ids:
                    network.neurons[nid].join_assembly(assembly_id)
                
                network.assemblies[assembly_id] = assembly
                logger.info(f"Force-created {assembly_id}")
    
    @staticmethod
    def enhanced_discovery_attempt(assembly):
        """Tentative de découverte améliorée avec seuils plus bas"""
        # Réduire le nombre minimum d'exemples requis
        MIN_EXAMPLES_FOR_DISCOVERY = 2  # Réduit de 4 ou 6
        
        if len(assembly.transformation_examples) < MIN_EXAMPLES_FOR_DISCOVERY:
            return False
        
        # Utiliser tous les exemples disponibles
        all_examples = list(assembly.transformation_examples)
        
        # Tenter plusieurs stratégies de découverte
        operations = None
        
        # 1. Découverte standard
        operations = assembly.discoverer.discover_pattern(all_examples)
        
        # 2. Si échec, essayer avec des sous-ensembles
        if not operations and len(all_examples) > 2:
            for i in range(len(all_examples) - 1):
                subset = all_examples[i:i+2]
                operations = assembly.discoverer.discover_pattern(subset)
                if operations:
                    break
        
        # 3. Si toujours échec, essayer une découverte partielle
        if not operations:
            from core.discovery.position_mapping import ImprovedTransformationDiscoverer
            partial_discoverer = ImprovedTransformationDiscoverer(assembly.grid_shape)
            partial_patterns = partial_discoverer.discover_partial_patterns(all_examples)
            
            if partial_patterns:
                # Créer une opération à partir du premier pattern partiel
                from core.discovery.atomic_operations import AtomicOperation
                operations = [AtomicOperation("position_mapping", (partial_patterns[0],))]
        
        if operations:
            assembly.transformation_knowledge.operations = operations
            assembly.transformation_knowledge.discovery_method = "enhanced_discovery"
            assembly.transformation_knowledge.confidence = 0.7  # Confiance initiale plus basse
            assembly.is_specialized = True
            
            logger.info(f"{assembly.id} discovered pattern with enhanced method")
            return True
        
        return False
    
    @staticmethod
    def apply_enhancements_to_network(network):
        """Applique toutes les améliorations au réseau"""
        # Remplacer les méthodes du réseau
        original_activate = network._activate_for_pattern
        original_form = network._form_assemblies_near_activation
        
        def new_activate(pattern):
            NetworkEnhancements.enhanced_activation_pattern(network, pattern)
        
        def new_form(center):
            NetworkEnhancements.enhanced_assembly_formation(network, center)
        
        network._activate_for_pattern = new_activate
        network._form_assemblies_near_activation = new_form
        
        # Ajouter une méthode pour forcer la découverte
        def force_discovery():
            for assembly in network.assemblies.values():
                if not assembly.is_specialized:
                    NetworkEnhancements.enhanced_discovery_attempt(assembly)
        
        network.force_discovery = force_discovery
        
        # Paramètres plus permissifs
        network.min_confidence_threshold = 0.1  # Très bas
        network.energy_decay_rate = 0.01      # Plus lent
        network.max_assemblies = 20           # Plus d'assemblées
        
        logger.info("Network enhancements applied")


def create_enhanced_network(num_neurons: int, grid_shape: Tuple[int, int]):
    """Crée un réseau EAN avec toutes les améliorations"""
    from core.network import create_ean_network
    
    network = create_ean_network(num_neurons, grid_shape)
    NetworkEnhancements.apply_enhancements_to_network(network)
    
    return network


# Amélioration spécifique pour les patterns récursifs
class RecursivePatternEnhancement:
    """Améliorations spécifiques pour les patterns récursifs"""
    
    @staticmethod
    def create_recursive_mapping(examples):
        """Crée un mapping récursif simple basé sur les exemples"""
        mappings = {}
        
        for inp, out in examples:
            # Trouver les positions actives dans l'input
            input_positions = [(i, j) for i in range(inp.shape[0]) 
                             for j in range(inp.shape[1]) if inp[i, j] != 0]
            
            # Pour chaque position active, trouver où la valeur apparaît dans l'output
            for i, j in input_positions:
                value = inp[i, j]
                targets = [(ii, jj) for ii in range(out.shape[0]) 
                          for jj in range(out.shape[1]) if out[ii, jj] == value]
                
                if (i, j) not in mappings:
                    mappings[(i, j)] = targets
                else:
                    # Vérifier la cohérence
                    if set(mappings[(i, j)]) != set(targets):
                        # Essayer de trouver un pattern commun
                        mappings[(i, j)] = list(set(mappings[(i, j)]) | set(targets))
        
        return mappings if mappings else None


# Test des améliorations
if __name__ == "__main__":
    print("Testing enhanced network...")
    
    # Test simple
    network = create_enhanced_network(60, (2, 2))
    
    # Exemples de rotation
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
    ]
    
    # Entraîner avec paramètres ajustés
    for epoch in range(3):
        for inp, out in examples:
            for _ in range(10):
                network.train_step(inp, out)
        
        # Forcer la découverte
        network.force_discovery()
        
        stats = network.get_statistics()
        print(f"Epoch {epoch + 1}: {stats['total_assemblies']} assemblies, "
              f"{stats['specialized_assemblies']} specialized")
    
    # Tester
    accuracy = network.test_on_examples(examples)
    print(f"Accuracy: {accuracy}%")