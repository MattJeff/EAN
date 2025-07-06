"""
Version équilibrée pour l'émergence : fonctionnelle mais sans brute force
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random

logger = logging.getLogger("EAN.BalancedEmergence")

class BalancedEmergenceEnhancements:
    """Version équilibrée des améliorations d'émergence"""
    
    @staticmethod
    def configure_for_balanced_emergence(network):
        """Configure le réseau avec des paramètres équilibrés"""
        
        # Paramètres équilibrés - permettre la formation mais avec qualité
        network.max_assemblies = 8  # Suffisant pour commencer
        network.max_recursive_assemblies = 3
        
        # Seuils progressifs
        network.min_confidence_threshold = 0.4  # Plus raisonnable
        network.energy_decay_rate = 0.02
        
        # Formation d'assemblées - plus permissive mais sélective
        network.min_neurons_for_assembly = 4  # Réduit de 8
        network.min_energy_threshold = 40.0  # Réduit de 60
        network.assembly_formation_cooldown_max = 5  # Réduit de 10
        
        # Rayon d'activation augmenté pour faciliter la formation
        network.activation_radius = 5.0  # Au lieu de 4.0
        
        logger.info("Network configured for balanced emergence")
    
    @staticmethod
    def progressive_assembly_formation(network, activation_center: Tuple[float, float]):
        """Formation progressive d'assemblées avec standards croissants"""
        
        # Standards qui augmentent avec le nombre d'assemblées
        current_count = len(network.assemblies)
        
        # Standards progressifs
        if current_count < 2:
            min_neurons = 3
            min_energy = 35.0
            quality_threshold = 0.3
        elif current_count < 4:
            min_neurons = 4
            min_energy = 45.0
            quality_threshold = 0.5
        else:
            min_neurons = 6
            min_energy = 55.0
            quality_threshold = 0.7
        
        # Vérifier le cooldown
        if network.assembly_formation_cooldown > 0:
            return False
        
        # Trouver les neurones candidats
        candidates = []
        for neuron in network.neurons.values():
            if neuron.is_available_for_assembly() and neuron.energy > min_energy:
                dist = neuron.calculate_distance_to(activation_center)
                if dist < network.activation_radius * 1.2:  # Rayon légèrement augmenté
                    quality = neuron.energy * (1 - dist/(network.activation_radius * 1.2))
                    candidates.append((neuron.id, quality, neuron.energy))
        
        # Former une assemblée si on a assez de candidats de qualité
        if len(candidates) >= min_neurons:
            # Filtrer par qualité
            candidates = [c for c in candidates if c[1] >= quality_threshold * 100]
            
            if len(candidates) >= min_neurons:
                # Prendre les meilleurs
                candidates.sort(key=lambda x: -x[1])
                founder_ids = {c[0] for c in candidates[:min(8, len(candidates))]}
                
                # Créer l'assemblée
                assembly_id = f"balanced_assembly_{network.current_step}_{current_count}"
                from core.assembly import EmergentAssemblyEAN
                
                assembly = EmergentAssemblyEAN(
                    assembly_id,
                    frozenset(founder_ids),
                    network.current_time,
                    network.grid_shape
                )
                
                # Protection basée sur la qualité
                avg_quality = np.mean([c[1] for c in candidates[:len(founder_ids)]])
                assembly.protection_counter = int(100 + avg_quality)
                
                # Assigner les neurones
                for nid in founder_ids:
                    network.neurons[nid].join_assembly(assembly_id)
                
                network.assemblies[assembly_id] = assembly
                network.assembly_formation_cooldown = 3
                
                logger.info(f"Formed {assembly_id} with {len(founder_ids)} neurons, "
                           f"avg quality: {avg_quality:.1f}")
                return True
        
        return False
    
    @staticmethod
    def adaptive_discovery(assembly):
        """Découverte adaptative avec seuils progressifs"""
        
        # Seuils adaptatifs basés sur l'âge de l'assemblée
        if assembly.age < 50:
            min_examples = 3
            min_confidence = 0.6
        elif assembly.age < 100:
            min_examples = 4
            min_confidence = 0.7
        else:
            min_examples = 6
            min_confidence = 0.8
        
        if len(assembly.transformation_examples) < min_examples:
            return False
        
        # Utiliser tous les exemples
        examples = list(assembly.transformation_examples)
        
        # Essayer plusieurs approches de découverte
        operations = None
        
        # 1. Découverte standard
        operations = assembly.discoverer.discover_pattern(examples)
        
        # 2. Si échec, essayer avec relaxation
        if not operations and len(examples) >= 2:
            # Essayer avec moins d'exemples
            for subset_size in range(len(examples), 1, -1):
                subset = examples[:subset_size]
                operations = assembly.discoverer.discover_pattern(subset)
                if operations:
                    min_confidence *= 0.8  # Réduire la confiance
                    break
        
        # 3. Validation
        if operations:
            # Validation simple
            correct = 0
            for inp, expected in examples:
                result = inp.copy()
                for op in operations:
                    result = op.apply(result)
                if np.array_equal(result, expected):
                    correct += 1
            
            success_rate = correct / len(examples)
            
            if success_rate >= min_confidence:
                assembly.transformation_knowledge.operations = operations
                assembly.transformation_knowledge.discovery_method = "adaptive_discovery"
                assembly.transformation_knowledge.confidence = success_rate
                assembly.is_specialized = True
                
                logger.info(f"{assembly.id} discovered pattern with {success_rate:.2f} success rate")
                return True
        
        return False
    
    @staticmethod
    def smart_consolidation(network):
        """Consolidation intelligente qui préserve la diversité"""
        
        if len(network.assemblies) < 4:
            return  # Pas assez d'assemblées pour consolider
        
        # Analyser les patterns
        pattern_groups = defaultdict(list)
        
        for assembly in network.assemblies.values():
            if assembly.is_specialized and assembly.transformation_knowledge.operations:
                # Signature simplifiée
                sig = BalancedEmergenceEnhancements._get_simple_signature(assembly)
                pattern_groups[sig].append(assembly)
        
        # Consolider seulement les groupes avec beaucoup de redondance
        for sig, assemblies in pattern_groups.items():
            if len(assemblies) > 2:  # Au moins 3 assemblées similaires
                # Garder les 2 meilleures
                assemblies.sort(key=lambda a: -a.transformation_knowledge.success_rate)
                to_keep = assemblies[:2]
                to_remove = assemblies[2:]
                
                # Transférer les neurones
                for assembly in to_remove:
                    # Distribuer les neurones aux assemblées gardées
                    neurons_list = list(assembly.member_neuron_ids)
                    for i, nid in enumerate(neurons_list):
                        if nid in network.neurons:
                            target_assembly = to_keep[i % len(to_keep)]
                            network.neurons[nid].leave_assembly()
                            network.neurons[nid].join_assembly(target_assembly.id)
                            target_assembly.add_neuron(nid)
                    
                    # Supprimer l'assemblée
                    del network.assemblies[assembly.id]
                
                logger.info(f"Consolidated {len(to_remove)} redundant assemblies")
    
    @staticmethod
    def _get_simple_signature(assembly) -> str:
        """Signature simplifiée pour grouper les patterns similaires"""
        if not assembly.transformation_knowledge.operations:
            return "none"
        
        # Juste le type d'opération principal
        main_op = assembly.transformation_knowledge.operations[0]
        return f"{main_op.name}_{assembly.grid_shape}"

    @staticmethod 
    def boosted_activation(pattern: np.ndarray):
            """Activation boostée pour faciliter la formation d'assemblées"""
            h, w = pattern.shape
            
            # D'abord l'activation normale
            original_activate(pattern)
            
            # Puis boost supplémentaire si peu d'assemblées
            if len(network.assemblies) < 3:
                boost_factor = 2.0 - 0.3 * len(network.assemblies)
                
                for i in range(h):
                    for j in range(w):
                        if pattern[i, j] > 0:
                            world_x = (j + 0.5) * network.world_size[0] / w
                            world_y = (i + 0.5) * network.world_size[1] / h
                            
                            # Boost additionnel
                            for neuron in network.neurons.values():
                                dist = neuron.calculate_distance_to((world_x, world_y))
                                if dist < network.activation_radius * 1.5:
                                    extra_strength = 30.0 * boost_factor * pattern[i, j]
                                    neuron.energy = min(100, neuron.energy + extra_strength * 0.5)
                            
                            # Forcer la formation près du centre d'activation
                            network._form_assemblies_near_activation((world_x, world_y))
        
            network._activate_for_pattern = boosted_activation
        
    
    @staticmethod
    def periodic_optimization(network):
        """Optimisation périodique du réseau"""
        
        # Récompenser les assemblées performantes
        for assembly in network.assemblies.values():
            if assembly.is_specialized:
                success_rate = assembly.transformation_knowledge.success_rate
                if success_rate > 0.7:
                    # Boost les neurones membres
                    for nid in assembly.member_neuron_ids:
                        if nid in network.neurons:
                            network.neurons[nid].energy = min(100, 
                                network.neurons[nid].energy + 10 * success_rate)
        
        # Pénaliser légèrement les neurones inactifs
        for neuron in network.neurons.values():
            if not neuron.assembly_membership:
                neuron.energy *= 0.95
    
    @staticmethod
    def apply_balanced_emergence(network):
        """Applique les améliorations équilibrées"""
        
        # Configuration
        BalancedEmergenceEnhancements.configure_for_balanced_emergence(network)
        
        # Remplacer les méthodes critiques
        original_activate = network._activate_for_pattern
        original_form = network._form_assemblies_near_activation
        
        def new_activate(pattern):
            BalancedEmergenceEnhancements.boost_activation(network, pattern)
        
        def new_form(center):
            return BalancedEmergenceEnhancements.progressive_assembly_formation(network, center)
        
        network._activate_for_pattern = new_activate
        network._form_assemblies_near_activation = new_form
        
        # Améliorer la découverte pour toutes les assemblées
        def enhance_assemblies():
            for assembly in network.assemblies.values():
                # Remplacer la méthode de découverte
                assembly.attempt_discovery = lambda: BalancedEmergenceEnhancements.adaptive_discovery(assembly)
        
        # Hook pour le train_step
        original_train = network.train_step
        
        def new_train_step(inp, out):
            original_train(inp, out)
            
            # Améliorer les assemblées existantes
            if network.current_step % 10 == 0:
                enhance_assemblies()
            
            # Consolidation périodique
            if network.current_step % 100 == 0 and network.current_step > 0:
                BalancedEmergenceEnhancements.smart_consolidation(network)
            
            # Optimisation périodique
            if network.current_step % 50 == 0:
                BalancedEmergenceEnhancements.periodic_optimization(network)
        
        network.train_step = new_train_step
        
        # Ajouter une méthode pour forcer la formation si nécessaire
        def force_initial_assemblies():
            """Force la création d'assemblées initiales si aucune n'existe"""
            if len(network.assemblies) == 0 and network.current_step > 50:
                # Créer artificiellement des centres d'activation
                for i in range(2):
                    x = random.uniform(0, network.world_size[0])
                    y = random.uniform(0, network.world_size[1])
                    
                    # Activer fortement autour de ce point
                    for neuron in network.neurons.values():
                        dist = neuron.calculate_distance_to((x, y))
                        if dist < network.activation_radius:
                            neuron.activate(80.0, network.current_time)
                    
                    # Forcer la formation
                    BalancedEmergenceEnhancements.progressive_assembly_formation(network, (x, y))
        
        network.force_initial_assemblies = force_initial_assemblies
        
        logger.info("Balanced emergence enhancements applied")

        # Améliorer la découverte pour être moins stricte
    
    @staticmethod
    def enhance_discovery():
        """Améliore la découverte pour toutes les assemblées"""
        for assembly in network.assemblies.values():
            if not assembly.is_specialized and len(assembly.transformation_examples) >= 2:
                # Essayer avec seulement 2 exemples au début
                examples = list(assembly.transformation_examples)[:4]
                operations = assembly.discoverer.discover_pattern(examples)
                
                if operations:
                    assembly.transformation_knowledge.operations = operations
                    assembly.transformation_knowledge.confidence = 0.6
                    assembly.transformation_knowledge.discovery_method = "balanced_discovery"
                    assembly.is_specialized = True
                    logger.info(f"{assembly.id} discovered pattern with {len(examples)} examples")
    
    # Hook dans train_step
    original_train = network.train_step
    
    @staticmethod
    def balanced_train_step(inp, out):
        original_train(inp, out)
        
        # Découverte périodique
        if network.current_step % 20 == 0:
            enhance_discovery()
        
        # Consolidation légère
        if network.current_step % 100 == 0 and len(network.assemblies) > 5:
            consolidate_similar_assemblies(network)
    
    network.train_step = balanced_train_step
    
    # Méthode pour forcer la création initiale
    @staticmethod
    def force_assembly_creation():
        """Force la création d'assemblées si aucune n'existe"""
        if len(network.assemblies) > 0:
            return
        
        logger.info("Forcing initial assembly creation...")
        
        # Créer 2-3 centres d'activation artificiels
        for i in range(2):
            x = random.uniform(2, network.world_size[0]-2)
            y = random.uniform(2, network.world_size[1]-2)
            
            # Booster fortement l'énergie autour
            for neuron in network.neurons.values():
                dist = neuron.calculate_distance_to((x, y))
                if dist < network.activation_radius:
                    neuron.energy = 80.0
            
            # Forcer la formation
            network.assembly_formation_cooldown = 0  # Reset cooldown
            improved_form_assemblies((x, y))
    
        network.force_assembly_creation = force_assembly_creation
    
        return network


    @staticmethod
    def consolidate_similar_assemblies(network):
        """Consolide les assemblées similaires"""
        if len(network.assemblies) <= 3:
            return
        
        # Grouper par performance
        high_perf = []
        low_perf = []
        
        for assembly in network.assemblies.values():
            if assembly.is_specialized:
                if assembly.transformation_knowledge.success_rate > 0.5:
                    high_perf.append(assembly)
                else:
                    low_perf.append(assembly)
        
        # Supprimer les assemblées peu performantes s'il y en a trop
        if len(network.assemblies) > 8 and len(low_perf) > 2:
            # Garder seulement les 2 meilleures des peu performantes
            low_perf.sort(key=lambda a: a.transformation_knowledge.success_rate)
            for assembly in low_perf[:-2]:
                network._dissolve_assembly(assembly.id)
                logger.info(f"Consolidated by removing low-performing {assembly.id}")


    class AdaptiveLearningRate:
        """Contrôleur de taux d'apprentissage adaptatif"""
        def __init__(self):
            self.rate = 1.0
            self.history = []
            
        def update(self, performance):
            self.history.append(performance)
            if len(self.history) > 3:
                recent = self.history[-3:]
                if max(recent) - min(recent) < 0.05:
                    self.rate *= 0.9
                elif recent[-1] > recent[0] + 0.1:
                    self.rate = min(1.2, self.rate * 1.05)
        
        def get_rate(self):
            return self.rate

def create_working_balanced_network(num_neurons: int = 100, grid_shape: Tuple[int, int] = (2, 2)):
    """Crée un réseau avec émergence équilibrée qui FONCTIONNE"""
    
    # Importer et créer le réseau de base
    from core.network import IntegratedEAN
    
    # Créer le réseau avec des paramètres de base
    network = IntegratedEAN(
        num_neurons=num_neurons,
        world_size=(10.0, 10.0),
        grid_shape=grid_shape
    )
    
    # IMPORTANT: Configurer les paramètres AVANT toute modification
    network.max_assemblies = 10  # Limite raisonnable
    network.min_confidence_threshold = 0.3  # Pas trop strict
    network.energy_decay_rate = 0.015  # Standard
    network.activation_radius = 5.0  # Plus large
    
    # Paramètres critiques pour la formation d'assemblées
    network.min_neurons_for_assembly = 3  # TRÈS IMPORTANT: était peut-être à 5 dans le code original
    network.min_energy_threshold = 30.0  # Bas pour permettre la formation
    network.assembly_formation_cooldown_max = 2  # Court pour permettre plus de formations
    
    logger.info(f"Network configured with balanced parameters")
    
    # Remplacer la méthode de formation d'assemblées
    original_form_method = network._form_assemblies_near_activation
    
    def improved_form_assemblies(activation_center: Tuple[float, float]):
        """Version améliorée qui forme vraiment des assemblées"""
        
        # Vérifier le cooldown
        if network.assembly_formation_cooldown > 0:
            network.assembly_formation_cooldown -= 1
            return
        
        # Si on a trop d'assemblées, nettoyer
        if len(network.assemblies) >= network.max_assemblies:
            # Supprimer la pire
            worst_assembly = None
            worst_score = float('inf')
            
            for assembly in network.assemblies.values():
                if assembly.protection_counter == 0:
                    score = assembly.transformation_knowledge.success_rate
                    if assembly.is_specialized:
                        score += 0.5
                    
                    if score < worst_score:
                        worst_score = score
                        worst_assembly = assembly
            
            if worst_assembly and worst_score < 0.3:
                network._dissolve_assembly(worst_assembly.id)
        
        # Chercher les neurones disponibles avec critères souples
        nearby_neurons = []
        for neuron in network.neurons.values():
            # Critères plus souples
            if neuron.energy > network.min_energy_threshold:
                dist = neuron.calculate_distance_to(activation_center)
                if dist < network.activation_radius:
                    # Ne pas exiger is_available_for_assembly au début
                    if neuron.assembly_membership is None or len(network.assemblies) < 3:
                        score = neuron.energy * (1 - dist/network.activation_radius)
                        nearby_neurons.append((neuron.id, score))
        
        # Former l'assemblée si on a assez de neurones
        min_required = network.min_neurons_for_assembly
        if len(network.assemblies) < 2:
            min_required = 3  # Très souple au début
        
        if len(nearby_neurons) >= min_required:
            # Prendre les meilleurs
            nearby_neurons.sort(key=lambda x: -x[1])
            founder_ids = {n[0] for n in nearby_neurons[:min(8, len(nearby_neurons))]}
            
            # Créer l'assemblée
            assembly_id = f"balanced_{network.current_step}_{len(network.assemblies)}"
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
                    # Libérer d'abord si déjà dans une assemblée
                    if network.neurons[nid].assembly_membership:
                        network.neurons[nid].leave_assembly()
                    network.neurons[nid].join_assembly(assembly_id)
            
            network.assemblies[assembly_id] = assembly
            network.assembly_formation_cooldown = 2
            
            logger.info(f"Formed assembly {assembly_id} with {len(founder_ids)} neurons")
            return True
        
        return False
    
    # Remplacer la méthode
    network._form_assemblies_near_activation = improved_form_assemblies
    
    # Améliorer aussi l'activation pour booster la formation
    original_activate = network._activate_for_pattern
    


# Test rapide
# Test rapide intégré
if __name__ == "__main__":
    print("🧪 TEST RAPIDE - ÉMERGENCE ÉQUILIBRÉE FONCTIONNELLE")
    print("="*60)
    
    # Créer le réseau
    network = create_working_balanced_network(80, (2, 2))
    
    # Exemples simples
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[1, 0], [0, 0]])),
    ]
    
    # Entraînement
    for epoch in range(10):
        for inp, out in examples:
            for _ in range(20):
                network.train_step(inp, out)
        
        # Forcer la création au besoin
        if epoch == 2 and len(network.assemblies) == 0:
            print("  ⚡ Forçage de la création d'assemblées...")
            network.force_assembly_creation()
        
        stats = network.get_statistics()
        if epoch % 3 == 0:
            print(f"  Epoch {epoch+1}: {stats['total_assemblies']} assemblées, "
                  f"{stats['specialized_assemblies']} spécialisées")
    
    # Test final
    accuracy = network.test_on_examples(examples)
    print(f"\n📊 RÉSULTATS:")
    print(f"  Accuracy: {accuracy:.0f}%")
    print(f"  Assemblées: {len(network.assemblies)}")
    if len(network.assemblies) > 0:
        print(f"  Efficience: {accuracy/len(network.assemblies):.1f}")
    
    print("\n✅ Le système fonctionne!" if accuracy > 0 else "❌ Ajustements nécessaires")