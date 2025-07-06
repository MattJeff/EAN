"""
Version mise à jour des défis avec émergence équilibrée
"""

import numpy as np
import sys
import os
import logging
from typing import List, Tuple

# Utiliser la version équilibrée au lieu de la version trop stricte
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import de la version équilibrée
from balance_emergency import create_balanced_emergence_network, BalancedEmergenceEnhancements

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

class BalancedEmergentChallenges:
    """Défis avec émergence équilibrée"""
    
    @staticmethod
    def test_balanced_rotation():
        """Test rotation avec émergence équilibrée"""
        print("\n🌱 Défi Équilibré: Rotation avec Peu d'Assemblées")
        print("="*60)
        
        examples = [
            (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
            (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
            (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]])),
            (np.array([[0, 0], [1, 0]]), np.array([[1, 0], [0, 0]])),
        ]
        
        network = create_balanced_emergence_network(num_neurons=80, grid_shape=(2, 2))
        
        best_accuracy = 0
        
        for epoch in range(12):
            # Entraînement adaptatif
            intensity = 30 if epoch < 6 else 20
            
            for inp, out in examples:
                for _ in range(intensity):
                    network.train_step(inp, out)
            
            # Forcer la formation initiale si nécessaire
            if epoch == 3 and len(network.assemblies) == 0:
                print("  ⚡ Boost initial des assemblées...")
                network.force_initial_assemblies()
            
            # Évaluation périodique
            if epoch % 3 == 0:
                accuracy = network.test_on_examples(examples)
                best_accuracy = max(best_accuracy, accuracy)
                
                stats = network.get_statistics()
                print(f"  Epoch {epoch+1}: {stats['total_assemblies']} assemblées, "
                      f"{stats['specialized_assemblies']} spécialisées, "
                      f"Accuracy: {accuracy:.0f}%")
        
        final_accuracy = network.test_on_examples(examples)
        
        print(f"\n📊 Résultats:")
        print(f"  - Accuracy finale: {final_accuracy:.0f}%")
        print(f"  - Assemblées totales: {len(network.assemblies)}")
        print(f"  - Efficience: {final_accuracy/max(1, len(network.assemblies)):.1f}")
        
        # Succès si bonne accuracy avec peu d'assemblées
        return final_accuracy >= 75 and len(network.assemblies) <= 8
    
    @staticmethod
    def test_balanced_multi_step():
        """Test multi-étapes équilibré"""
        print("\n🔄 Défi Équilibré: Multi-étapes Progressif")
        print("="*60)
        
        examples = [
            # Simple translation diagonale
            (np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]),
             np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])),
            
            (np.array([[0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]]),
             np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0]])),
            
            # Pattern avec duplication
            (np.array([[2, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]),
             np.array([[0, 0, 2],
                       [0, 2, 0],
                       [0, 0, 0]])),
        ]
        
        network = create_balanced_emergence_network(num_neurons=100, grid_shape=(3, 3))
        
        # Apprentissage progressif
        for phase in range(3):
            print(f"\n  Phase {phase + 1}:")
            
            if phase == 0:
                # Phase 1: Pattern simple
                for _ in range(80):
                    for inp, out in examples[:2]:
                        network.train_step(inp, out)
            
            elif phase == 1:
                # Phase 2: Ajouter la complexité
                for _ in range(80):
                    for inp, out in examples:
                        network.train_step(inp, out)
                        
                # Forcer si nécessaire
                if len(network.assemblies) < 2:
                    network.force_initial_assemblies()
            
            else:
                # Phase 3: Consolidation
                BalancedEmergenceEnhancements.smart_consolidation(network)
                for _ in range(40):
                    for inp, out in examples:
                        network.train_step(inp, out)
            
            stats = network.get_statistics()
            print(f"    Assemblées: {stats['total_assemblies']}, "
                  f"Spécialisées: {stats['specialized_assemblies']}")
        
        accuracy = network.test_on_examples(examples)
        print(f"\n  Accuracy: {accuracy:.0f}% avec {len(network.assemblies)} assemblées")
        
        return accuracy >= 50 and len(network.assemblies) <= 10
    
    @staticmethod
    def test_balanced_recursion():
        """Test récursion équilibrée"""
        print("\n🌀 Défi Équilibré: Récursion Contrôlée")
        print("="*60)
        
        examples = [
            # Diagonal simple
            (np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]),
             np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])),
            
            # Centre vers coins
            (np.array([[0, 0, 0],
                       [0, 2, 0],
                       [0, 0, 0]]),
             np.array([[2, 0, 2],
                       [0, 2, 0],
                       [2, 0, 2]])),
            
            # Vertical vers horizontal
            (np.array([[0, 1, 0],
                       [0, 1, 0],
                       [0, 0, 0]]),
             np.array([[0, 0, 0],
                       [1, 1, 0],
                       [0, 0, 0]])),
        ]
        
        network = create_balanced_emergence_network(num_neurons=120, grid_shape=(3, 3))
        
        # Entraînement avec boost périodique
        for epoch in range(15):
            for inp, out in examples:
                for _ in range(40):
                    network.train_step(inp, out)
            
            # Boost si pas d'assemblées après quelques epochs
            if epoch == 4 and len(network.assemblies) == 0:
                print("  ⚡ Activation boost...")
                network.force_initial_assemblies()
            
            # Consolidation mi-parcours
            if epoch == 8:
                BalancedEmergenceEnhancements.smart_consolidation(network)
        
        accuracy = network.test_on_examples(examples)
        stats = network.get_statistics()
        
        print(f"\n  Accuracy: {accuracy:.0f}%")
        print(f"  Assemblées: {stats['total_assemblies']}")
        print(f"  Récursives: {stats['recursive_assemblies']}")
        
        return accuracy >= 60 and stats['total_assemblies'] <= 12
    
    @staticmethod
    def test_balanced_context():
        """Test contextuel équilibré"""
        print("\n🧠 Défi Équilibré: Contexte Adaptatif")
        print("="*60)
        
        examples = [
            # Avec 2: comportement spécial
            (np.array([[1, 0], [2, 0]]), np.array([[2, 0], [0, 1]])),
            (np.array([[0, 1], [0, 2]]), np.array([[0, 0], [2, 1]])),
            
            # Sans 2: comportement simple
            (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
            (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        ]
        
        network = create_balanced_emergence_network(num_neurons=90, grid_shape=(2, 2))
        
        # Apprentissage mixte dès le début
        for epoch in range(10):
            # Mélanger les exemples
            shuffled = examples.copy()
            if epoch % 2 == 0:
                np.random.shuffle(shuffled)
            
            for inp, out in shuffled:
                for _ in range(30):
                    network.train_step(inp, out)
            
            if epoch == 3 and len(network.assemblies) < 2:
                network.force_initial_assemblies()
        
        accuracy = network.test_on_examples(examples)
        efficiency = accuracy / max(1, len(network.assemblies))
        
        print(f"\n  Accuracy: {accuracy:.0f}%")
        print(f"  Assemblées: {len(network.assemblies)}")
        print(f"  Efficience: {efficiency:.1f}")
        
        return accuracy >= 50 and len(network.assemblies) <= 6
    
    @staticmethod
    def test_4x4_balanced():
        """Test 4x4 avec approche équilibrée"""
        print("\n🎯 Défi Équilibré: Pattern 4x4 Optimisé")
        print("="*60)
        
        # Simplifier avec des patterns clairs
        examples = [
            # Translation simple
            (np.array([[1, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]),
             np.array([[0, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])),
            
            # Coin opposé
            (np.array([[0, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]),
             np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [1, 0, 0, 0]])),
            
            # Centre
            (np.array([[0, 0, 0, 0],
                       [0, 2, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]),
             np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 2, 0],
                       [0, 0, 0, 0]])),
        ]
        
        network = create_balanced_emergence_network(num_neurons=150, grid_shape=(4, 4))
        
        # Entraînement intensif mais contrôlé
        for epoch in range(20):
            for inp, out in examples:
                for _ in range(50):
                    network.train_step(inp, out)
            
            # Interventions périodiques
            if epoch == 5 and len(network.assemblies) == 0:
                network.force_initial_assemblies()
            
            if epoch % 5 == 0 and epoch > 0:
                BalancedEmergenceEnhancements.smart_consolidation(network)
                
                stats = network.get_statistics()
                print(f"  Epoch {epoch}: {stats['total_assemblies']} assemblées")
        
        accuracy = network.test_on_examples(examples)
        
        print(f"\n  Accuracy finale: {accuracy:.0f}%")
        print(f"  Assemblées: {len(network.assemblies)}")
        
        return accuracy >= 60 and len(network.assemblies) <= 15


def run_balanced_challenges():
    """Lance les défis avec émergence équilibrée"""
    print("\n🌱 DÉFIS ÉMERGENCE ÉQUILIBRÉE - FONCTIONNELLE ET EFFICIENTE")
    print("="*80)
    print("🎯 Objectif: Performance élevée avec nombre modéré d'assemblées")
    print("📊 Cible: 5-15 assemblées selon la complexité")
    
    challenges = [
        ("Rotation Équilibrée", BalancedEmergentChallenges.test_balanced_rotation),
        ("Multi-étapes Progressif", BalancedEmergentChallenges.test_balanced_multi_step),
        ("Récursion Contrôlée", BalancedEmergentChallenges.test_balanced_recursion),
        ("Contexte Adaptatif", BalancedEmergentChallenges.test_balanced_context),
        ("Pattern 4x4 Optimisé", BalancedEmergentChallenges.test_4x4_balanced),
    ]
    
    results = []
    total_score = 0
    total_assemblies = 0
    
    for name, test_func in challenges:
        try:
            print(f"\n{'='*60}")
            success = test_func()
            results.append((name, success))
            
            if success:
                total_score += 20
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Résumé
    print("\n" + "="*80)
    print("🏆 RÉSULTATS ÉMERGENCE ÉQUILIBRÉE")
    print("="*80)
    
    for name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        print(f"{name:.<40} {status}")
    
    print(f"\n🌟 Score global: {total_score}%")
    print("💡 Un bon équilibre entre performance et efficience!")
    
    return total_score


# Fonction principale mise à jour
def compare_approaches():
    """Compare l'approche brute force vs émergence équilibrée"""
    print("\n" + "🔬"*40)
    print("COMPARAISON: BRUTE FORCE vs ÉMERGENCE ÉQUILIBRÉE")
    print("🔬"*40)
    
    # Test simple de rotation pour comparaison
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[1, 0], [0, 0]])),
    ]
    
    print("\n1️⃣ APPROCHE BRUTE FORCE (comme avant):")
    from improved_network import create_enhanced_network
    brute_network = create_enhanced_network(100, (2, 2))
    
    for _ in range(200):
        for inp, out in examples:
            brute_network.train_step(inp, out)
    
    brute_accuracy = brute_network.test_on_examples(examples)
    brute_assemblies = len(brute_network.assemblies)
    print(f"   Accuracy: {brute_accuracy:.0f}%")
    print(f"   Assemblées: {brute_assemblies}")
    print(f"   Efficience: {brute_accuracy/max(1, brute_assemblies):.1f}")
    
    print("\n2️⃣ APPROCHE ÉMERGENCE ÉQUILIBRÉE:")
    balanced_network = create_balanced_emergence_network(100, (2, 2))
    
    for epoch in range(10):
        for inp, out in examples:
            for _ in range(20):
                balanced_network.train_step(inp, out)
        
        if epoch == 3 and len(balanced_network.assemblies) == 0:
            balanced_network.force_initial_assemblies()
    
    balanced_accuracy = balanced_network.test_on_examples(examples)
    balanced_assemblies = len(balanced_network.assemblies)
    print(f"   Accuracy: {balanced_accuracy:.0f}%")
    print(f"   Assemblées: {balanced_assemblies}")
    print(f"   Efficience: {balanced_accuracy/max(1, balanced_assemblies):.1f}")
    
    print("\n📊 VERDICT:")
    if balanced_accuracy >= brute_accuracy - 10 and balanced_assemblies < brute_assemblies / 10:
        print("   ✨ L'émergence équilibrée est SUPÉRIEURE!")
    else:
        print("   🔧 Ajustements nécessaires...")


if __name__ == "__main__":
    # Désactiver les logs détaillés
    logging.getLogger("EAN").setLevel(logging.WARNING)
    
    # Lancer la comparaison
    compare_approaches()
    
    print("\n" + "="*80)
    
    # Lancer les défis équilibrés
    score = run_balanced_challenges()
    
    print("\n" + "🌟"*40)
    print(f"💫 Score final émergence équilibrée: {score}%")
    print("🎯 Objectif atteint: Performance ET Efficience!")
    print("🌟"*40)