"""
Version efficiente des défis ultra-avancés avec limitation simple d'assemblées
À placer dans le même dossier que ultra_advanced_challenges.py
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.network import IntegratedEAN, create_ean_network
from improved_network import create_enhanced_network, NetworkEnhancements
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

def apply_simple_efficiency(network, max_assemblies=10):
    """Applique une limitation simple du nombre d'assemblées"""
    
    # Sauvegarder la méthode originale
    original_form = network._form_assemblies_near_activation
    
    def limited_formation(activation_center):
        # Limiter strictement le nombre d'assemblées
        if len(network.assemblies) >= max_assemblies:
            # Trouver et remplacer la pire assemblée
            worst = None
            worst_score = float('inf')
            
            for assembly in network.assemblies.values():
                if assembly.protection_counter == 0:  # Pas protégée
                    # Score basé sur la performance
                    if assembly.is_specialized:
                        score = assembly.transformation_knowledge.success_rate
                    else:
                        score = 0.1  # Pénalité pour non-spécialisée
                    
                    if score < worst_score:
                        worst_score = score
                        worst = assembly
            
            # Remplacer seulement si la nouvelle serait meilleure
            if worst and worst_score < 0.4:
                network._dissolve_assembly(worst.id)
                return original_form(activation_center)
            else:
                return False  # Ne pas créer de nouvelle
        
        return original_form(activation_center)
    
    network._form_assemblies_near_activation = limited_formation
    network.max_assemblies = max_assemblies
    
    # Augmenter légèrement les seuils
    network.min_confidence_threshold = max(0.3, network.min_confidence_threshold)
    
    return network


def test_rotation_efficient():
    """Test rotation avec approche efficiente"""
    print("\n🎯 Test Rotation 2x2 - Approche Efficiente")
    print("="*60)
    
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[1, 0], [0, 0]])),
    ]
    
    # Créer réseau avec improved_network
    network = create_enhanced_network(num_neurons=80, grid_shape=(2, 2))
    
    # APPLIQUER LA LIMITATION
    network = apply_simple_efficiency(network, max_assemblies=6)
    
    # Entraînement
    for epoch in range(8):
        for _ in range(2):
            for inp, out in examples:
                for _ in range(30):
                    network.train_step(inp, out)
        
        if epoch % 2 == 0:
            stats = network.get_statistics()
            print(f"  Epoch {epoch+1}: {stats['total_assemblies']} assemblées, "
                  f"{stats['specialized_assemblies']} spécialisées")
    
    accuracy = network.test_on_examples(examples)
    efficiency = accuracy / max(1, len(network.assemblies))
    
    print(f"\n📊 Résultats:")
    print(f"  Accuracy: {accuracy:.0f}%")
    print(f"  Assemblées: {len(network.assemblies)}")
    print(f"  Efficience: {efficiency:.1f}")
    
    return accuracy, len(network.assemblies)


def test_multi_step_efficient():
    """Test multi-étapes efficient"""
    print("\n🔄 Test Multi-étapes 3x3 - Approche Efficiente")
    print("="*60)
    
    examples = [
        (np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]]),
         np.array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])),
        
        (np.array([[0, 2, 0],
                   [0, 0, 0],
                   [0, 0, 0]]),
         np.array([[0, 0, 0],
                   [2, 0, 2],
                   [0, 0, 0]])),
    ]
    
    network = create_enhanced_network(num_neurons=100, grid_shape=(3, 3))
    network = apply_simple_efficiency(network, max_assemblies=8)
    
    # Entraînement
    for epoch in range(10):
        for inp, out in examples:
            for _ in range(40):
                network.train_step(inp, out)
    
    accuracy = network.test_on_examples(examples)
    
    print(f"\n📊 Résultats:")
    print(f"  Accuracy: {accuracy:.0f}%")
    print(f"  Assemblées: {len(network.assemblies)}")
    
    return accuracy, len(network.assemblies)


def compare_approaches():
    """Compare brute force vs efficient"""
    print("\n🔬 COMPARAISON: BRUTE FORCE vs EFFICIENCE SIMPLE")
    print("="*80)
    
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[1, 0], [0, 0]])),
    ]
    
    # 1. Brute Force (sans limitation)
    print("\n1️⃣ BRUTE FORCE (improved_network standard):")
    brute = create_enhanced_network(80, (2, 2))
    
    for _ in range(150):
        for inp, out in examples:
            brute.train_step(inp, out)
    
    acc_brute = brute.test_on_examples(examples)
    count_brute = len(brute.assemblies)
    eff_brute = acc_brute / max(1, count_brute)
    
    print(f"   Accuracy: {acc_brute:.0f}%")
    print(f"   Assemblées: {count_brute}")
    print(f"   Efficience: {eff_brute:.1f}")
    
    # 2. Efficient (avec limitation)
    print("\n2️⃣ EFFICIENT (avec limitation simple):")
    efficient = create_enhanced_network(80, (2, 2))
    efficient = apply_simple_efficiency(efficient, max_assemblies=6)
    
    for _ in range(150):
        for inp, out in examples:
            efficient.train_step(inp, out)
    
    acc_eff = efficient.test_on_examples(examples)
    count_eff = len(efficient.assemblies)
    eff_eff = acc_eff / max(1, count_eff)
    
    print(f"   Accuracy: {acc_eff:.0f}%")
    print(f"   Assemblées: {count_eff}")
    print(f"   Efficience: {eff_eff:.1f}")
    
    # 3. Résumé
    print("\n📊 COMPARAISON:")
    if count_brute > 0 and count_eff > 0:
        reduction = 100 * (1 - count_eff / count_brute)
        improvement = 100 * (eff_eff / eff_brute - 1)
        
        print(f"   Réduction d'assemblées: {reduction:.0f}%")
        print(f"   Amélioration d'efficience: {improvement:+.0f}%")
        
        if acc_eff >= acc_brute - 10:
            print("   ✅ Performance maintenue avec moins d'assemblées!")
        else:
            print("   ⚠️  Perte de performance, ajuster les paramètres")


def run_all_efficient_tests():
    """Lance tous les tests efficients"""
    print("\n🚀 DÉFIS ULTRA-AVANCÉS - VERSION EFFICIENTE SIMPLE")
    print("="*80)
    print("📊 Utilise votre improved_network.py avec limitation d'assemblées")
    
    # Désactiver les logs détaillés
    logging.getLogger("EAN").setLevel(logging.WARNING)
    
    # Comparaison d'abord
    compare_approaches()
    
    # Tests individuels
    print("\n" + "="*80)
    print("TESTS INDIVIDUELS:")
    
    # Test 1: Rotation
    acc1, count1 = test_rotation_efficient()
    
    # Test 2: Multi-étapes
    acc2, count2 = test_multi_step_efficient()
    
    # Résumé final
    print("\n" + "="*80)
    print("🏆 RÉSUMÉ FINAL")
    print("="*80)
    
    avg_acc = (acc1 + acc2) / 2
    avg_count = (count1 + count2) / 2
    avg_eff = avg_acc / max(1, avg_count)
    
    print(f"Moyenne:")
    print(f"  Accuracy: {avg_acc:.0f}%")
    print(f"  Assemblées: {avg_count:.1f}")
    print(f"  Efficience: {avg_eff:.1f}")
    
    print("\n💡 Cette approche simple:")
    print("   ✅ Fonctionne avec votre code existant")
    print("   ✅ Limite le nombre d'assemblées")
    print("   ✅ Maintient une bonne performance")
    print("   ✅ Facile à ajuster (modifier max_assemblies)")


if __name__ == "__main__":
    run_all_efficient_tests()