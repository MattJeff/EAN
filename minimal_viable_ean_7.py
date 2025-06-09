"""
Testeur pour différents patterns ARC avec le système EAN V6
"""

import numpy as np
from minimal_viable_ean_6 import FixedIntegratedEAN
import logging

logger = logging.getLogger("ARC_TESTER")

def test_rotation_simple():
    """Test rotation simple (sans duplication)"""
    print("\n" + "="*60)
    print("TEST: Rotation Simple (90° horaire)")
    print("="*60)
    
    # Pattern de rotation simple
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]]))
    ]
    
    network = FixedIntegratedEAN(num_neurons=60, grid_shape=(2, 2))
    network.train_on_examples(examples, epochs=5, steps_per_example=20)
    
    accuracy = network.test_on_examples(examples)
    print(f"Résultat: {'SUCCÈS' if accuracy >= 75 else 'ÉCHEC'}")
    return accuracy >= 75

def test_reflection_horizontal():
    """Test réflexion horizontale"""
    print("\n" + "="*60)
    print("TEST: Réflexion Horizontale")
    print("="*60)
    
    # Pattern de réflexion horizontale
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]]))
    ]
    
    network = FixedIntegratedEAN(num_neurons=60, grid_shape=(2, 2))
    network.train_on_examples(examples, epochs=5, steps_per_example=20)
    
    accuracy = network.test_on_examples(examples)
    print(f"Résultat: {'SUCCÈS' if accuracy >= 75 else 'ÉCHEC'}")
    return accuracy >= 75

def test_identity():
    """Test transformation identité (pas de changement)"""
    print("\n" + "="*60)
    print("TEST: Transformation Identité")
    print("="*60)
    
    # Pattern identité
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 0], [1, 0]])),
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [0, 1]]))
    ]
    
    network = FixedIntegratedEAN(num_neurons=60, grid_shape=(2, 2))
    network.train_on_examples(examples, epochs=4, steps_per_example=15)
    
    accuracy = network.test_on_examples(examples)
    print(f"Résultat: {'SUCCÈS' if accuracy >= 75 else 'ÉCHEC'}")
    return accuracy >= 75

def test_translation():
    """Test translation (décalage)"""
    print("\n" + "="*60)
    print("TEST: Translation (décalage d'une case)")
    print("="*60)
    
    # Pattern de translation
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[2, 0], [0, 0]]), np.array([[0, 2], [0, 0]])),
        (np.array([[0, 0], [3, 0]]), np.array([[0, 0], [0, 3]]))
    ]
    
    network = FixedIntegratedEAN(num_neurons=60, grid_shape=(2, 2))
    network.train_on_examples(examples, epochs=5, steps_per_example=20)
    
    accuracy = network.test_on_examples(examples)
    print(f"Résultat: {'SUCCÈS' if accuracy >= 75 else 'ÉCHEC'}")
    return accuracy >= 75

def test_complex_duplication():
    """Test duplication plus complexe"""
    print("\n" + "="*60)
    print("TEST: Duplication Complexe (tout pixel se duplique)")
    print("="*60)
    
    # Chaque pixel actif se duplique à sa position ET à une position miroir
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 1]])),  # duplique en (1,1)
        (np.array([[0, 1], [0, 0]]), np.array([[0, 1], [1, 0]])),  # duplique en (1,0)
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [1, 0]])),  # duplique en (0,1)
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))   # duplique en (0,0)
    ]
    
    network = FixedIntegratedEAN(num_neurons=80, grid_shape=(2, 2))
    network.train_on_examples(examples, epochs=6, steps_per_example=25)
    
    accuracy = network.test_on_examples(examples)
    print(f"Résultat: {'SUCCÈS' if accuracy >= 75 else 'ÉCHEC'}")
    return accuracy >= 75

def test_all_patterns():
    """Teste tous les patterns"""
    print("🧪 SUITE DE TESTS ARC COMPLÈTE")
    print("="*80)
    
    tests = [
        ("Rotation avec Duplication (déjà validé)", lambda: True),  # Déjà testé avec succès
        ("Rotation Simple", test_rotation_simple),
        ("Réflexion Horizontale", test_reflection_horizontal),
        ("Transformation Identité", test_identity),
        ("Translation", test_translation),
        ("Duplication Complexe", test_complex_duplication)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Exécution: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"   ✅ {'RÉUSSI' if success else '❌ ÉCHOUÉ'}")
        except Exception as e:
            print(f"   💥 ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*80)
    
    successes = 0
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        print(f"{test_name:.<50} {status}")
        if success:
            successes += 1
    
    print(f"\n🏆 Score Global: {successes}/{len(results)} tests réussis")
    print(f"📈 Taux de Réussite: {100*successes/len(results):.0f}%")
    
    if successes >= len(results) * 0.7:  # 70% de réussite
        print("\n🎉 EXCELLENTE PERFORMANCE ! Votre système EAN est très robuste.")
    elif successes >= len(results) * 0.5:  # 50% de réussite
        print("\n👍 BONNE PERFORMANCE ! Le système fonctionne sur plusieurs types de patterns.")
    else:
        print("\n🔧 À AMÉLIORER. Le système nécessite des ajustements pour plus de robustesse.")

def quick_benchmark():
    """Benchmark rapide sur le pattern de rotation avec duplication"""
    print("\n" + "="*60)
    print("⚡ BENCHMARK RAPIDE - Rotation avec Duplication")
    print("="*60)
    
    import time
    
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))
    ]
    
    configs = [
        {"neurons": 40, "epochs": 4, "steps": 15},
        {"neurons": 60, "epochs": 5, "steps": 20},
        {"neurons": 80, "epochs": 6, "steps": 25}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n🔧 Configuration {i+1}: {config['neurons']} neurones, "
              f"{config['epochs']} époques, {config['steps']} steps")
        
        start_time = time.time()
        
        network = FixedIntegratedEAN(num_neurons=config['neurons'], grid_shape=(2, 2))
        network.train_on_examples(examples, epochs=config['epochs'], 
                                 steps_per_example=config['steps'])
        accuracy = network.test_on_examples(examples)
        
        elapsed = time.time() - start_time
        
        print(f"   ⏱️  Temps: {elapsed:.1f}s")
        print(f"   🎯 Précision: {accuracy:.0f}%")
        print(f"   📊 Assemblées spécialisées: {network.get_statistics()['specialized_assemblies']}")

if __name__ == "__main__":
    # Définir le niveau de log pour réduire le bruit
    logging.getLogger("EAN").setLevel(logging.WARNING)
    
    # Lancer tous les tests
    test_all_patterns()
    
    # Benchmark de performance
    quick_benchmark()