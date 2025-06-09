#!/usr/bin/env python3
"""
Analyse détaillée pour comprendre exactement la transformation ARC
"""

import numpy as np

def analyze_arc_transformation():
    # Positions dans une grille 2x2
    # [0,0] [0,1]
    # [1,0] [1,1]
    
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]])),  # Pos 0 -> Pos 1,2
        (np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),  # Pos 1 -> Pos 3
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 0]])),  # Pos 3 -> Pos 0
        (np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]))   # Pos 2 -> Pos 1
    ]
    
    print("Analyse de la transformation ARC")
    print("=" * 50)
    
    # Analyser chaque transformation
    for i, (inp, out) in enumerate(examples):
        print(f"\nExemple {i+1}:")
        print("Input:")
        print(inp)
        print("Output:")
        print(out)
        
        # Trouver où est le 1 dans l'input
        input_pos = None
        for r in range(2):
            for c in range(2):
                if inp[r,c] == 1:
                    input_pos = (r, c)
                    
        # Trouver où sont les 1 dans l'output
        output_positions = []
        for r in range(2):
            for c in range(2):
                if out[r,c] == 1:
                    output_positions.append((r, c))
                    
        print(f"Position du 1 dans input: {input_pos}")
        print(f"Positions des 1 dans output: {output_positions}")
        
    print("\n" + "=" * 50)
    print("Règle détectée:")
    print("- Position (0,0) -> Positions (0,1) ET (1,0) [duplication]")
    print("- Position (0,1) -> Position (1,1)")
    print("- Position (1,0) -> Position (0,1)")
    print("- Position (1,1) -> Position (1,0)")
    
    print("\nVérification de la règle:")
    
    def apply_rule(pattern):
        result = np.zeros_like(pattern)
        if pattern[0,0] == 1:
            result[0,1] = 1
            result[1,0] = 1
        elif pattern[0,1] == 1:
            result[1,1] = 1
        elif pattern[1,0] == 1:
            result[0,1] = 1
        elif pattern[1,1] == 1:
            result[1,0] = 1
        return result
    
    all_correct = True
    for i, (inp, expected) in enumerate(examples):
        result = apply_rule(inp)
        is_correct = np.array_equal(result, expected)
        print(f"Exemple {i+1}: {'✓' if is_correct else '✗'}")
        if not is_correct:
            print(f"  Attendu: {expected.flatten()}")
            print(f"  Obtenu:  {result.flatten()}")
            all_correct = False
            
    if all_correct:
        print("\nToutes les transformations sont correctes !")
    else:
        print("\nIl y a des erreurs dans la règle...")

if __name__ == "__main__":
    analyze_arc_transformation()