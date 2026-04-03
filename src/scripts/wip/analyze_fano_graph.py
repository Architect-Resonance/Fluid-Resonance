import numpy as np
import scipy.linalg as la

def build_leff_k5_2a():
    # L_eff = L_cluster + diag(d)
    # Spoke vars: 0, 1, 2, 3, 4, 5, 6, 7 (Total 8)
    
    # K5 core: vertices 0, 1, 2, 3, 4 are fully connected
    # Anchors: 5, 6, 7
    # Connectivity from FORMAL_PROOFS.md:
    # L_eff (8x8) provided:
    # rows:
    # 0: [ 7 -1 -1 -1 -1 -1  0  0]
    # 1: [-1  6 -1 -1 -1  0  0  0]
    # 2: [-1 -1  6 -1 -1  0 -1  0]
    # 3: [-1 -1 -1  5 -1 -1  0  0]
    # 4: [-1 -1 -1 -1  6  0 -1  0]
    # 5: [-1  0  0 -1  0  4 -1 -1]
    # 6: [ 0  0 -1  0 -1 -1  4 -1]
    # 7: [ 0  0  0  0  0 -1 -1  2]

    L_eff = np.array([
        [ 7, -1, -1, -1, -1, -1,  0,  0],
        [-1,  6, -1, -1, -1,  0,  0,  0],
        [-1, -1,  6, -1, -1,  0, -1,  0],
        [-1, -1, -1,  5, -1, -1,  0,  0],
        [-1, -1, -1, -1,  6,  0, -1,  0],
        [-1,  0,  0, -1,  0,  4, -1, -1],
        [ 0,  0, -1,  0, -1, -1,  4, -1],
        [ 0,  0,  0,  0,  0, -1, -1,  2]
    ])
    
    return L_eff

def analyze_leff():
    L_eff = build_leff_k5_2a()
    eigs = np.sort(np.real(la.eigvals(L_eff)))
    
    print("Eigenvalues of L_eff (8x8):")
    for i, lam in enumerate(eigs):
        print(f"  lambda_{i} = {lam:.10f}")
        
    # Check if lambda_min = 0.494998...
    l_min = eigs[0]
    print(f"\nl_min = {l_min:.10f}")
    
    # Check degree sequence / structure
    deg = np.diag(L_eff)
    print(f"Degree sequence (diagonal): {deg}")
    
    # Relate to 13/7
    R_target = 1.85731
    print(f"Target R = {R_target}")
    
    # R is l_min(L_eff) / l_min(L_red)
    # Red system: valve removes 2 core positions. Let's say vars 0 and 1.
    L_red = L_eff[2:, 2:]
    eigs_red = np.sort(np.real(la.eigvals(L_red)))
    l_min_red = eigs_red[0]
    print(f"l_min (reduced 6x6) = {l_min_red:.10f}")
    print(f"Ratio R = {l_min_red / l_min:.10f}")

if __name__ == "__main__":
    analyze_leff()
