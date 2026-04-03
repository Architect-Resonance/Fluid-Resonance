import numpy as np
import sympy as sp

def check_fano_frustration():
    # Fano incidence matrix (7x7)
    # 7 points, 7 lines. A_ij = 1 if point i is in line j.
    # Lines:
    # 0: (1, 2, 3)
    # 1: (1, 4, 5)
    # 2: (1, 6, 7)
    # 3: (2, 4, 6)
    # 4: (2, 5, 7)
    # 5: (3, 4, 7)
    # 6: (3, 5, 6)
    
    A = np.zeros((7, 7))
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6),
        (2, 3, 6), (2, 4, 5)
    ]
    for j, l in enumerate(lines):
        for p in l:
            A[p, j] = 1
            
    print("Fano Incidence Matrix A:")
    print(A)
    
    rank = np.linalg.matrix_rank(A)
    det = np.linalg.det(A)
    
    print(f"\nRank: {rank}")
    print(f"Determinant: {det:.4f}")
    
    # Check if there's a null space (obstruction)
    # sum phi_i = -Omega_j
    # This is A.T @ phi = -Omega
    # If A is invertible, we can ALWAYS solve for phi.
    # Frustration = 0.
    
    M = sp.Matrix(A)
    print(f"\nMatrix rank (SymPy): {M.rank()}")
    
    # Wait, the Wanderer says "b1 = 1".
    # If the incidence graph (Levi) has b1=8, and the Points graph has b1=15.
    # Where does b1=1 come from?
    # Maybe he means the Triad Graph H^1.
    # If triads are edges...
    
if __name__ == "__main__":
    check_fano_frustration()
