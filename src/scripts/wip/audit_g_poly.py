import numpy as np
import sympy as sp

def audit_g_polynomial():
    # User-provided Matrix L_eff (8x8)
    L = np.array([
        [ 7, -1, -1, -1, -1, -1, -1, -1], # Node 0
        [-1,  6, -1, -1, -1, -1, -1,  0], # Node 1
        [-1, -1,  6, -1, -1, -1, -1,  0], # Node 2
        [-1, -1, -1,  5, -1, -1,  0,  0], # Node 3
        [-1, -1, -1, -1,  6, -1, -1,  0], # Node 4
        [-1, -1, -1, -1, -1,  6, -1,  0], # Node 5
        [-1, -1, -1,  0, -1, -1,  5, -1], # Node 6
        [-1,  0,  0,  0,  0,  0, -1,  2]  # Node 7 (Anchor)
    ])
    
    # Ground at the anchor (Node 7)
    L_g = np.delete(L, 7, axis=0)
    L_g = np.delete(L_g, 7, axis=1)
    
    # Use SymPy to get the characteristic polynomial
    M = sp.Matrix(L_g)
    lam = sp.symbols('lambda')
    cp = M.charpoly(lam)
    
    eigs = np.linalg.eigvalsh(np.array(L_g, dtype=float))
    print(f"Smallest eigenvalue of 7x7: {eigs[0]:.10f}")
    
    # Schur complement of the Hub (Node 0) in the Grounded 7x7
    # Node 0 is a 1x1 block
    # Nodes 1-6 are the 6x6 block C
    A = sp.Matrix([L_g[0, 0]])
    B = sp.Matrix(L_g[0, 1:]).T # B is 1x6
    C = sp.Matrix(L_g[1:, 1:])  # C is 6x6
    
    # Schur = C - B^T * A^-1 * B  (if we reduce hub out)
    # Schur = A - B * C^-1 * B^T  (if we reduce spokes out)
    
    # Let's find the determinant of (L_g - lambda*I) via Schur
    I6 = sp.eye(6)
    C_lam = C - lam * I6
    A_lam = A - lam * sp.eye(1)
    
    # det(M - lam*I) = det(C_lam) * (A_lam - B * C_lam^-1 * B.T)
    # This might give the g-polynomial if we assume a specific state.
    
    det_M = A_lam[0,0] - (B * C_lam.inv() * B.T)[0,0]
    # det_M is a rational function. Its numerator is the char poly.
    
    print(f"\nRational function from Schur reduction of Spokes:")
    # print(sp.simplify(det_M))

    # The user's target g polynomial: 512000g^2 + 770840g - 45639 = 0
    # Let's check the ratio again
    g_target = 0.057045381792665
    l_min = eigs[0]
    
    # Maybe R = l_min / g is an integer?
    print(f"l_min / g: {l_min / g_target}") # 8.677

if __name__ == "__main__":
    audit_g_polynomial()
