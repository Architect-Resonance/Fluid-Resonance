import numpy as np
from scipy import linalg

def build_8x8_grounded(alpha=0):
    # Clauses: (0,1,2),(1,2,3),(2,3,4),(3,4,0),(4,0,1),(5,0,3),(6,2,4),(7,5,6)
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    adj = np.zeros((8, 8), dtype=complex)
    for c in clauses:
        u, v, w = c
        for i, j in [(u,v), (v,w), (w,u)]:
            adj[i, j] += np.exp(1j * alpha)
            adj[j, i] += np.exp(-1j * alpha)
            
    deg = np.zeros(8)
    for c in clauses:
        u, v, w = c
        deg[u] += 2
        deg[v] += 2
        deg[w] += 2
        
    # Wanderer's Grounding: [2, 2, 0, 1, 0, 1, 0, 0]
    grounding = np.array([2, 2, 0, 1, 0, 1, 0, 0])
    
    L = np.diag(deg + grounding) - adj
    return L

def build_7x7_magnetic(alpha):
    # Only spoke nodes (0-6).
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    adj = np.zeros((7, 7), dtype=complex)
    deg = np.zeros(7)
    for c in clauses:
        u, v, w = c
        for i, j in [(u,v), (v,w), (w,u)]:
            if i < 7 and j < 7:
                adj[i, j] += np.exp(1j * alpha)
                adj[j, i] += np.exp(-1j * alpha)
                deg[i] += 1
                deg[j] += 1
    # Ungrounded: No grounding vector added.
    L = np.diag(deg) - adj
    return L

def duality_test():
    # 1. Non-magnetic Grounded 8x8 (The physical reference)
    L_ref = build_8x8_grounded(0)
    ref_eigs = linalg.eigvalsh(L_ref)
    print("Reference Grounded 8x8 λ_min:", ref_eigs[0])
    
    # 2. Sweep alpha on Ungrounded 7x7 Magnetic Spoke
    print("\nSweeping alpha on 7x7 Magnetic Spoke...")
    alphas = np.linspace(0, np.pi, 5000)
    best_a = 0
    min_err = 100
    for a in alphas:
        Lmag = build_7x7_magnetic(a)
        l = linalg.eigvalsh(Lmag)[0]
        if abs(l - ref_eigs[0]) < min_err:
            min_err = abs(l - ref_eigs[0])
            best_a = a
            
    print(f"Target λ_min: {ref_eigs[0]:.8f}")
    print(f"Equivalent alpha: {best_a/np.pi:.6f} pi")
    
    L_match = build_7x7_magnetic(best_a)
    match_eigs = linalg.eigvalsh(L_match)
    
    print("\nSpectrum Comparison (Top 7 modes):")
    for i in range(7):
        print(f"Mode {i}: Grounded={ref_eigs[i]:.6f}, Magnetic={match_eigs[i]:.6f}, Diff={abs(ref_eigs[i]-match_eigs[i]):.6f}")

    # Check for full isomorphism (is every eigenvalue matched?)
    # Grounded 8x8 has 8 eigenvalues. Magnetic 7x7 has 7.
    # So they CANNOT have the same full spectrum.
    # But maybe the first 7 eigenvalues match?
    
if __name__ == "__main__":
    duality_test()
