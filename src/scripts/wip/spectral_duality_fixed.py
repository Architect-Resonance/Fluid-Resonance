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
        
    grounding = np.array([2, 2, 0, 1, 0, 1, 0, 0])
    L = np.diag(deg + grounding) - adj
    return L

def build_7x7_magnetic(alpha):
    # Only spoke nodes (0-6).
    # We include all edges between nodes 0-6 from ALL clauses.
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
    L = np.diag(deg) - adj
    return L

def duality_check():
    # 1. Target Spectrum from Non-magnetic Grounded 8x8
    L0 = build_8x8_grounded(0)
    target_eigs = linalg.eigvalsh(L0)
    print("Grounded 8x8 Spectrum (alpha=0):")
    print(target_eigs)
    target_min = target_eigs[0]

    # 2. Sweep alpha on Ungrounded 7x7 Magnetic
    print("\nSweeping alpha on 7x7 Magnetic Spoke...")
    alphas = np.linspace(0, np.pi, 5000)
    best_a = 0
    min_diff = 100
    for a in alphas:
        Lmag = build_7x7_magnetic(a)
        l = linalg.eigvalsh(Lmag)[0]
        if abs(l - target_min) < min_diff:
            min_diff = abs(l - target_min)
            best_a = a
            
    print(f"Target λ_min: {target_min:.8f}")
    print(f"Equivalent alpha: {best_a/np.pi:.6f} pi")
    
    L_match = build_7x7_magnetic(best_a)
    match_eigs = linalg.eigvalsh(L_match)
    print("\nMagnetic 7x7 Spectrum at best alpha:")
    print(match_eigs)
    
    # Compare spectra
    print("\nSpectrum Comparison (Matched λ_min):")
    for i in range(7):
        print(f"Mode {i}: Grounded={target_eigs[i]:.4f}, Magnetic={match_eigs[i]:.4f}, Diff={abs(target_eigs[i]-match_eigs[i]):.4f}")

if __name__ == "__main__":
    duality_check()
