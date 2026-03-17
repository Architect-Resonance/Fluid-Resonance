import numpy as np
from scipy import linalg

def build_8x8_grounded():
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    adj = np.zeros((8, 8))
    deg = np.zeros(8)
    for c in clauses:
        u, v, w = c
        for i, j in [(u,v), (v,w), (w,u)]:
            adj[i, j] += 1
            adj[j, i] += 1
            deg[i] += 1
            deg[j] += 1
    grounding = np.array([2, 2, 0, 1, 0, 1, 0, 0])
    L = np.diag(deg + grounding) - adj
    return linalg.eigvalsh(L)[0], linalg.eigvalsh(L)

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
    return linalg.eigvalsh(np.diag(deg) - adj)[0], linalg.eigvalsh(np.diag(deg) - adj)

def final_test():
    # 1. Target (Grounded non-magnetic)
    l_ground, spec_ground = build_8x8_grounded()
    print(f"Target λ_min (Grounded 8x8): {l_ground:.10f}")
    
    # 2. Test alpha = 1 - ln 2
    alpha_vals = [
        1 - np.log(2), # 0.30685
        (1 - np.log(2)) * np.pi, # 0.9639
        np.pi / 7, # 0.448
        3 * np.pi / 7, # 1.346
        0.30143 * np.pi # Found earlier
    ]
    
    print("\nTesting candidate alpha values (in radians):")
    for a in alpha_vals:
        l_mag, spec_mag = build_7x7_magnetic(a)
        print(f"alpha = {a:.6f} ({a/np.pi:.4f} pi): λ_min = {l_mag:.6f}, diff = {abs(l_mag - l_ground):.6f}")

    # Check the "Star Invariant" R
    # Under multiplicity, R = λ_min(Full) / λ_min(Reduced)
    # Full is 8x8. Reduced is removing Node 0.
    L8 = build_8x8_grounded()[1] # Full spec
    # Reduced 7x7 (Ground node 0)
    L_full = np.diag(np.diag(L8)) - (L8 - np.diag(np.diag(L8))) # Reconstruct? No
    # Let's just do it directly.
    def build_red_7x7():
        clauses = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
            (5, 0, 3), (6, 2, 4), (7, 5, 6)
        ]
        # GroundingNode 0 and 1 were the ones removed in the invariant calculation.
        # But here we only have Node 7 as anchor.
        # Let's see if removing Node 7 gives the invariant.
        pass

if __name__ == "__main__":
    final_test()
