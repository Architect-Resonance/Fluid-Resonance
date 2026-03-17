import numpy as np
from scipy import linalg

def build_8x8_correct(alpha=0):
    # Clauses: (0,1,2),(1,2,3),(2,3,4),(3,4,0),(4,0,1),(5,0,3),(6,2,4),(7,5,6)
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    
    adj = np.zeros((8, 8), dtype=complex)
    for c in clauses:
        u, v, w = c
        # Edges in triad: (u,v), (v,w), (w,u)
        for i, j in [(u,v), (v,w), (w,u)]:
            adj[i, j] += np.exp(1j * alpha)
            adj[j, i] += np.exp(-1j * alpha)
            
    # Degrees (diagonal)
    # The adjacency matrix currently has complex phases.
    # The degree of node i is the sum of weights of edges incident to it.
    # In non-magnetic case (alpha=0), this is just the row sum.
    # In magnetic case, degree is still the count of connections.
    deg_vals = np.zeros(8)
    for c in clauses:
        u, v, w = c
        deg_vals[u] += 2
        deg_vals[v] += 2
        deg_vals[w] += 2
        
    # Grounding d = [2, 2, 0, 1, 0, 1, 0, 0]
    grounding = np.array([2, 2, 0, 1, 0, 1, 0, 0])
    
    L = np.diag(deg_vals + grounding) - adj
    return L

def audit():
    # 1. Non-magnetic 8x8 check
    L0 = build_8x8_correct(0)
    eigs0 = linalg.eigvalsh(L0)
    print(f"Corrected Non-magnetic 8x8 λ_min: {eigs0[0]:.8f}")
    
    # Check 0.49499887 target
    target = 0.49499887
    print(f"Target λ_min: {target:.8f}")
    print(f"Difference: {eigs0[0] - target:.8e}")

    # 2. Magnetic sweep on 7x7 spoke (multiplicity included)
    # We remove Node 7 (the boundary anchor) and grounding d.
    # Spoke nodes are 0-6.
    def build_7x7_magnetic(alpha):
        cls7 = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
            (5, 0, 3), (6, 2, 4), (7, 5, 6) # Node 7 edges will be dropped
        ]
        
        adj = np.zeros((7, 7), dtype=complex)
        deg = np.zeros(7)
        for c in cls7:
            u, v, w = c
            edges = [(u,v), (v,w), (w,u)]
            for i, j in edges:
                if i < 7 and j < 7:
                    adj[i, j] += np.exp(1j * alpha)
                    adj[j, i] += np.exp(-1j * alpha)
                    deg[i] += 1
                    deg[j] += 1
        
        L = np.diag(deg) - adj
        return L

    print("\nSweeping alpha on 7x7 (ungrounded, correct multiplicity)...")
    alphas = np.linspace(0, np.pi, 20000)
    best_a = 0
    min_diff = 100
    for a in alphas:
        Lmag = build_7x7_magnetic(a)
        l_min = linalg.eigvalsh(Lmag)[0]
        if abs(l_min - target) < min_diff:
            min_diff = abs(l_min - target)
            best_a = a
            
    print(f"Alpha that matches target: {best_a:.8f} ({best_a/np.pi:.6f} pi)")
    print(f"λ_min at this alpha: {linalg.eigvalsh(build_7x7_magnetic(best_a))[0]:.8f}")

    # 3. Frustration Index Check
    # From Lange et al: Frustration index ι on a graph is related to the smallest eigenvalue.
    # For a magnetic Laplacian L_alpha, λ_1 (or λ_min for Dirichlet) is bounded by frustration.
    
if __name__ == "__main__":
    audit()
