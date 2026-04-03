import numpy as np
from scipy import linalg

def build_8x8_star(alpha=0):
    # Triad Clauses: (0,1,2),(1,2,3),(2,3,4),(3,4,0),(4,0,1),(5,0,3),(6,2,4),(7,5,6)
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    
    adj = np.zeros((8, 8), dtype=complex)
    for c in clauses:
        u, v, w = c
        # Edges (u,v), (v,w), (w,u)
        for i, j in [(u,v), (v,w), (w,u)]:
            adj[i, j] += np.exp(1j * alpha)
            adj[j, i] += np.exp(-1j * alpha)
            
    deg = np.zeros(8)
    # The degree should be the count of incident edges (with multiplicity)
    for c in clauses:
        u, v, w = c
        deg[u] += 2
        deg[v] += 2
        deg[w] += 2
        
    grounding = np.array([2, 2, 0, 1, 0, 1, 0, 0])
    
    L = np.diag(deg + grounding) - adj
    return L

def build_7x7_spoke_magnetic(alpha):
    # Spoke version (0-6)
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    
    adj = np.zeros((7, 7), dtype=complex)
    deg = np.zeros(7)
    for c in clauses:
        # We skip Node 7 but count edges between non-7 nodes in Cl 8
        u, v, w = c
        for i, j in [(u,v), (v,w), (w,u)]:
            if i < 7 and j < 7:
                adj[i, j] += np.exp(1j * alpha)
                adj[j, i] += np.exp(-1j * alpha)
                deg[i] += 1
                deg[j] += 1
                
    L = np.diag(deg) - adj
    return L

def run_audit():
    # 1. Check Grounded λ_min (Corrected multiplicity)
    L_grounded_0 = build_8x8_star(0)
    l_min_grounded = linalg.eigvalsh(L_grounded_0)[0]
    print(f"Non-magnetic Grounded 8x8 λ_min: {l_min_grounded:.8f}")

    # 2. Check Reduced λ_min (Corrected multiplicity)
    # Reduced means Node 0 removed? Or Node 7?
    # In Strategy A/B, "Reduced" refers to removing the hub.
    # If Node 0 is the center of the Fano plane...
    # But wait, Node 7 is mentioned as the anchor.
    # Removed set = {0, 1}? Let's assume Node 0 is the hub variable.
    L_red_0 = np.delete(L_grounded_0, 0, axis=0)
    L_red_0 = np.delete(L_red_0, 0, axis=1)
    l_min_red = linalg.eigvalsh(L_red_0)[0]
    print(f"Reduced 7x7 (Node 0 out) λ_min: {l_min_red:.8f}")
    
    R = l_min_grounded / l_min_red
    print(f"Ratio R: {R:.6f}")

    # 3. Magnetic Equivalence
    # Find alpha on the ungrounded 7x7 spoke (Node 7 removed entirely)
    # that reproduces the grounded λ_min.
    print("\nSweeping alpha on 7x7 spoke (multiplicities)...")
    alphas = np.linspace(0, np.pi, 2000)
    target = l_min_grounded
    best_a = 0
    min_err = 100
    for a in alphas:
        Lmag = build_7x7_spoke_magnetic(a)
        l = linalg.eigvalsh(Lmag)[0]
        if abs(l - target) < min_err:
            min_err = abs(l - target)
            best_a = a
            
    print(f"Target: {target:.8f}")
    print(f"Equivalent alpha: {best_a/np.pi:.6f} pi")
    print(f"λ_min at alpha: {linalg.eigvalsh(build_7x7_spoke_magnetic(best_a))[0]:.8f}")

if __name__ == "__main__":
    run_audit()
