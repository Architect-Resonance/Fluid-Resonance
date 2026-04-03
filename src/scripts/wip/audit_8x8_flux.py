import numpy as np
from scipy import linalg

def build_8x8_laplacian(alpha=0):
    # Clauses: (0,1,2),(1,2,3),(2,3,4),(3,4,0),(4,0,1),(5,0,3),(6,2,4),(7,5,6)
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    
    adj = np.zeros((8, 8), dtype=complex)
    edges = set()
    for c in clauses:
        u, v, w = c
        for i, j in [(u,v), (v,w), (w,u)]:
            u_min, v_max = (i, j) if i < j else (j, i)
            edges.add((u_min, v_max))
    
    # Apply uniform flux alpha on all edges (oriented u < v)
    for u, v in edges:
        adj[u, v] = np.exp(1j * alpha)
        adj[v, u] = np.exp(-1j * alpha)
    
    # Degrees (non-magnetic degree)
    deg_vals = np.zeros(8)
    for u, v in edges:
        deg_vals[u] += 1
        deg_vals[v] += 1
        
    # Grounding d = [2, 2, 0, 1, 0, 1, 0, 0]
    grounding = np.array([2, 2, 0, 1, 0, 1, 0, 0])
    
    D = np.diag(deg_vals + grounding)
    L = D - adj
    return L

def audit():
    # 1. Non-magnetic 8x8 check
    L0 = build_8x8_laplacian(0)
    eigs0 = linalg.eigvalsh(L0)
    print(f"Non-magnetic 8x8 λ_min: {eigs0[0]:.6f}")
    
    # 2. Magnetic sweep on "underlying 7-vertex Fano graph"
    # User says: "effective magnetic flux on the underlying 7-vertex Fano graph"
    # This implies we remove Node 7 (the anchor) and the grounding d.
    # Let's define the 7x7 ungrounded version.
    
    def build_7x7_magnetic(alpha):
        # 7-node spoke graph (0-6) from the first 7 clauses
        # (Though Node 7 is in the 8th clause, it's the anchor).
        # We take the 8x8, remove Node 7 and remove grounding.
        
        # Recalculate 7x7 purely from spoke interactions
        clauses = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
            (5, 0, 3), (6, 2, 4)
        ]
        # Wait, the 8th clause (7,5,6) connects 5 and 6 as well.
        # So we should probably keep all edges except those involving 7.
        all_clauses = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
            (5, 0, 3), (6, 2, 4), (7, 5, 6)
        ]
        adj7 = np.zeros((7, 7), dtype=complex)
        edges7 = set()
        for c in all_clauses:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = c[i], c[j]
                    if u < 7 and v < 7:
                        u_min, v_max = (u, v) if u < v else (v, u)
                        edges7.add((u_min, v_max))
        
        for u, v in edges7:
            adj7[u, v] = np.exp(1j * alpha)
            adj7[v, u] = np.exp(-1j * alpha)
            
        deg7 = np.zeros(7)
        for u, v in edges7:
            deg7[u] += 1
            deg7[v] += 1
        
        L7 = np.diag(deg7) - adj7
        return L7

    print("\nSweeping alpha on 7x7 (ungrounded) to match 0.4950...")
    alphas = np.linspace(0, np.pi, 200)
    best_alpha = 0
    min_diff = 100
    target = 0.4950
    
    for a in alphas:
        Lmag = build_7x7_magnetic(a)
        # For an ungrounded graph, λ_0 is always 0. λ_1 (spectral gap) is what matters?
        # No, a magnetic Laplacian on a cycle doesn't have a 0 eigenvalue unless flux is 2pi*k.
        eigs_mag = linalg.eigvalsh(Lmag)
        l_min_mag = eigs_mag[0]
        
        if abs(l_min_mag - target) < min_diff:
            min_diff = abs(l_min_mag - target)
            best_alpha = a
            
    print(f"Target λ_min: {target}")
    print(f"Alpha that matches: {best_alpha/np.pi:.6f} pi")
    print(f"Value at that alpha: {linalg.eigvalsh(build_7x7_magnetic(best_alpha))[0]:.6f}")

    # 3. Magnetic sweep on the 8x8 (grounded) to see if alpha=0 is special
    print("\nSweeping alpha on 8x8 (grounded)...")
    best_a_grounded = 0
    min_d_grounded = 100
    for a in alphas:
        L8 = build_8x8_laplacian(a)
        l8 = linalg.eigvalsh(L8)[0]
        if abs(l8 - target) < min_d_grounded and a > 0.01: # Avoid trivial 0
            min_d_grounded = abs(l8 - target)
            best_a_grounded = a
    
    if min_d_grounded < 0.01:
        print(f"Found non-zero alpha on 8x8 matching 0.4950: {best_a_grounded/np.pi:.6f} pi")
    else:
        print("No other alpha on 8x8 produces 0.4950 in this range.")

if __name__ == "__main__":
    audit()
