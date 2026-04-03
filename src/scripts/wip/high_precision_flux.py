import numpy as np
from scipy import linalg

def build_L_spoke(alpha):
    # Clauses for the 7-node spoke (0-6):
    # (0,1,2),(1,2,3),(2,3,4),(3,4,0),(4,0,1),(5,0,3),(6,2,4)
    # Plus edge (5,6) from the anchor triad (7,5,6) - if we count all non-7 edges.
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    edges = set()
    for c in clauses:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = c[i], c[j]
                if u < 7 and v < 7:
                    u_min, v_max = (u, v) if u < v else (v, u)
                    edges.add((u_min, v_max))
    
    adj = np.zeros((7, 7), dtype=complex)
    for u, v in edges:
        adj[u, v] = np.exp(1j * alpha)
        adj[v, u] = np.exp(-1j * alpha)
        
    deg = np.zeros(7)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        
    L = np.diag(deg) - adj
    return L

def search():
    target = 0.49499887 # The Star Invariant target from doc
    
    # Grid search
    alphas = np.linspace(0, np.pi, 10000)
    best_a = 0
    min_err = 100
    for a in alphas:
        l = linalg.eigvalsh(build_L_spoke(a))[0]
        if abs(l - target) < min_err:
            min_err = abs(l - target)
            best_a = a
            
    print(f"Target: {target}")
    print(f"Alpha: {best_a:.8f} ({best_a/np.pi:.6f} pi)")
    print(f"Error: {min_err:.8e}")
    
    # Check 5pi/14
    a_fano = 5 * np.pi / 14
    l_fano = linalg.eigvalsh(build_L_spoke(a_fano))[0]
    print(f"5pi/14 produces: {l_fano:.6f} (Error: {abs(l_fano-target):.4f})")
    
    # Check 3pi/7
    a_7 = 3 * np.pi / 7
    l_7 = linalg.eigvalsh(build_L_spoke(a_7))[0]
    print(f"3pi/7 produces: {l_7:.6f} (Error: {abs(l_7-target):.4f})")

    # Check pi/2 (One octant?)
    a_oct = np.pi / 2
    l_oct = linalg.eigvalsh(build_L_spoke(a_oct))[0]
    print(f"pi/2 produces: {l_oct:.6f}")

if __name__ == "__main__":
    search()
