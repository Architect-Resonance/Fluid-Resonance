import numpy as np
from scipy import linalg

def build_8x8(alpha=0):
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
        deg[u] += 2; deg[v] += 2; deg[w] += 2
    grounding = np.array([2, 2, 0, 1, 0, 1, 0, 0])
    L = np.diag(deg + grounding) - adj
    return L

def final_r_check():
    L_full = build_8x8(0)
    l_full = linalg.eigvalsh(L_full)[0]
    
    # Reduced system: removal of hub variables 0 and 1
    L_red = np.delete(L_full, [0, 1], axis=0)
    L_red = np.delete(L_red, [0, 1], axis=1)
    l_red = linalg.eigvalsh(L_red)[0]
    
    print(f"Corrected λ_min (Full): {l_full:.8f}")
    print(f"Corrected λ_min (Red): {l_red:.8f}")
    print(f"Ratio R = λ_red / λ_full: {l_red / l_full:.8f}")
    print(f"Inverse Ratio 1/R: {l_full / l_red:.8f}")

if __name__ == "__main__":
    final_r_check()
