import numpy as np
from scipy.optimize import minimize
from scipy import linalg

def get_edges_with_multiplicity():
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    edge_counts = {}
    for c in clauses:
        u, v, w = c
        for i, j in [(u,v), (v,w), (w,u)]:
            if i < 7 and j < 7:
                u_min, v_max = (i, j) if i < j else (j, i)
                edge_counts[(u_min, v_max)] = edge_counts.get((u_min, v_max), 0) + 1
    return edge_counts

def frustration_index(alpha, edge_counts):
    # E(theta) = sum w_uv * (1 - cos(alpha + theta_u - theta_v))
    # Note: we need to handle the direction of alpha.
    # We assume alpha is uniform on edges u < v.
    
    def objective(theta):
        E = 0
        for (u, v), w in edge_counts.items():
            # theta_u, theta_v
            diff = alpha + theta[u] - theta[v]
            E += w * (1 - np.cos(diff))
        return E

    res = minimize(objective, np.zeros(7))
    return res.fun

def duality_audit():
    edge_counts = get_edges_with_multiplicity()
    
    # 1. Physical Target from Grounded 8x8
    # (Using the clauses and multiplicities)
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
        return linalg.eigvalsh(L)[0]

    target_gap = build_8x8_grounded()
    print(f"Target Grounded Gap λ_min: {target_gap:.8f}")

    # 2. Magnetic λ_min vs α
    def magnetic_gap(alpha):
        adj = np.zeros((7, 7), dtype=complex)
        deg = np.zeros(7)
        for (u, v), w in edge_counts.items():
            adj[u, v] += w * np.exp(1j * alpha)
            adj[v, u] += w * np.exp(-1j * alpha)
            deg[u] += w
            deg[v] += w
        L = np.diag(deg) - adj
        return linalg.eigvalsh(L)[0]

    print("\nSweeping alpha for λ_min and Frustration Index...")
    alphas = np.linspace(0, np.pi/2, 10)
    for a in alphas:
        l_mag = magnetic_gap(a)
        iota = frustration_index(a, edge_counts)
        print(f"alpha={a/np.pi:.3f}pi: λ_min={l_mag:.4f}, iota={iota:.4f}, ratio={l_mag/iota if iota!=0 else 0:.4f}")

    # Find alpha for target gap
    print("\nSearching for match...")
    from scipy.optimize import brentq
    def find_alpha(a):
        return magnetic_gap(a) - target_gap
    
    try:
        match_a = brentq(find_alpha, 0, np.pi)
        print(f"Found Match! alpha = {match_a:.8f} ({match_a/np.pi:.6f} pi)")
    except Exception as e:
        print(f"No match found in [0, pi]: {e}")

if __name__ == "__main__":
    duality_audit()
