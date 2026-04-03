import numpy as np
from scipy.linalg import svd, eigh
from itertools import combinations

def build_L1(clauses, n_vars):
    edges = set()
    for c in clauses:
        for i, j in combinations(c, 2):
            edges.add((min(i, j), max(i, j)))
    edges = sorted(edges)
    e_idx = {e: idx for idx, e in enumerate(edges)}
    
    B1 = np.zeros((n_vars, len(edges)))
    for idx, (i, j) in enumerate(edges):
        B1[i, idx] = -1
        B1[j, idx] = 1
        
    triangles = [tuple(sorted(c)) for c in clauses]
    B2 = np.zeros((len(edges), len(triangles)))
    for t_idx, tri in enumerate(triangles):
        i, j, k = tri
        for e in [(i, j), (i, k), (j, k)]:
            if e in e_idx:
                B2[e_idx[e], t_idx] = 1 if e != (i, k) else -1
    return B1, B2, edges, e_idx, triangles

# Current State after Core Suture
all_clauses = [
    (10, 4, 12), (9, 1, 2), (8, 0, 1),
    (0, 1, 2), (1, 2, 3), (4, 0, 1), # Remaining Core
    (5, 0, 3), (6, 2, 4), (7, 5, 6)  # Spoke influence
]
# Triangles (2,3,4) and (3,4,0) are surgically removed

def interrogate_the_fluid():
    print("--- INTERROGATING THE 1.224 FLUID STATE ---")
    B1, B2, edges, e_idx, triangles = build_L1(all_clauses, 13)
    L1 = B1.T @ B1 + B2 @ B2.T
    
    # Stokes Projection
    U, S, _ = svd(B1.T)
    rank = np.sum(S > 1e-8)
    df_basis = U[:, rank:]
    
    L1_stokes = df_basis.T @ L1 @ df_basis
    evals, evecs = eigh(L1_stokes)
    
    # The 1.224 Mode
    curr_idx = np.sum(evals < 1e-8)
    curr_gap = evals[curr_idx]
    curr_vec = df_basis @ evecs[:, curr_idx]
    
    print(f"Current Resonance: {curr_gap:.6f}")
    
    # Where is the Tension (Edge Support)?
    support = np.abs(curr_vec)
    top_e = np.argsort(support)[::-1]
    
    print("\nResidual Tension (Edge Level):")
    for i in range(8):
        idx = top_e[i]
        print(f"  Edge {edges[idx]}: {support[idx]:.6f}")
        
    # Why is it 'Thick' (Triangle vorticity)?
    vorticity = np.abs(B2.T @ curr_vec)
    top_t = np.argsort(vorticity)[::-1]
    
    print("\nThe 'Thickness' anchors (Triangle Vorticity):")
    for i in range(5):
        idx = top_t[i]
        print(f"  Triangle {triangles[idx]}: {vorticity[idx]:.6f}")

if __name__ == "__main__":
    interrogate_the_fluid()
