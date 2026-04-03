import numpy as np
from scipy.linalg import svd, eigh
from itertools import combinations

R_INV = 1.857306874

def build_complex(clauses, n_vars):
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
    return B1, B2, edges

clauses = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
    (5, 0, 3), (6, 2, 4), (7, 5, 6), (8, 0, 1), (9, 1, 2), (10, 4, 12)
]

def audit_ghost_edge():
    print("--- TOPOLOGICAL AUDIT: SEEKING THE GHOST EDGE ---")
    B1, B2, edges = build_complex(clauses, 13)
    L1 = B1.T @ B1 + B2 @ B2.T
    
    # Div-free basis
    U, S, _ = svd(B1.T)
    rank = np.sum(S > 1e-8)
    df_basis = U[:, rank:]
    
    L1_stokes = df_basis.T @ L1 @ df_basis
    evals, evecs = eigh(L1_stokes)
    
    # Target the first non-zero mode (The Lock Frequency)
    lock_idx = np.sum(evals < 1e-8)
    lock_val = evals[lock_idx]
    lock_vec_df = evecs[:, lock_idx]
    lock_vec_edges = df_basis @ lock_vec_df
    
    print(f"Lock Frequency: {lock_val:.10f}")
    
    # Find edges with highest support in this mode
    support = np.abs(lock_vec_edges)
    top_indices = np.argsort(support)[::-1]
    
    print("Top Edge Support (The Skeleton of the Twist):")
    for i in range(10):
        e_idx = top_indices[i]
        print(f"  Edge {edges[e_idx]}: {support[e_idx]:.6f}")
        
    # Analyze B2 (Triangle influence) on this mode
    tri_support = np.abs(B2.T @ lock_vec_edges)
    top_tri = np.argsort(tri_support)[::-1]
    
    print("\nTop Triangle Support (The Weaver's Fingers):")
    for i in range(5):
        t_idx = top_tri[i]
        print(f"  Triangle {clauses[t_idx]}: {tri_support[t_idx]:.6f}")

if __name__ == "__main__":
    audit_ghost_edge()
