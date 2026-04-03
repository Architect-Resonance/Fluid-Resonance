import sympy as sp
import numpy as np

def verify_hodge_identity(n):
    print(f"\n--- Symbolically Verifying Hodge Identity for K_{n} ---")
    
    # 1. Edges of Kn
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    ne = len(edges)
    nv = n
    
    # 2. Triangles of Kn
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triangles.append((i, j, k))
                
    nt = len(triangles)
    
    # 3. Boundary operators
    # d0: edges -> vertices (actually vertices -> edges in our Hodge L1 definition)
    # L1 = d0 @ d0.T + d1.T @ d1
    
    d0 = np.zeros((ne, nv))
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        d0[idx, i] = -1
        d0[idx, j] = 1
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
        
    d1 = np.zeros((nt, ne))
    for idx, (i, j, k) in enumerate(triangles):
        # [i,j,k] -> [j,k] - [i,k] + [i,j]
        e_jk = edge_index[(j, k)]
        e_ik = edge_index[(i, k)]
        e_ij = edge_index[(i, j)]
        
        # Orientations
        d1[idx, e_ij] = 1
        d1[idx, e_ik] = -1
        d1[idx, e_jk] = 1
        
    # 4. Compute L1
    L_down = d0 @ d0.T
    L_up = d1.T @ d1
    L1 = L_down + L_up
    
    # 5. Check Diagonal
    diag_vals = np.diag(L1)
    print(f"Diagonal Values (First 5): {diag_vals[:5]}... (Should be {n})")
    all_diag_correct = np.allclose(diag_vals, n)
    
    # 6. Check Off-Diagonal
    off_diag = L1 - np.diag(diag_vals)
    max_off = np.max(np.abs(off_diag))
    print(f"Max Off-Diagonal Entry: {max_off:.2e} (Should be 0)")
    all_off_correct = np.allclose(off_diag, 0)
    
    if all_diag_correct and all_off_correct:
        print(f"RESULT: VERIFIED. L1(K_{n}) = {n}*I")
    else:
        print(f"RESULT: FAILED verification.")

if __name__ == "__main__":
    for n in [4, 5, 6, 7]:
        verify_hodge_identity(n)
