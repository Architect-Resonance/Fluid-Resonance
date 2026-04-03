import numpy as np
from scipy.linalg import null_space

def compute_fano_homology():
    # 7 points
    V = 7
    # 7 lines (each line is a triplet of points)
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6),
        (2, 3, 6), (2, 4, 5)
    ]
    
    # Edges (1-cells): All pairs within each line
    edges = set()
    for l in lines:
        l = sorted(l)
        edges.add((l[0], l[1]))
        edges.add((l[1], l[2]))
        edges.add((l[0], l[2]))
    
    edges = sorted(list(edges))
    E = len(edges)
    
    print(f"Vertices (V): {V}")
    print(f"Edges (E):    {E}")
    print(f"Faces (F):    {len(lines)}")
    print(f"Euler Characteristic (V-E+F): {V - E + len(lines)}")
    
    # Boundary map d1: C1 -> C0
    # d1_ij = 1 for column k (edge e_ij) and row i, -1 for row j
    d1 = np.zeros((V, E))
    for k, (u, v) in enumerate(edges):
        d1[u, k] = -1
        d1[v, k] = 1
        
    # Boundary map d2: C2 -> C1
    # For a face (u, v, w) with edges (u,v), (v,w), (u,w)
    # d2(f) = e_uv + e_vw - e_uw
    d2 = np.zeros((E, len(lines)))
    for k, (u, v, w) in enumerate(lines):
        # We need to find the edge indices and assign directions
        # Standard orientation: u < v < w
        e1_idx = edges.index((u, v))
        e2_idx = edges.index((v, w))
        e3_idx = edges.index((u, w))
        
        d2[e1_idx, k] = 1
        d2[e2_idx, k] = 1
        d2[e3_idx, k] = -1
        
    # Betti numbers
    # b0 = dim(H0) = dim(ker d1) / dim(im d2) - wait
    # Actually b0 = V - rank(d1)
    b0 = V - np.linalg.matrix_rank(d1)
    # b1 = dim(ker d1) - rank(d2)
    b1 = (E - np.linalg.matrix_rank(d1)) - np.linalg.matrix_rank(d2)
    # b2 = dim(ker d2)
    b2 = len(lines) - np.linalg.matrix_rank(d2)
    
    print(f"\nBetti numbers:")
    print(f"b0: {b0}")
    print(f"b1: {b1}")
    print(f"b2: {b2}")
    
    print(f"\nVerification (b0 - b1 + b2): {b0 - b1 + b2}")

if __name__ == "__main__":
    compute_fano_homology()
