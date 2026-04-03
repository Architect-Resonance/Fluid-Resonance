import numpy as np
from scipy.linalg import eigh

def get_boundary_operators(n_vertices, edges, faces=None):
    n_edges = len(edges)
    d0 = np.zeros((n_edges, n_vertices))
    for i, (u, v) in enumerate(edges):
        d0[i, u] = -1
        d0[i, v] = 1
        
    if faces is None or len(faces) == 0:
        return d0, None
        
    n_faces = len(faces)
    d1 = np.zeros((n_faces, n_edges))
    edge_map = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    
    for i, (u, v, w) in enumerate(faces):
        e1 = tuple(sorted((u, v)))
        e2 = tuple(sorted((v, w)))
        e3 = tuple(sorted((w, u)))
        
        d1[i, edge_map[e1]] = 1 if (u < v) else -1
        d1[i, edge_map[e2]] = 1 if (v < w) else -1
        d1[i, edge_map[e3]] = 1 if (w < u) else -1
        
    return d0, d1

def compute_hodge_laplacians(n_vertices, edges, faces=None):
    d0, d1 = get_boundary_operators(n_vertices, edges, faces)
    
    # L0 = d0.T * d0
    L0 = np.dot(d0.T, d0)
    evals0 = np.sort(eigh(L0, eigvals_only=True))
    gap0 = evals0[1] if len(evals0) > 1 else 0
    
    # L1 = d0 * d0.T + d1.T * d1
    L1_down = np.dot(d0, d0.T)
    if d1 is not None and d1.size > 0:
        L1_up = np.dot(d1.T, d1)
        L1 = L1_down + L1_up
    else:
        L1 = L1_down
    evals1 = np.sort(eigh(L1, eigvals_only=True))
    gap1 = evals1[0] if len(evals1) > 0 else 0
    
    # L2 = d1 * d1.T
    if d1 is not None and d1.size > 0:
        L2 = np.dot(d1, d1.T)
        evals2 = np.sort(eigh(L2, eigvals_only=True))
        gap2 = evals2[0] if len(evals2) > 0 else 0
    else:
        gap2 = 0
        
    return gap0, gap1, gap2

def create_complete_simplicial_complex(n):
    vertices = list(range(n))
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
    faces = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                faces.append((i, j, k))
    return n, edges, faces

def create_triangulated_cylinder(n_slices=10, n_per_slice=6):
    """
    Creates a triangulated cylinder (vortex tube proxy)
    """
    vertices = []
    for i in range(n_slices):
        for j in range(n_per_slice):
            vertices.append((i, j)) # (slice_idx, vertex_in_slice)
            
    n_v = len(vertices)
    edges = []
    faces = []
    
    def get_idx(s, v):
        return s * n_per_slice + (v % n_per_slice)
        
    for s in range(n_slices):
        for v in range(n_per_slice):
            # Horizontal edges (within slice)
            edges.append((get_idx(s, v), get_idx(s, v+1)))
            if s < n_slices - 1:
                # Vertical edges (between slices)
                edges.append((get_idx(s, v), get_idx(s+1, v)))
                # Diagonal edges (for triangulation)
                edges.append((get_idx(s, v), get_idx(s+1, v+1)))
                # Faces
                faces.append((get_idx(s, v), get_idx(s+1, v), get_idx(s+1, v+1)))
                faces.append((get_idx(s, v), get_idx(s, v+1), get_idx(s+1, v+1)))
                
    # Remove duplicate edges
    unique_edges = []
    seen = set()
    for u, v in edges:
        e = tuple(sorted((u, v)))
        if e not in seen:
            unique_edges.append(e)
            seen.add(e)
            
    return n_v, unique_edges, faces

def run_hodge_audit():
    print(f"--- HODGE LAPLACIAN DIMENSION AUDIT ---")
    print(f"{'Complex':<20} | {'L0 Gap':<10} | {'L1 Gap':<10} | {'L2 Gap':<10}")
    print("-" * 60)
    
    # 1. K3, K4, K5
    for n_comp in [3, 4, 5]:
        n, e, f = create_complete_simplicial_complex(n_comp)
        g0, g1, g2 = compute_hodge_laplacians(n, e, f)
        print(f"{f'K{n_comp} Complete':<20} | {g0:<10.4f} | {g1:<10.4f} | {g2:<10.4f}")
    
    # 2. Star Graph
    n = 5
    e = [(0,1), (0,2), (0,3), (0,4)]
    g0, g1, g2 = compute_hodge_laplacians(n, e, [])
    print(f"{'Star Graph':<20} | {g0:<10.4f} | {g1:<10.4f} | {g2:<10.4f}")
    
    # 3. Vortex Tube (Cylinder)
    n, e, f = create_triangulated_cylinder(n_slices=5, n_per_slice=4)
    g0, g1, g2 = compute_hodge_laplacians(n, e, f)
    print(f"{'Vortex Tube':<20} | {g0:<10.4f} | {g1:<10.4f} | {g2:<10.4f}")

if __name__ == "__main__":
    run_hodge_audit()
