import numpy as np
from scipy.linalg import eigh

def get_boundary_operators(n_vertices, edges, faces=None):
    n_edges = len(edges)
    d0 = np.zeros((n_edges, n_vertices))
    edge_map = {}
    for i, (u, v) in enumerate(edges):
        d0[i, u] = -1
        d0[i, v] = 1
        edge_map[tuple(sorted((u, v)))] = i
        
    if faces is None or len(faces) == 0:
        return d0, None
        
    n_faces = len(faces)
    d1 = np.zeros((n_faces, n_edges))
    
    for i, (u, v, w) in enumerate(faces):
        e1 = tuple(sorted((u, v)))
        e2 = tuple(sorted((v, w)))
        e3 = tuple(sorted((w, u)))
        
        # Orientations
        d1[i, edge_map[e1]] = 1 if (u < v) else -1
        d1[i, edge_map[e2]] = 1 if (v < w) else -1
        d1[i, edge_map[e3]] = 1 if (w < u) else -1
        
    return d0, d1

def compute_gaps(n_v, edges, faces):
    d0, d1 = get_boundary_operators(n_v, edges, faces)
    L0 = np.dot(d0.T, d0)
    e0 = np.sort(eigh(L0, eigvals_only=True))
    g0 = e0[1] if len(e0) > 1 else 0
    
    L1_down = np.dot(d0, d0.T)
    if d1 is not None:
        L1_up = np.dot(d1.T, d1)
        L1 = L1_down + L1_up
    else:
        L1 = L1_down
    e1 = np.sort(eigh(L1, eigvals_only=True))
    g1 = e1[0] if len(e1) > 0 else 0
    
    g2 = 0
    if d1 is not None:
        L2 = np.dot(d1, d1.T)
        e2 = np.sort(eigh(L2, eigvals_only=True))
        g2 = e2[0] if len(e2) > 0 else 0
        
    return g0, g1, g2

def run_reconnection_audit():
    print(f"--- VORTEX RECONNECTION: HODGE DIMENSION AUDIT ---")
    print(f"{'State':<20} | {'L0 Gap':<10} | {'L1 Gap':<10} | {'L2 Gap':<10}")
    print("-" * 65)
    
    # 1. Disjoint Loops (Two 3-cycles)
    n_v = 6
    edges = [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)]
    g0, g1, g2 = compute_gaps(n_v, edges, [])
    print(f"{'Disjoint Loops':<20} | {g0:<10.4f} | {g1:<10.4f} | {g2:<10.4f}")
    
    # 2. Contact (Bridged)
    # Add bridge (1,4)
    edges_v2 = edges + [(1,4)]
    g0, g1, g2 = compute_gaps(n_v, edges_v2, [])
    print(f"{'Contact (Bridge)':<20} | {g0:<10.4f} | {g1:<10.4f} | {g2:<10.4f}")
    
    # 3. Suture (Bridging Faces)
    # Add faces (0,1,4), (1,4,5) etc if edges exist
    edges_v3 = edges + [(1,4), (0,4), (1,5)]
    faces_v3 = [(0,1,4), (1,4,5)]
    g0, g1, g2 = compute_gaps(n_v, edges_v3, faces_v3)
    print(f"{'Suture (Faces)':<20} | {g0:<10.4f} | {g1:<10.4f} | {g2:<10.4f}")
    
    # 4. Reconnected (Merged Cycle)
    # 0 -> 2 -> 1 -> 4 -> 3 -> 5 -> 0
    edges_v4 = [(0,2), (2,1), (1,4), (4,3), (3,5), (5,0)]
    g0, g1, g2 = compute_gaps(n_v, edges_v4, [])
    print(f"{'Reconnected':<20} | {g0:<10.4f} | {g1:<10.4f} | {g2:<10.4f}")

if __name__ == "__main__":
    run_reconnection_audit()
