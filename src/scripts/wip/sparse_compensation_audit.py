import numpy as np
import networkx as nx
from scipy.linalg import eigh

def analyze_sparse_compensation(n_nodes=20, density=0.05, weight_scale=10.0):
    """
    Test if high-vorticity 'Intense Edges' can rescue the spectral gap
    in the sparse networks observed in real turbulence (Meridian Vector 3).
    """
    print(f"--- SPARSE COMPENSATION AUDIT (Density: {density}) ---")
    
    # 1. Create Sparse Turbulent-like Graph
    G = nx.erdos_renyi_graph(n_nodes, density)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n_nodes, density)
    
    adj = nx.to_numpy_array(G)
    
    # 2. Base Case: Uniform Weights
    L_uni = np.diag(adj.sum(axis=1)) - adj
    gap_uni = np.sort(eigh(L_uni, eigvals_only=True))[1]
    
    # 3. Compensated Case: High-Vorticity Core (Lei et al. hypothesis)
    # We assign higher weights to a 'spanning set' of directions
    adj_comp = adj.copy()
    edges = list(G.edges())
    # Spike the weights of the 'core' interaction
    for i in range(min(5, len(edges))):
        u, v = edges[i]
        adj_comp[u, v] = adj_comp[v, u] = weight_scale
        
    L_comp = np.diag(adj_comp.sum(axis=1)) - adj_comp
    gap_comp = np.sort(eigh(L_comp, eigvals_only=True))[1]
    
    # 4. Reduced Gap (Simulating 'Twist' removal)
    G_red = G.copy()
    u, v = edges[0]
    G_red.remove_edge(u, v)
    adj_red = nx.to_numpy_array(G_red)
    # Maintain weight scale for remaining core
    for i in range(1, min(5, len(edges))):
        u_c, v_c = edges[i]
        adj_red[u_c, v_c] = adj_red[v_c, u_c] = weight_scale
        
    L_red = np.diag(adj_red.sum(axis=1)) - adj_red
    gap_red = np.sort(eigh(L_red, eigvals_only=True))[1]
    
    ratio = gap_comp / (gap_red + 1e-9)
    
    return gap_uni, gap_comp, ratio

if __name__ == "__main__":
    uni, comp, ratio = analyze_sparse_compensation()
    print(f"Uniform Sparse Gap : {uni:.6f}")
    print(f"Compensated Gap    : {comp:.6f}")
    print(f"Snap Ratio         : {ratio:.6f}")
    
    if ratio < 1.518:
        print("\nREFUTATION: Sparse networks cannot achieve the 1.518 Snap through weight alone.")
    else:
        print("\nADVERSARIAL WIN: Weight concentration rescues sparse connectivity.")
