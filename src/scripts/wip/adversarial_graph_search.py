import numpy as np
import networkx as nx
from scipy.linalg import eigh

def analyze_graph_dissipation(G, hub_weight=1.0):
    """
    Computes the Stokes/Reduced ratio for an arbitrary graph G.
    """
    # 1. Laplacian
    adj = nx.to_numpy_array(G)
    # Scale interactions by hub-weight proxy (if center exists)
    deg = np.diag(adj.sum(axis=1))
    L = deg - adj
    
    # 2. Stokes Gap (Full)
    evals = np.sort(eigh(L, eigvals_only=True))
    gap_full = evals[1] if len(evals) > 1 else 0
    
    # 3. Reduced Gap (Remove one random high-influence edge)
    edges = list(G.edges())
    if not edges: return 0
    e = edges[np.random.randint(len(edges))]
    G_red = G.copy()
    G_red.remove_edge(*e)
    adj_red = nx.to_numpy_array(G_red)
    L_red = np.diag(adj_red.sum(axis=1)) - adj_red
    evals_red = np.sort(eigh(L_red, eigvals_only=True))
    gap_red = evals_red[1] if len(evals_red) > 1 else 0
    
    return gap_full / (gap_red + 1e-9)

def min_graph_search(n_nodes=20, trials=100):
    """
    Scans different graph topologies (Erdos-Renyi, Barabasi-Albert, Sparse)
    to find the lower bound of the snap ratio.
    """
    print(f"--- MIN-GRAPH SEARCH (Adversarial Audit: N={n_nodes}) ---")
    
    results = {
        "Random (p=0.2)": [],
        "Scale-Free (m=2)": [],
        "Sparse (p=0.05)": [],
        "Star-like": []
    }
    
    for _ in range(trials):
        # 1. Random (Erdos-Renyi) - simulating 'Haze'
        rg = nx.erdos_renyi_graph(n_nodes, 0.2)
        if nx.is_connected(rg):
            results["Random (p=0.2)"].append(analyze_graph_dissipation(rg))
            
        # 2. Scale-Free (Barabasi) - simulating 'Filaments'
        ba = nx.barabasi_albert_graph(n_nodes, 2)
        results["Scale-Free (m=2)"].append(analyze_graph_dissipation(ba))
        
        # 3. Sparse - Meridian's Turbulence density (0.05)
        sp = nx.erdos_renyi_graph(n_nodes, 0.05)
        if nx.is_connected(sp):
            results["Sparse (p=0.05)"].append(analyze_graph_dissipation(sp))
            
        # 4. Star-Topology (The biased model)
        st = nx.star_graph(n_nodes - 1)
        results["Star-like"].append(analyze_graph_dissipation(st))

    for k, v in results.items():
        if v:
            print(f"{k:18}: Mean Snap Ratio = {np.mean(v):.4f} (Min: {np.min(v):.4f})")

if __name__ == "__main__":
    min_graph_search()
