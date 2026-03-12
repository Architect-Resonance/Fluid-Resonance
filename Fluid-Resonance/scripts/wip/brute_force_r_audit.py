import numpy as np
import networkx as nx

def check_graph_ratio(G, weight_halving=0.5):
    # Assume vertex 0 is the "Hub"
    adj = nx.to_numpy_array(G)
    hub_idx = 0
    
    def get_gap(w):
        m = adj.copy()
        # Scale hub edges
        for j in range(len(m)):
            if m[hub_idx, j] > 0:
                m[hub_idx, j] = m[j, hub_idx] = w
        
        L = np.diag(m.sum(axis=1)) - m
        # Grounded Laplacian (remove hub)
        L_grounded = np.delete(np.delete(L, hub_idx, axis=0), hub_idx, axis=1)
        evals = np.linalg.eigvalsh(L_grounded)
        return evals[0]

    g1 = get_gap(1.0)
    g2 = get_gap(weight_halving)
    if g2 == 0: return 0
    return g1 / g2

if __name__ == "__main__":
    R_TARGET = 1.857312485
    print(f"Searching for graphs with R close to {R_TARGET}...")
    
    # Try all small graphs up to 8 nodes
    for n in range(4, 9):
        # Generate random connected graphs
        for i in range(1000):
            G = nx.fast_gnp_random_graph(n, 0.4)
            if nx.is_connected(G):
                r = check_graph_ratio(G)
                if abs(r - R_TARGET) < 0.01:
                    print(f"Nodes={n}, Graph={i}, R={r:.6f}")
                if abs(r - R_TARGET) < 1e-5:
                    print(f"!!! MATCH FOUND !!! Nodes={n}, R={r:.10f}")
                    # Print edges
                    print(G.edges())
                    break

