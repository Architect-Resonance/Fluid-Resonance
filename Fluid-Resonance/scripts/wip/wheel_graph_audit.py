import numpy as np

def compute_wheel_ratio(n_rim_vertices):
    """
    Computes the spectral ratio R for a Wheel Graph W_{n+1}.
    Surgery: Remove the hub (Biot-Savart surgery).
    R = Gap_original / Gap_after_surgery
    """
    n = n_rim_vertices
    hub_idx = n
    W = np.zeros((n+1, n+1))
    
    # 1. Hub-Rim Interactions (The 'Chord')
    for i in range(n):
        W[i, hub_idx] = W[hub_idx, i] = 1.0
        
    # 2. Rim-Rim Interactions (The 'Syrup/Circle')
    for i in range(n):
        j = (i + 1) % n
        W[i, j] = W[j, i] = 1.0 # Rim edges are strong
    
    def get_gap(hub_weight):
        W_curr = W.copy()
        for i in range(n):
            W_curr[i, hub_idx] = W_curr[hub_idx, i] = hub_weight
            
        # Grounded Laplacian (remove hub)
        # This models the 'Flow' in the presence of a hub
        L = np.diag(W_curr.sum(axis=1)) - W_curr
        L_grounded = L[:n, :n]
        evals = np.linalg.eigvalsh(L_grounded)
        return evals[0]

    # Claude's R is usually Original / Surgically Reduced
    # Let's test Hub=1.0 vs Hub=0.1 (surgery)
    gap_orig = get_gap(1.0)
    gap_surg = get_gap(0.1)
    
    return gap_orig / gap_surg

if __name__ == "__main__":
    print(f"{'Rim Edges/b1':<15} | {'Ratio R (Hub 1.0->0.1)'}")
    print("-" * 40)
    for b1 in range(3, 15):
        r = compute_wheel_ratio(b1)
        print(f"{b1:<15} | {r:.10f}")
