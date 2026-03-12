import numpy as np

def compute_spectral_ratio_refined(n_branches, cross_weight=0.1):
    """
    Computes the spectral ratio R for an n-branch star graph with
    cross-spoke interactions (The 'Syrup').
    """
    n = n_branches
    hub_idx = n
    W = np.zeros((n+1, n+1))
    
    # 1. Hub-Spoke Interactions (The 'Chord')
    for i in range(n):
        W[i, hub_idx] = W[hub_idx, i] = 1.0
        
    # 2. Cross-Spoke Interactions (The 'Syrup')
    # Each spoke interacts with its neighbors in a ring
    for i in range(n):
        j = (i + 1) % n
        W[i, j] = W[j, i] = cross_weight
    
    # Function to get gap
    def get_gap(weight_mod):
        W_curr = W.copy()
        # Surgery: Adjust the hub interaction
        W_curr[:, hub_idx] *= weight_mod
        W_curr[hub_idx, :] *= weight_mod
        
        L = np.diag(W_curr.sum(axis=1)) - W_curr
        # Grounded Laplacian (remove hub)
        L_grounded = L[:n, :n]
        evals = np.linalg.eigvalsh(L_grounded)
        return evals[0]

    # R = Gap_1.0 / Gap_0.5 (Halving the chord)
    gap1 = get_gap(1.0)
    gap2 = get_gap(0.5)
    
    return gap1 / gap2

if __name__ == "__main__":
    print(f"{'Branches':<10} | {'Ratio R (w=0.1)':<15} | {'Ratio R (w=0.2)'}")
    print("-" * 50)
    for n in range(3, 15):
        r1 = compute_spectral_ratio_refined(n, cross_weight=0.1)
        r2 = compute_spectral_ratio_refined(n, cross_weight=0.2)
        print(f"{n:<10} | {r1:.10f} | {r2:.10f}")
