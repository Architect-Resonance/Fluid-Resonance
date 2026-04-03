import numpy as np
from scipy.linalg import eigh

def compute_normalized_gap(nodes, sigma=0.1):
    N = len(nodes)
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    W = 1.0 / (dist + sigma)
    np.fill_diagonal(W, 0)
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-9))
    L_norm = D_inv_sqrt @ (np.diag(d) - W) @ D_inv_sqrt
    evals = eigh(L_norm, eigvals_only=True)
    return evals[1] if len(evals) > 1 else 0

def find_global_min_gap(n_points=100, iterations=1000):
    """
    Randomized search for the 'Worst Case' topology.
    What is the absolute minimum of lambda_norm?
    """
    min_gap = 10.0
    best_config = None
    
    print(f"--- GLOBAL MINIMUM GAP SEARCH: NORMALIZED LAPLACIAN ---")
    print(f"{'Iteration':<10} | {'Current Gap':<15} | {'Global Min':<15}")
    print("-" * 45)
    
    # Start with a random sphere
    nodes = np.random.normal(0, 1, (n_points, 3))
    
    for i in range(iterations):
        # Apply a 'Bottleneck' perturbation (trying to split the graph)
        perturbation = np.random.normal(0, 0.05, (n_points, 3))
        # Sometimes try to polarize the points into two clusters
        if i % 10 == 0:
            pivot = np.random.choice([1, -1], n_points)
            nodes += (pivot[:, np.newaxis] * 0.1)
            
        nodes += perturbation
        # Normalize to keep it in a bounding box
        nodes /= (np.max(np.abs(nodes)) + 1e-9)
        
        gap = compute_normalized_gap(nodes)
        
        if gap < min_gap:
            min_gap = gap
            best_config = nodes.copy()
            
        if i % 100 == 0:
            print(f"{i:<10} | {gap:<15.6f} | {min_gap:<15.6f}")
            
    return min_gap, best_config

if __name__ == "__main__":
    m_gap, _ = find_global_min_gap()
    print(f"\nSEARCH CONCLUDED. FOUND LOWER BOUND: {m_gap:.6f}")
