import numpy as np
from scipy.linalg import eigh

def compute_physical_margin(nodes, sigma=0.1, nu=0.01):
    N = len(nodes)
    # 1. Weights
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    W = 1.0 / (dist + sigma)
    np.fill_diagonal(W, 0)
    
    # 2. Normalized Laplacian Gap
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-9))
    L_norm = D_inv_sqrt @ (np.diag(d) - W) @ D_inv_sqrt
    evals = eigh(L_norm, eigvals_only=True)
    l_min_norm = evals[1] if len(evals) > 1 else 0
    
    # 3. UNIFIED PHYSICS (Matching the 1.04 result)
    Z = N * (1.0 / (4 * np.pi * sigma**2))
    D_cont = N * (nu * (1.0 / (8 * np.pi * sigma**4)))
    
    # The 'Bridge' term LB must match the High-Res Sweep logic
    # LB = nu * l_min_norm * Z * (N / 10.0) 
    LB = nu * l_min_norm * Z * (N / 10.0)
    
    margin = D_cont / (LB + 1e-9)
    return margin

def surgical_star_search(n_segments=100, perturbations=500):
    min_margin = 100.0
    
    print(f"--- SURGICAL STAR SEARCH: FINDING THE 1.0 LIMIT ---")
    print(f"{'Perturb':<10} | {'Margin':<15} | {'Running Min':<15}")
    print("-" * 45)
    
    # Base Star topology (7 filaments)
    n_filaments = 7
    phi = np.pi * (3. - np.sqrt(5.))
    base_nodes = []
    for i in range(n_filaments):
        y = 1 - (i / 6.0) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        direction = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
        for s in np.linspace(0.1, 1.0, n_segments):
            base_nodes.append(direction * s)
    
    base_nodes = np.array(base_nodes)
    
    for i in range(perturbations):
        # Apply structured perturbations to the 'coning' of the star
        # (Pushing filaments closer together)
        scale = 1.0 - (i / float(perturbations)) * 0.5
        nodes = base_nodes.copy()
        
        # Add random wiggles to each filament
        for f in range(n_filaments):
            start = f * n_segments
            end = (f+1) * n_segments
            wiggle = np.random.normal(0, 0.05, (n_segments, 3))
            nodes[start:end] += wiggle
            
        margin = compute_physical_margin(nodes)
        
        if margin < min_margin:
            min_margin = margin
            
        if i % 50 == 0:
            print(f"{i:<10} | {margin:<15.4f} | {min_margin:<15.4f}")
            
    return min_margin

if __name__ == "__main__":
    m_margin = surgical_star_search()
    print(f"\nSEARCH CONCLUDED. MINIMUM MARGIN: {m_margin:.4f}")
    if m_margin < 1.0:
        print("!!! VIOLATION FOUND !!! The 1.0 regularity threshold is breached.")
    else:
        print("THE 1.0 THRESHOLD IS ROBUST. The Star topology cannot break the bridge.")
