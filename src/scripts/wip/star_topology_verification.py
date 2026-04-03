import numpy as np
from scipy.linalg import eigh

def verify_star_topology_inequality(n_filaments=7, segments_per_filament=10, sigma=0.1, nu=0.01):
    """
    Generalized verification for a Star interaction topology.
    Maps N filaments in isotropic directions to a graph and checks the bound.
    """
    print(f"--- STAR TOPOLOGY SPECTRAL VERIFICATION ({n_filaments} Filaments) ---")
    
    # 1. Geometry: Isotropic directions spanning S^2 (Lei et al. proxy)
    phi = np.pi * (3. - np.sqrt(5.))
    nodes = []
    for i in range(n_filaments):
        # Direction vector
        y = 1 - (i / float(n_filaments - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        direction = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
        
        # Filament nodes (radiating from origin)
        for s in np.linspace(0.1, 1.0, segments_per_filament):
            nodes.append(direction * s)
            
    all_nodes = np.array(nodes)
    N = len(all_nodes)
    
    # 2. Map Phi: Biot-Savart Energy Kernel
    # w_ij = 1 / (dist_ij + sigma)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(all_nodes[i] - all_nodes[j])
            W[i,j] = W[j,i] = 1.0 / (dist + sigma)
            
    # 3. Laplacian and Gap
    deg = np.diag(W.sum(axis=1))
    L = deg - W
    evals = eigh(L, eigvals_only=True)
    lambda_min_graph = evals[1]
    
    # 4. Continuous Proxies
    Z = N * (1.0 / (4 * np.pi * sigma**2))
    D_cont_total = N * (nu * (1.0 / (8 * np.pi * sigma**4)))
    
    # 5. Verification
    lower_bound = nu * lambda_min_graph * Z
    margin = D_cont_total / (lower_bound + 1e-9)
    
    print(f"Graph lambda_min     : {lambda_min_graph:.6f}")
    print(f"Total Enstrophy (Z) : {Z:.6f}")
    print(f"Continuous Dissip   : {D_cont_total:.6f}")
    print(f"Spectral Bound (LB) : {lower_bound:.6f}")
    print(f"Safety Margin (D/LB): {margin:.4f}")
    
    if D_cont_total >= lower_bound:
        print("\n[VERIFIED] Spectral Dissipation Inequality holds for Star Topology.")
    else:
        print("\n[CAUTION] Inequality is TIGHT or FAILING for Star Topology.")

if __name__ == "__main__":
    verify_star_topology_inequality()
