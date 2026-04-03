import numpy as np
from scipy.linalg import eigh

def compute_physical_margin(nodes, sigma=0.1, nu=0.01):
    N = len(nodes)
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    W = 1.0 / (dist + sigma)
    np.fill_diagonal(W, 0)
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-9))
    L_norm = D_inv_sqrt @ (np.diag(d) - W) @ D_inv_sqrt
    evals = eigh(L_norm, eigvals_only=True)
    l_min_norm = evals[1]
    
    Z = N * (1.0 / (4 * np.pi * sigma**2))
    D_cont = N * (nu * (1.0 / (8 * np.pi * sigma**4)))
    LB = nu * l_min_norm * Z * (N / 10.0)
    
    return D_cont / (LB + 1e-9)

def high_res_star_floor_check(n_segments_range=[100, 200, 300]):
    print(f"--- HIGH-RES STAR FLOOR CHECK (Universal Lower Bound Search) ---")
    print(f"{'N_seg':<10} | {'N_total':<10} | {'Min Margin (Worst Case)':<25}")
    print("-" * 50)
    
    for n_seg in n_segments_range:
        # Base Star topology (7 filaments)
        n_filaments = 7
        phi = np.pi * (3. - np.sqrt(5.))
        nodes = []
        for i in range(n_filaments):
            y = 1 - (i / 6.0) * 2; radius = np.sqrt(1 - y * y); theta = phi * i
            direction = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
            for s in np.linspace(0.1, 1.0, n_seg): nodes.append(direction * s)
        
        nodes = np.array(nodes)
        
        # Test 10 random perturbations to find 'Worst Case' at this resolution
        margins = []
        for _ in range(10):
            p_nodes = nodes + np.random.normal(0, 0.05, nodes.shape)
            margins.append(compute_physical_margin(p_nodes))
            
        print(f"{n_seg:<10} | {len(nodes):<10} | {np.min(margins):<25.6f}")

if __name__ == "__main__":
    high_res_star_floor_check()
