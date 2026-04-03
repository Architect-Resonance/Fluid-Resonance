import numpy as np
from scipy.linalg import eigh

def check_normalized_resolution(n_filaments=7, segments_range=[5, 10, 20, 50, 100], sigma=0.1, nu=0.01):
    """
    Sweeps N (segments) using the NORMALIZED LAPLACIAN:
    L_norm = D^-1/2 * (D - W) * D^-1/2
    Tests if the safety margin stabilizes or continues to collapse.
    """
    print(f"--- NORMALIZED RESOLUTION SWEEP: STAR TOPOLOGY (Sigma={sigma}) ---")
    print(f"{'N_seg':<10} | {'Gap (l_norm)':<12} | {'Safety Margin (D/LB)':<20}")
    print("-" * 55)
    
    for s_count in segments_range:
        # 1. Geometry
        phi = np.pi * (3. - np.sqrt(5.))
        nodes = []
        for i in range(n_filaments):
            y = 1 - (i / float(n_filaments - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            direction = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
            for s in np.linspace(0.1, 1.0, s_count):
                nodes.append(direction * s)
        
        all_nodes = np.array(nodes)
        N = len(all_nodes)
        
        # 2. Map Phi (Biot-Savart Energy)
        W = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(all_nodes[i] - all_nodes[j])
                W[i,j] = W[j,i] = 1.0 / (dist + sigma)
        
        # 3. Normalized Laplacian Gap
        d = W.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-9))
        L = np.diag(d) - W
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        
        evals = eigh(L_norm, eigvals_only=True)
        l_min_norm = evals[1]
        
        # 4. Continuous Proxies
        # Dissipation scales as N (sum of segments)
        Z = N * (1.0 / (4 * np.pi * sigma**2))
        D_cont = N * (nu * (1.0 / (8 * np.pi * sigma**4)))
        
        # 5. Result
        # We need a new 'Physical Constant' K for normalized scaling
        # LB = K * nu * l_min_norm * Z
        # Initially we check if l_min_norm itself is resolution-invariant
        LB = nu * l_min_norm * Z * (N / 10.0) # Heuristic scaling check
        margin = D_cont / (LB + 1e-9)
        
        print(f"{s_count:<10} | {l_min_norm:<12.6f} | {margin:<20.4f}")

if __name__ == "__main__":
    check_normalized_resolution()
