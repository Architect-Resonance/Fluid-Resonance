import numpy as np
from scipy.linalg import eigh

def check_star_resolution(n_filaments=7, segments_range=[5, 10, 20, 50], sigma=0.1, nu=0.01):
    """
    Sweeps N (segments) to see if the safety margin D/LB stabilizes or worsens.
    """
    print(f"--- RESOLUTION SWEEP: STAR TOPOLOGY (Sigma={sigma}) ---")
    print(f"{'N (Segments)':<15} | {'Gap (l_min)':<12} | {'Safety Margin':<15}")
    print("-" * 50)
    
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
        
        # 2. Map Phi
        W = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(all_nodes[i] - all_nodes[j])
                W[i,j] = W[j,i] = 1.0 / (dist + sigma)
        
        # 3. Laplacian Gap
        L = np.diag(W.sum(axis=1)) - W
        evals = eigh(L, eigvals_only=True)
        l_min = evals[1]
        
        # 4. Continuous Proxies
        Z = N * (1.0 / (4 * np.pi * sigma**2))
        D_cont = N * (nu * (1.0 / (8 * np.pi * sigma**4)))
        
        # 5. Result
        LB = nu * l_min * Z
        margin = D_cont / (LB + 1e-9)
        
        print(f"{s_count:<15} | {l_min:<12.6f} | {margin:<15.4f}")

if __name__ == "__main__":
    check_star_resolution()
