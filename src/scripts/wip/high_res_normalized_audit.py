import numpy as np
from scipy.linalg import eigh
import time

def high_res_normalized_audit(n_filaments=7, segments_range=[100, 200, 500, 1000], sigma=0.1):
    print(f"--- HIGH-RESOLUTION NORMALIZED LAPLACIAN AUDIT (Trap #7 Check) ---")
    print(f"{'N_seg':<10} | {'N_total':<10} | {'Gap (l_norm)':<15} | {'Compute Time (s)':<15}")
    print("-" * 60)
    
    for s_count in segments_range:
        start_time = time.time()
        # 1. Geometry (Spherical Star)
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
        
        # 2. Optimized Weight Construction (avoiding full N^2 matrix in memory if possible)
        # For N=1000, N^2 = 1e6, so it fits in memory. 
        # For N=5000, we'd need to be more careful.
        
        # Vectorized weight calculation
        diff = all_nodes[:, np.newaxis, :] - all_nodes[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        W = 1.0 / (dist + sigma)
        np.fill_diagonal(W, 0)
        
        # 3. Normalized Laplacian Gap
        d = W.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-9))
        L = np.diag(d) - W
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        
        evals = eigh(L_norm, eigvals_only=True)
        l_min_norm = evals[1] # First non-zero eigenvalue
        
        elapsed = time.time() - start_time
        print(f"{s_count:<10} | {N:<10} | {l_min_norm:<15.6f} | {elapsed:<15.2f}")

if __name__ == "__main__":
    high_res_normalized_audit()
