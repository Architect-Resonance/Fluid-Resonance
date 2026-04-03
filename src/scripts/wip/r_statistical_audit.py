import numpy as np
import scipy.linalg as la
from spectral_ns_solver import SpectralNSSolver

def audit_r_statistical():
    print(f"--- S52: STATISTICAL INVARIANT AUDIT ---")
    print(f"{'Cluster':<10} | {'R Ratio'}")
    print("-" * 30)
    
    solver = SpectralNSSolver(N=64)
    solver.initialize_pelz_flow()
    u, omega = solver.get_real_fields()
    mag = np.sqrt(np.sum(omega**2, axis=0))
    
    # 1. Sample many top points
    N_total = 512
    idx = np.argsort(mag.flatten())[-N_total:]
    coords = np.array(np.unravel_index(idx, mag.shape)).T * (2*np.pi / 64)
    
    # 2. Extract multiple clusters
    r_values = []
    # We take every 8th point as a 'hub' candidate to cover distinct regions
    for i in range(0, N_total, 32):
        hub_coord = coords[i]
        # Find 6 neighbors
        dists = np.linalg.norm(coords - hub_coord, axis=1)
        nn_idx = np.argsort(dists)[1:7]
        
        # Build local sub-adjacency
        cluster_coords = np.vstack([coords[nn_idx], hub_coord])
        W_sub = np.zeros((7, 7))
        for j in range(7):
            for k in range(j+1, 7):
                d = np.linalg.norm(cluster_coords[j] - cluster_coords[k])
                W_sub[j, k] = W_sub[k, j] = 1.0 / (d**2 + 0.1)
                
        # Laplacian Ratio
        L_sub = np.diag(W_sub.sum(axis=1)) - W_sub
        evals = la.eigvalsh(L_sub)
        gap_full = evals[1]
        L_red = L_sub[:-1, :-1]
        evals_red = la.eigvalsh(L_red)
        gap_red = evals_red[0]
        
        r = gap_full / gap_red
        r_values.append(r)
        print(f"Cluster {i//32:<3} | {r:.6f}")

    print("-" * 30)
    mean_r = np.mean(r_values)
    std_r = np.std(r_values)
    print(f"MEAN R: {mean_r:.6f} +/- {std_r:.6f}")
    print(f"TARGET: 1.85731")

if __name__ == "__main__":
    audit_r_statistical()
