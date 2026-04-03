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
    return evals[1] # Gap

def run_twist_surgery(n_filaments=3, n_segments=100, twist_range=np.linspace(0, 4*np.pi, 20)):
    print(f"--- FILAMENT TWIST SURGERY: NORMALIZED GAP SEARCH ---")
    print(f"{'Twist (rad)':<15} | {'Normalized Gap':<20} | {'Helicity (Proxy)':<15}")
    print("-" * 55)
    
    for twist in twist_range:
        nodes = []
        for i in range(n_filaments):
            angle_offset = 2 * np.pi * i / n_filaments
            for z in np.linspace(0, 1, n_segments):
                # Helical path
                r = 0.5
                theta = angle_offset + twist * z
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                nodes.append(np.array([x, y, z]))
        
        gap = compute_normalized_gap(np.array(nodes))
        # Helicity proxy: linking number/twist-related
        helicity = twist * n_filaments 
        
        print(f"{twist:<15.4f} | {gap:<20.6f} | {helicity:<15.4f}")

if __name__ == "__main__":
    run_twist_surgery()
