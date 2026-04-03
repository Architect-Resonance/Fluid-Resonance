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

def generate_fractal_cloud(n_filaments=10, n_seg=50, alpha=5/3):
    nodes = []
    for f in range(n_filaments):
        pos = np.random.normal(0, 1, 3)
        direction = np.random.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        for s in range(n_seg):
            # Kolmogorov-like noise/perturbation at each scale
            scale = (s + 1)**(-alpha)
            pos += direction * 0.1 + np.random.normal(0, scale, 3)
            nodes.append(pos.copy())
    return np.array(nodes)

def run_fractal_audit(iterations=20):
    print(f"--- FRACTAL TURBULENCE AUDIT: KOLMOGOROV FILAMENTS ---")
    print(f"{'Iter':<10} | {'Safety Margin':<15} | {'Status':<10}")
    print("-" * 40)
    
    margins = []
    for i in range(iterations):
        nodes = generate_fractal_cloud()
        margin = compute_physical_margin(nodes)
        margins.append(margin)
        status = "PASS" if margin >= 1.0 else "FAIL"
        print(f"{i:<10} | {margin:<15.4f} | {status:<10}")
        
    print(f"\nFractal Mean Margin: {np.mean(margins):.4f}")
    print(f"Fractal Min Margin: {np.min(margins):.4f}")

if __name__ == "__main__":
    run_fractal_audit()
