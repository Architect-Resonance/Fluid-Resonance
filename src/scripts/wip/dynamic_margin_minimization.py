import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

def compute_physical_margin_minimal(nodes_flat, n_points, sigma=0.1, nu=0.01):
    nodes = nodes_flat.reshape((n_points, 3))
    N = n_points
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
    l_min_norm = evals[1] if len(evals) > 1 else 1e-9
    
    # 3. Physics (Continuum Proxies)
    Z = N * (1.0 / (4 * np.pi * sigma**2))
    D_cont = N * (nu * (1.0 / (8 * np.pi * sigma**4)))
    LB = nu * l_min_norm * Z * (N / 10.0)
    
    margin = D_cont / (LB + 1e-9)
    return margin

def run_dynamic_minimization(n_points=100, iterations=5):
    print(f"--- DYNAMIC MARGIN MINIMIZATION (Searching for Blow-up) ---")
    
    # Initial guess: A random cloud
    x0 = np.random.normal(0, 1, (n_points, 3)).flatten()
    
    res = minimize(compute_physical_margin_minimal, x0, args=(n_points,), 
                   method='Nelder-Mead', options={'maxiter': 500})
    
    print(f"Final Minimum Margin Found: {res.fun:.6f}")
    if res.fun < 1.0:
        print("!!! VIOLATION !!! Found a configuration that breaches the regularity floor.")
    else:
        print("SUCCESS: The 1.0 floor survives even systematic minimization.")

if __name__ == "__main__":
    run_dynamic_minimization()
