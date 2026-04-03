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

def create_star(origin, n_filaments=7, n_seg=50):
    phi = np.pi * (3. - np.sqrt(5.))
    nodes = []
    for i in range(n_filaments):
        y = 1 - (i / 6.0) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        direction = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
        for s in np.linspace(0.1, 1.0, n_seg):
            nodes.append(origin + direction * s)
    return np.array(nodes)

def run_colliding_star_audit(dist_range=np.linspace(0.1, 5, 20)):
    print(f"--- COLLIDING STAR AUDIT: MULTI-VORTEX INTERACTION ---")
    print(f"{'Distance':<10} | {'Safety Margin':<15} | {'Status':<10}")
    print("-" * 40)
    
    for d in dist_range:
        star1 = create_star(np.array([0, 0, 0]))
        star2 = create_star(np.array([d, 0, 0]))
        nodes = np.concatenate([star1, star2])
        
        margin = compute_physical_margin(nodes)
        status = "PASS" if margin >= 1.0 else "FAIL"
        print(f"{d:<10.2f} | {margin:<15.4f} | {status:<10}")

if __name__ == "__main__":
    run_colliding_star_audit()
