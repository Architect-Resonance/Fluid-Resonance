import numpy as np
from scipy.linalg import eigh

def compute_normalized_r(nodes, sigma=0.1):
    N = len(nodes)
    # 1. Weights
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    W = 1.0 / (dist + sigma)
    np.fill_diagonal(W, 0)
    
    # 2. Normalized Laplacian
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-9))
    L = np.diag(d) - W
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    evals = eigh(L_norm, eigvals_only=True)
    gap = evals[1]
    
    # 3. Renormalized Z
    # Following Meridian's suggestion: if we weight enstrophy by degree
    # Z_renorm = trace(D) or similar? 
    # For now, we test the standard ratio Z = N (since each point is unit vorticity)
    # but normalized by the 'average interaction strength'
    Z_renorm = N / np.mean(d) 
    
    return gap, Z_renorm, gap * Z_renorm

def run_critical_r_search_normalized():
    print(f"--- NORMALIZED 'CRITICAL R' SEARCH ---")
    print(f"{'Topology':<15} | {'Gap':<10} | {'Z_renorm':<10} | {'R_norm':<10}")
    print("-" * 55)
    
    # 1. 7-Filament Star
    nodes_star = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(7):
        y = 1 - (i / 6.0) * 2; radius = np.sqrt(1 - y * y); theta = phi * i
        direction = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
        for s in np.linspace(0.1, 1.0, 50): nodes_star.append(direction * s)
    g, z, r = compute_normalized_r(np.array(nodes_star))
    print(f"{'Star':<15} | {g:<10.4f} | {z:<10.4f} | {r:<10.4f}")

    # 2. Two Parallel Tubes
    nodes_para = []
    for x in [-0.5, 0.5]:
        for z in np.linspace(0, 5, 175): nodes_para.append(np.array([x, 0, z]))
    g, z, r = compute_normalized_r(np.array(nodes_para))
    print(f"{'Parallel':<15} | {g:<10.4f} | {z:<10.4f} | {r:<10.4f}")
    
    # 3. Random Cluster
    nodes_rand = np.random.normal(0, 0.5, (350, 3))
    g, z, r = compute_normalized_r(nodes_rand)
    print(f"{'Random':<15} | {g:<10.4f} | {z:<10.4f} | {r:<10.4f}")

if __name__ == "__main__":
    run_critical_r_search_normalized()
