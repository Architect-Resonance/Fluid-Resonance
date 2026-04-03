import numpy as np
from scipy.linalg import eigh

def compute_physical_stretching(nodes, omegas, sigma=0.1):
    N = len(nodes)
    total_stretch = 0
    for i in range(N):
        local_stretch = 0
        for j in range(N):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r2 = np.dot(r_vec, r_vec) + sigma**2
            r = np.sqrt(r2)
            det = np.dot(np.cross(omegas[i], omegas[j]), r_vec)
            term = (3.0 / (4.0 * np.pi)) * np.dot(omegas[i], r_vec) * det / (r**5)
            local_stretch += term
        total_stretch += np.abs(local_stretch)
    return total_stretch

def compute_graph_majorant(nodes, omegas, sigma=0.1):
    N = len(nodes)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            r_vec = nodes[i] - nodes[j]
            r = np.sqrt(np.dot(r_vec, r_vec) + sigma**2)
            W[i,j] = W[j,i] = (3.0 / (4.0 * np.pi)) / (r**3)
    
    d = W.sum(axis=1)
    L = np.diag(d) - W
    evals = eigh(L, eigvals_only=True)
    l_max = evals[-1]
    
    Z = np.sum(np.linalg.norm(omegas, axis=1)**2)
    return l_max * Z

def create_star(origin, n_filaments=7, n_seg=50):
    phi = np.pi * (3. - np.sqrt(5.))
    nodes = []
    omegas = []
    for i in range(n_filaments):
        y = 1 - (i / 6.0) * 2; radius = np.sqrt(1 - y * y); theta = phi * i
        direction = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
        for s in np.linspace(0.1, 1.0, n_seg):
            pos = origin + direction * s
            nodes.append(pos)
            omegas.append(direction)
    return np.array(nodes), np.array(omegas)

def run_adversarial_stretching_audit():
    print(f"--- ADVERSARIAL STRETCHING AUDIT: STAR TOPOLOGY ---")
    print(f"{'Structure':<25} | {'Stretching':<15} | {'Graph Bound':<15} | {'Ratio C':<10}")
    print("-" * 75)
    
    # 1. Single Star
    nodes, omegas = create_star(np.zeros(3))
    S = compute_physical_stretching(nodes, omegas)
    GB = compute_graph_majorant(nodes, omegas)
    print(f"{'Single Star':<25} | {S:<15.4f} | {GB:<15.4f} | {S/GB:<10.6f}")
    
    # 2. Colliding Stars (d=0.36, the crisis distance)
    star1_nodes, star1_omegas = create_star(np.array([0, 0, 0]))
    star2_nodes, star2_omegas = create_star(np.array([0.36, 0, 0]))
    nodes = np.concatenate([star1_nodes, star2_nodes])
    omegas = np.concatenate([star1_omegas, star2_omegas])
    S = compute_physical_stretching(nodes, omegas)
    GB = compute_graph_majorant(nodes, omegas)
    print(f"{'Colliding Stars (d=0.36)':<25} | {S:<15.4f} | {GB:<15.4f} | {S/GB:<10.6f}")

    # 3. Random Orthogonal Clusters (Maximize Det)
    nodes = []
    omegas = []
    for _ in range(30):
        pos = np.random.uniform(-0.5, 0.5, 3)
        # Choose omega to maximize cross product with pos (crudely)
        nodes.append(pos)
        omegas.append(np.cross(pos, [1,0,0]) + 1e-3)
    nodes = np.array(nodes)
    omegas = np.array(omegas)
    S = compute_physical_stretching(nodes, omegas)
    GB = compute_graph_majorant(nodes, omegas)
    print(f"{'Orthogonal Cluster':<25} | {S:<15.4f} | {GB:<15.4f} | {S/GB:<10.6f}")

if __name__ == "__main__":
    run_adversarial_stretching_audit()
