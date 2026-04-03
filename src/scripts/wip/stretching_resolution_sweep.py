import numpy as np
from scipy.linalg import eigh

def compute_stretching_and_graph(n_filaments=5, n_seg=50, sigma=0.1):
    nodes = []
    omegas = []
    for _ in range(n_filaments):
        origin = np.random.uniform(-1, 1, 3)
        direction = np.random.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        length = 1.0
        for s in np.linspace(0, length, n_seg):
            nodes.append(origin + direction * s)
            omegas.append(direction)
    nodes = np.array(nodes)
    omegas = np.array(omegas)
    N = len(nodes)
    
    # Stretching
    S = 0
    for i in range(N):
        local_S = 0
        for j in range(N):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r = np.sqrt(np.dot(r_vec, r_vec) + sigma**2)
            det = np.dot(np.cross(omegas[i], omegas[j]), r_vec)
            term = (3.0 / (4.0 * np.pi)) * np.dot(omegas[i], r_vec) * det / (r**5)
            local_S += term
        S += np.abs(local_S)
    
    # Graph Bound
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            r = np.sqrt(np.dot(nodes[i]-nodes[j], nodes[i]-nodes[j]) + sigma**2)
            W[i,j] = W[j,i] = (3.0 / (4.0 * np.pi)) / (r**3)
    d = W.sum(axis=1)
    L = np.diag(d) - W
    evals = eigh(L, eigvals_only=True)
    l_max = evals[-1]
    Z = np.sum(np.linalg.norm(omegas, axis=1)**2)
    
    return S, l_max, Z

def run_resolution_sweep_c(n_seg_range=[20, 40, 80, 160]):
    print(f"--- RESOLUTION SWEEP: STRETCHING BOUND CONSTANT C ---")
    print(f"{'N_total':<10} | {'Ratio C':<15} | {'Scaling':<10}")
    print("-" * 45)
    
    prev_c = None
    for n_seg in n_seg_range:
        S, L, Z = compute_stretching_and_graph(n_seg=n_seg)
        c = S / (L * Z + 1e-9)
        
        scaling = "INIT" if prev_c is None else f"{c/prev_c:.4f}x"
        print(f"{n_seg*5:<10} | {c:<15.6f} | {scaling:<10}")
        prev_c = c

if __name__ == "__main__":
    run_resolution_sweep_c()
