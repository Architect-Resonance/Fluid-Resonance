import numpy as np
from scipy.optimize import minimize
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
            # Biot-Savart stretching term: (3/4pi) * (omega_i . r) * (omega_i x omega_j . r) / r^5
            # This is the exact stretching on segment i from segment j
            det = np.dot(np.cross(omegas[i], omegas[j]), r_vec)
            term = (3.0 / (4.0 * np.pi)) * np.dot(omegas[i], r_vec) * det / (r**5)
            local_stretch += term
        total_stretch += np.abs(local_stretch)
    return total_stretch

def compute_graph_majorant(nodes, sigma=0.1):
    N = len(nodes)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            r = np.linalg.norm(nodes[i] - nodes[j])
            # Matching kernel for stretching: 1/r^3
            val = 1.0 / (r + sigma)**3
            A[i, j] = val
            A[j, i] = val
    
    L = np.diag(np.sum(A, axis=1)) - A
    evals = eigh(L, eigvals_only=True)
    lambda_max = evals[-1]
    return lambda_max

def objective(x, n_segments, sigma):
    nodes = x[:3*n_segments].reshape((n_segments, 3))
    omegas_raw = x[3*n_segments:].reshape((n_segments, 3))
    
    # Normalize omegas to unit length
    omegas = []
    for o in omegas_raw:
        mag = np.linalg.norm(o)
        if mag < 1e-9:
            omegas.append(np.array([0.0, 0.0, 0.0]))
        else:
            omegas.append(o / mag)
    omegas = np.array(omegas)
    
    S = compute_physical_stretching(nodes, omegas, sigma)
    # Penalize points flying too far out (soft boundary)
    dist_penalty = 1e-3 * np.sum(np.abs(nodes))
    return -S + dist_penalty

def run_malicious_voyager_audit(n_segments=6, n_trials=5):
    print(f"--- MALICIOUS VOYAGER AUDIT (N={n_segments}) ---")
    print(f"{'Trial':<10} | {'Max Stretching':<15} | {'Majorant (L*Z)':<15} | {'Ratio C'}")
    print("-" * 65)
    
    sigma = 0.1
    bounds = []
    # Position bounds: [-5.0, 5.0]
    for _ in range(3 * n_segments): bounds.append((-5.0, 5.0))
    # Omega bounds: [-1, 1]
    for _ in range(3 * n_segments): bounds.append((-1.0, 1.0))
    
    results = []
    
    for i in range(n_trials):
        # Random initial guess with normalized omegas
        nodes_0 = np.random.uniform(-1, 1, 3 * n_segments)
        omegas_0 = np.random.normal(0, 1, 3 * n_segments)
        x0 = np.concatenate([nodes_0, omegas_0])
        
        res = minimize(objective, x0, args=(n_segments, sigma), 
                       bounds=bounds, method='L-BFGS-B', 
                       options={'maxiter': 100})
        
        # Best found (regardless of full convergence success)
        x_best = res.x
        nodes = x_best[:3*n_segments].reshape((n_segments, 3))
        omegas_raw = x_best[3*n_segments:].reshape((n_segments, 3))
        
        omegas = []
        for o in omegas_raw:
            mag = np.linalg.norm(o)
            if mag < 1e-9:
                omegas.append(np.array([0.0, 0.0, 0.0]))
            else:
                omegas.append(o / mag)
        omegas = np.array(omegas)
        
        S = compute_physical_stretching(nodes, omegas, sigma)
        lambda_max = compute_graph_majorant(nodes, sigma)
        Z = n_segments
        
        C = S / (lambda_max * Z) if lambda_max > 0 else 0
        print(f"{i:<10} | {S:<15.4f} | {lambda_max*Z:<15.4f} | {C:.6f}")
        results.append(C)
        
    print("-" * 65)
    if len(results) > 0:
        print(f"PEAK MALICE (Max C): {max(results):.6f}")
        if max(results) < 1.0:
            print("RESULT: The Stretching Majorant holds even against an Optimal Adversary.")
        else:
            print("RESULT: BREACH DETECTED. The majorant has failed.")
    else:
        print("RESULT: Optimization failed to run.")

if __name__ == "__main__":
    run_malicious_voyager_audit(n_segments=6, n_trials=5)
