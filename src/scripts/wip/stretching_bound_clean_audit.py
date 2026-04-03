import numpy as np
from scipy.linalg import eigh

def compute_physical_stretching(nodes, omegas, sigma=0.1):
    """
    Computes total stretching using the exact Biot-Savart formula for point vortices:
    Sum_i omega_i . S_i omega_i = Sum_i Sum_j (3/4pi) * (omega_i . r_ij) * Det(omega_i, omega_j, r_ij) / r_ij^5
    """
    N = len(nodes)
    total_stretch = 0
    for i in range(N):
        local_stretch = 0
        for j in range(N):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r2 = np.dot(r_vec, r_vec) + sigma**2
            r = np.sqrt(r2)
            
            # Det(a, b, c) = (a x b) . c
            det = np.dot(np.cross(omegas[i], omegas[j]), r_vec)
            term = (3.0 / (4.0 * np.pi)) * np.dot(omegas[i], r_vec) * det / (r**5)
            local_stretch += term
        total_stretch += np.abs(local_stretch)
    return total_stretch

def compute_graph_majorant(nodes, omegas, sigma=0.1):
    """
    Computes lambda_max of the Interaction Laplacian L = D - W
    where W_ij = (3/4pi) / r_ij^3 (matching the BS stretching scale)
    and Z = sum |omega_i|^2
    """
    N = len(nodes)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            r_vec = nodes[i] - nodes[j]
            r = np.sqrt(np.dot(r_vec, r_vec) + sigma**2)
            # We use the same leading order scaling as the stretching kernel
            W[i,j] = W[j,i] = (3.0 / (4.0 * np.pi)) / (r**3)
    
    # We use lambda_max of the ADJACENCY matrix for direct stretching bound
    # or the LAPLACIAN if we consider the relative stretching.
    # Meridian suggested lambda_max of the interaction Laplacian.
    d = W.sum(axis=1)
    L = np.diag(d) - W
    evals = eigh(L, eigvals_only=True)
    l_max = evals[-1]
    
    Z = np.sum(np.linalg.norm(omegas, axis=1)**2)
    return l_max * Z

def run_stretching_audit(n_trials=100, n_filaments=5, n_seg=30):
    print(f"--- FUDGE-FREE STRETCHING AUDIT (Option C) ---")
    print(f"Goal: $|Stretching| \leq C \cdot \lambda_{max} \cdot Z$")
    print(f"{'Trial':<10} | {'Stretching':<15} | {'Graph Bound':<15} | {'Ratio C':<10}")
    print("-" * 60)
    
    ratios = []
    for t in range(n_trials):
        # Generate random filaments
        nodes = []
        omegas = []
        for _ in range(n_filaments):
            origin = np.random.uniform(-1, 1, 3)
            direction = np.random.normal(0, 1, 3)
            direction /= np.linalg.norm(direction)
            length = np.random.uniform(0.5, 2.0)
            for s in np.linspace(0, length, n_seg):
                nodes.append(origin + direction * s)
                omegas.append(direction)
        
        nodes = np.array(nodes)
        omegas = np.array(omegas)
        
        S = compute_physical_stretching(nodes, omegas)
        GB = compute_graph_majorant(nodes, omegas)
        
        ratio = S / (GB + 1e-9)
        ratios.append(ratio)
        
        if t % 20 == 0:
            print(f"{t:<10} | {S:<15.4f} | {GB:<15.4f} | {ratio:<10.4f}")
            
    print("-" * 60)
    print(f"AUDIT COMPLETE.")
    print(f"Max C: {np.max(ratios):.6f}")
    print(f"Mean C: {np.mean(ratios):.6f}")
    print(f"Std Dev: {np.std(ratios):.6f}")

if __name__ == "__main__":
    run_stretching_audit()
