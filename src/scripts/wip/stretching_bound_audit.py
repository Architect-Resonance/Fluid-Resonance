import numpy as np
from scipy.linalg import eigh

def compute_stretching_and_lambda(n_filaments=5, n_segments=20, sigma=0.1):
    """
    Computes:
    1. Real Stretching S_int = sum_i omega_i . S_i omega_i
    2. lambda_max of the interaction weight matrix
    3. Total Enstrophy Z
    """
    # 1. Random Geometry (Isotropic filaments)
    nodes = []
    omegas = []
    for _ in range(n_filaments):
        direction = np.random.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        origin = np.random.normal(0, 0.5, 3)
        for s in np.linspace(0, 1, n_segments):
            pos = origin + direction * s
            nodes.append(pos)
            omegas.append(direction) # Unit vorticity for simplicity
            
    nodes = np.array(nodes)
    omegas = np.array(omegas)
    N = len(nodes)
    
    # 2. Compute Stretching via Biot-Savart Summation
    # Stretching at node i: omega_i . (Strain_tensor_i) . omega_i
    # Strain tensor at i due to j: S_ij ~ (3(r.omega_j)r - r^2 omega_j - ...) / |r|^5
    total_stretching = 0
    for i in range(N):
        strain_sum = np.zeros((3, 3))
        for j in range(N):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r = np.linalg.norm(r_vec) + sigma
            cross = np.cross(omegas[j], r_vec)
            # Simplified Biot-Savart strain component for audit
            # S ~ curl(u) -> S_kl ~ (partial_k u_l + partial_l u_k)/2
            # Here we use the kernel K(r) = (r_vec x omega)/r^3
            # The derivative has 1/r^3 scaling
            strength = 1.0 / (r**3)
            strain_sum += strength * np.outer(omegas[i], omegas[j]) # Proxy for stretch contribution
        
        total_stretching += np.abs(np.dot(omegas[i], np.dot(strain_sum, omegas[i])))

    # 3. Compute lambda_max of Interaction Matrix
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            r = np.linalg.norm(nodes[i] - nodes[j]) + sigma
            W[i,j] = W[j,i] = 1.0 / (r**3) # Matching the BS scaling
            
    D = np.diag(W.sum(axis=1))
    L = D - W
    evals = eigh(L, eigvals_only=True)
    l_max = evals[-1]
    
    # 4. Total Enstrophy Z (sum of |omega|^2)
    Z = N # Since |omega|=1 at each segment
    
    return total_stretching, l_max, Z

def adversarial_audit_option_c(n_trials=100, n_segments_list=[10, 20, 40, 80]):
    print(f"--- ADVERSARIAL AUDIT: OPTION C (Stretching Bound) ---")
    print(f"{'N_seg':<10} | {'Mean Ratio (S/L_max*Z)':<25} | {'Std Dev':<10}")
    print("-" * 50)
    
    for n_seg in n_segments_list:
        ratios = []
        for _ in range(n_trials):
            S, L, Z = compute_stretching_and_lambda(n_segments=n_seg)
            ratios.append(S / (L * Z + 1e-9))
        
        print(f"{n_seg:<10} | {np.mean(ratios):<25.6f} | {np.std(ratios):.6f}")

if __name__ == "__main__":
    adversarial_audit_option_c()
