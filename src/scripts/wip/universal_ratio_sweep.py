import numpy as np

def compute_physical_fields(nodes, omegas, sigma=0.05):
    N = len(nodes)
    Strains = np.zeros((N, 3, 3))
    for i in range(N):
        S_local = np.zeros((3, 3))
        for j in range(N):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r2 = np.dot(r_vec, r_vec) + sigma**2
            r = np.sqrt(r2)
            for a in range(3):
                for b in range(3):
                    val = 0
                    for c in range(3):
                        for d in range(3):
                            eps = 0
                            if (a,c,d) in [(0,1,2), (1,2,0), (2,0,1)]: eps = 1
                            elif (a,c,d) in [(0,2,1), (2,1,0), (1,0,2)]: eps = -1
                            if eps == 0: continue
                            kernel = (3.0 * r_vec[b] * r_vec[c] / (r**5)) - ((1.0 if b==c else 0.0) / (r**3))
                            val += eps * omegas[j][d] * kernel
                    S_local[a, b] += (1.0 / (4.0 * np.pi)) * val
        S_sym = (S_local + S_local.T) / 2
        S_sym -= np.trace(S_sym) / 3.0 * np.eye(3)
        Strains[i] = S_sym
    return Strains

def run_universal_sweep():
    print(f"--- S44c: THE UNIVERSAL RATIO SWEEP ---")
    print(f"{'Config Type':<20} | {'Max Ratio':<15} | {'Mean Ratio'}")
    print("-" * 55)
    
    iters = 100
    all_max_ratios = []
    
    # 1. Random Clouds
    ratios_cloud = []
    for _ in range(iters):
        N = 10
        nodes = np.random.randn(N, 3)
        omegas = np.random.randn(N, 3)
        Strains = compute_physical_fields(nodes, omegas)
        for i in range(N):
            o_hat = omegas[i] / np.linalg.norm(omegas[i])
            alpha = np.dot(o_hat, Strains[i] @ o_hat)
            omega_rot = np.linalg.norm((Strains[i] @ o_hat) - alpha * o_hat)
            ratios_cloud.append(alpha / (omega_rot + 1e-9))
    
    print(f"{'Random Clouds':<20} | {np.max(ratios_cloud):<15.4f} | {np.mean(ratios_cloud):.4f}")

    # 2. Anisotropic Filaments (Stretched along Z)
    ratios_film = []
    for _ in range(iters):
        N = 10
        nodes = np.random.randn(N, 3) * np.array([1, 1, 10])
        omegas = np.random.randn(N, 3) * np.array([1, 1, 10])
        Strains = compute_physical_fields(nodes, omegas)
        for i in range(N):
            o_hat = omegas[i] / np.linalg.norm(omegas[i])
            alpha = np.dot(o_hat, Strains[i] @ o_hat)
            omega_rot = np.linalg.norm((Strains[i] @ o_hat) - alpha * o_hat)
            ratios_film.append(alpha / (omega_rot + 1e-9))
    
    print(f"{'Anisotrop Filaments':<20} | {np.max(ratios_film):<15.4f} | {np.mean(ratios_film):.4f}")

    print("-" * 55)
    print("FINAL ANALYSIS: If Max Ratio < 1.0, the growth rate is SLOWER")
    print("than the rotation rate. Stability is a geometrical necessity.")

if __name__ == "__main__":
    run_universal_sweep()
