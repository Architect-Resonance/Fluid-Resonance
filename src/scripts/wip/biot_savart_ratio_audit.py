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
            # Biot-Savart Strain Kernel
            for a in range(3):
                for b in range(3):
                    val = 0
                    for c in range(3):
                        for d in range(3):
                            eps = 0
                            if (a,c,d) in [(0,1,2), (1,2,0), (2,0,1)]: eps = 1
                            elif (a,c,d) in [(0,2,1), (2,1,0), (1,0,2)]: eps = -1
                            if eps == 0: continue
                            # Deriv of v is kernel * omega
                            # Kernel ~ 1/r^3 * (3*ri*rj/r^2 - delta_ij)
                            kernel = (3.0 * r_vec[b] * r_vec[c] / (r**5)) - ((1.0 if b==c else 0.0) / (r**3))
                            val += eps * omegas[j][d] * kernel
                    S_local[a, b] += (1.0 / (4.0 * np.pi)) * val
        # Symmetrize and trace-free
        S_sym = (S_local + S_local.T) / 2
        S_sym -= np.trace(S_sym) / 3.0 * np.eye(3)
        Strains[i] = S_sym
    return Strains

def run_biot_savart_ratio_audit():
    print(f"--- S44b: BIOT-SAVART RATIO AUDIT ---")
    print(f"{'Config':<20} | {'Alpha/Omega Ratio':<20} | {'Gain Floor'}")
    print("-" * 65)
    
    # Test 1: Random Cloud
    N = 20
    nodes = np.random.randn(N, 3)
    omegas = np.random.randn(N, 3)
    Strains = compute_physical_fields(nodes, omegas)
    
    ratios = []
    for i in range(N):
        o = omegas[i]
        o_hat = o / np.linalg.norm(o)
        S = Strains[i]
        
        alpha = np.dot(o_hat, S @ o_hat)
        rot_vec = (S @ o_hat) - np.dot(o_hat, S @ o_hat) * o_hat
        omega_rot = np.linalg.norm(rot_vec)
        
        ratios.append(alpha / (omega_rot + 1e-9))
    
    print(f"{'Random Cloud':<20} | {np.max(ratios):<20.6f} | {np.exp(np.pi/2 * np.max(ratios)):.4f}")

    # Test 2: The Ant-Coordinator (Malicious Case from S38)
    nodes_m = np.array([[0,0,0], [1,0,0], [-1,0,0]], dtype=float)
    omegas_m = np.array([[1,0,0], [0,1,0], [0,1,0]], dtype=float)
    S_m = compute_physical_fields(nodes_m, omegas_m)
    
    o = omegas_m[0]
    o_hat = o / np.linalg.norm(o)
    alpha = np.dot(o_hat, S_m[0] @ o_hat)
    rot_vec = (S_m[0] @ o_hat) - np.dot(o_hat, S_m[0] @ o_hat) * o_hat
    omega_rot = np.linalg.norm(rot_vec)
    r_m = alpha / (omega_rot + 1e-9)
    print(f"{'Anti-Coordinator':<20} | {r_m:<20.6f} | {np.exp(np.pi/2 * r_m):.4f}")

    print("-" * 65)
    print("INSIGHT: If Ratio < 2.0, the total gain per pass is < 23 (Finite).")
    print("If Ratio is bounded by geometry, then blow-up is impossible.")

if __name__ == "__main__":
    run_biot_savart_ratio_audit()
