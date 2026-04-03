import numpy as np
from scipy.integrate import solve_ivp

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

def dynamics_self_induced(t, state):
    # state: [nodes, omegas]
    N = len(state) // 6
    nodes = state[:3*N].reshape((N, 3))
    omegas = state[3*N:].reshape((N, 3))
    
    # 1. Compute Strain and Velocity
    Strains = compute_physical_fields(nodes, omegas)
    
    # 2. Physics: d(omega)/dt = S . omega, d(nodes)/dt = velocity
    # (Velocity simplified for this audit to focus on stretching)
    d_omegas = np.array([Strains[i] @ omegas[i] for i in range(N)])
    d_nodes = np.zeros_like(nodes) # Frozen positions for alignment check
    
    return np.concatenate([d_nodes.flatten(), d_omegas.flatten()])

def run_self_induced_audit():
    print(f"--- S44e: THE SELF-INDUCED RATIO AUDIT ---")
    print(f"{'Time':<10} | {'Intensity':<15} | {'Alpha/Omega (Self)'}")
    print("-" * 55)
    
    N = 5
    nodes = np.random.randn(N, 3)
    omegas = np.random.randn(N, 3)
    state0 = np.concatenate([nodes.flatten(), omegas.flatten()])
    
    t_span = [0, 5.0]
    sol = solve_ivp(dynamics_self_induced, t_span, state0, method='RK45',
                    t_eval=np.linspace(0, 5, 10))
    
    for k in range(len(sol.t)):
        s_k = sol.y[:, k]
        n_k = s_k[:3*N].reshape((N, 3))
        o_k = s_k[3*N:].reshape((N, 3))
        Strains = compute_physical_fields(n_k, o_k)
        
        # Track ratio for segment 0
        o0 = o_k[0]
        o0_hat = o0 / np.linalg.norm(o0)
        S0 = Strains[0]
        alpha = np.dot(o0_hat, S0 @ o0_hat)
        omega_rot = np.linalg.norm((S0 @ o0_hat) - alpha * o0_hat)
        ratio = alpha / (omega_rot + 1e-9)
        
        print(f"{sol.t[k]:<10.2f} | {np.linalg.norm(o_k):<15.4f} | {ratio:.6f}")

    print("-" * 55)
    print("INSIGHT: If Ratio saturates or oscillates, blow-up is prevented.")

if __name__ == "__main__":
    run_self_induced_audit()
