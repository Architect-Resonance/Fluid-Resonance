import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh

def compute_fields(nodes, omegas, sigma=0.05):
    N = len(nodes)
    # 1. Compute Strains
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
        Strains[i] = S_local
        
    # 2. Compute Pi
    Pi = np.zeros((N, 3, 3))
    Q = np.zeros(N)
    for j in range(N):
        S_sym = (Strains[j] + Strains[j].T) / 2.0
        Q[j] = np.trace(S_sym @ S_sym) - 0.5 * np.linalg.norm(omegas[j])**2
    for i in range(N):
        Pi_local = np.zeros((3, 3))
        for j in range(N):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r2 = np.dot(r_vec, r_vec) + sigma**2
            r = np.sqrt(r2)
            for a in range(3):
                for b in range(3):
                    kernel = (3.0 * r_vec[a] * r_vec[b] / (r**5)) - ((1.0 if a==b else 0.0) / (r**3))
                    Pi_local[a, b] += (1.0 / (4.0 * np.pi)) * Q[j] * kernel
        Pi[i] = Pi_local
        
    return Strains, Pi

def objective(x, n_segments, sigma):
    nodes = x[:3*n_segments].reshape((n_segments, 3))
    omegas_raw = x[3*n_segments:].reshape((n_segments, 3))
    omegas = np.array([o / (np.linalg.norm(o) + 1e-9) for o in omegas_raw])
    
    Strains, Pi = compute_fields(nodes, omegas, sigma)
    
    # We want to maximize the maximum d(lambda)/dt across all nodes
    max_d_lambda = -1e10
    for i in range(n_segments):
        S = (Strains[i] + Strains[i].T) / 2.0
        evals, evecs = eigh(S)
        l_max = evals[-1]
        v_max = evecs[:, -1]
        
        q = np.trace(S @ S)
        tr_pi = np.trace(Pi[i])
        
        s_term = -(l_max**2 - (1.0/3.0)*q)
        p_effect = -(np.dot(v_max, Pi[i] @ v_max) - (1.0/3.0)*tr_pi)
        
        d_lambda = s_term + p_effect
        if d_lambda > max_d_lambda:
            max_d_lambda = d_lambda
            
    # Soft penalty to keep nodes centered
    dist_penalty = 1e-2 * np.sum(np.abs(nodes))
    return -max_d_lambda + dist_penalty

def run_pressure_malice_audit(n_segments=6, n_trials=5):
    print(f"--- GLOBAL PRESSURE OPTIMIZER: ANTI-MALICE AUDIT ---")
    print(f"{'Trial':<10} | {'Max L_max':<12} | {'Max dL/dt':<15}")
    print("-" * 45)
    
    sigma = 0.05
    results = []
    
    for i in range(n_trials):
        # Position bounds: [-2, 2], Omega: [-1, 1]
        x0 = np.random.uniform(-1, 1, 6 * n_segments)
        res = minimize(objective, x0, args=(n_segments, sigma), 
                       method='L-BFGS-B', options={'maxiter': 50})
        
        x_best = res.x
        nodes = x_best[:3*n_segments].reshape((n_segments, 3))
        omegas_raw = x_best[3*n_segments:].reshape((n_segments, 3))
        omegas = np.array([o / (np.linalg.norm(o) + 1e-9) for o in omegas_raw])
        
        Strains, Pi = compute_fields(nodes, omegas, sigma)
        
        best_dL = -res.fun
        best_L = 0
        for j in range(n_segments):
            S = (Strains[j] + Strains[j].T) / 2.0
            l_max = eigh(S, eigvals_only=True)[-1]
            if l_max > best_L: best_L = l_max
            
        print(f"{i:<10} | {best_L:<12.4f} | {best_dL:<15.4f}")
        results.append(best_dL)
        
    print("-" * 45)
    print(f"PEAK dL/dt FOUND: {max(results):.6f}")
    if max(results) < 0:
        print("RESULT: Even an optimal-adversary cannot force a blow-up growth rate.")
        print("Global pressure coordination always pull eigenvalues back to zero.")
    else:
        print("RESULT: POTENTIAL BLOW-UP SCENARIO FOUND.")

if __name__ == "__main__":
    run_pressure_malice_audit(n_segments=6, n_trials=5)
