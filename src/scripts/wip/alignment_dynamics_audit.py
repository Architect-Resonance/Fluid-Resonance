import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

def compute_fields(nodes, omegas, sigma=0.05):
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
        Strains[i] = S_local
    return Strains

def dynamics_alignment(t, state, n_segments, nu, sigma):
    nodes = state[:3*n_segments].reshape((n_segments, 3))
    omegas = state[3*n_segments:].reshape((n_segments, 3))
    
    Strains = compute_fields(nodes, omegas, sigma)
    
    dn = np.zeros_like(nodes)
    do = np.zeros_like(omegas)
    
    for i in range(n_segments):
        S = Strains[i]
        do[i] = S @ omegas[i] - nu * omegas[i]
        
    return np.concatenate([dn.flatten(), do.flatten()])

def run_alignment_audit():
    print(f"--- S39: VORTICITY-STRAIN ALIGNMENT DYNAMICS (nu=0) ---")
    print(f"{'Time':<10} | {'Enstrophy Z':<15} | {'Align Cos':<10} | {'Stretch Rate'}")
    print("-" * 65)
    
    n_segments = 2
    sigma = 0.05
    nu = 0.0 # PURE EULER
    
    # MALICIOUS START 
    n0 = np.array([[0, 0, 0], [0.01, 0.01, 0]], dtype=float)
    o0 = np.array([[1, 1, 0], [-1, 1, 0]], dtype=float) * 20.0
    state0 = np.concatenate([n0.flatten(), o0.flatten()])
    
    sol = solve_ivp(dynamics_alignment, [0, 2], state0, args=(n_segments, nu, sigma),
                    method='RK45', rtol=1e-8)
    
    for i in range(len(sol.t)):
        t = sol.t[i]
        state = sol.y[:, i]
        nodes = state[:3*n_segments].reshape((n_segments, 3))
        omegas = state[3*n_segments:].reshape((n_segments, 3))
        
        Z = np.sum(np.linalg.norm(omegas, axis=1)**2)
        
        Strains = compute_fields(nodes, omegas, sigma)
        S = (Strains[0] + Strains[0].T) / 2.0
        evals, evecs = eigh(S)
        e_max = evecs[:, -1]
        o_unit = omegas[0] / (np.linalg.norm(omegas[0]) + 1e-9)
        alignment = np.abs(np.dot(o_unit, e_max))
        
        # Stretching rate: sum(omega_k . S_k . omega_k)
        stretch_rate = 0
        for k in range(n_segments):
            S_k = (Strains[k] + Strains[k].T) / 2.0
            stretch_rate += np.dot(omegas[k], S_k @ omegas[k])
            
        print(f"{t:<10.3f} | {Z:<15.4f} | {alignment:.6f} | {stretch_rate:.4f}")

if __name__ == "__main__":
    run_alignment_audit()
