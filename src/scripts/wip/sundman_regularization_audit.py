import numpy as np
from scipy.integrate import solve_ivp

def vortex_dynamics_sundman(tau, state, n_segments, nu):
    # state: [x1, y1, z1, ..., ox1, oy1, oz1, ..., t]
    # We add 't' as a state variable because tau is the new independent variable.
    nodes = state[:3*n_segments].reshape((n_segments, 3))
    omegas = state[3*n_segments:6*n_segments].reshape((n_segments, 3))
    t = state[-1]
    
    # Compute current Z (Enstrophy-like)
    Z = np.sum(np.linalg.norm(omegas, axis=1)**2)
    
    # Time Lifting Factor: dt/dtau = 1 / (1 + Z)
    # As Z -> inf, t crawls, tau flies.
    dt_dtau = 1.0 / (1.0 + Z)
    
    dn_dt = np.zeros_like(nodes)
    do_dt = np.zeros_like(omegas)
    
    sigma = 0.05 # Smaller core for more intensity
    
    for i in range(n_segments):
        for j in range(n_segments):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r2 = np.dot(r_vec, r_vec) + sigma**2
            r = np.sqrt(r2)
            
            # Stretching vec: d(omega_i)/dt
            stretch_vec = (3.0 / (4.0 * np.pi)) * np.dot(omegas[i], r_vec) * np.cross(omegas[j], r_vec) / (r**5)
            do_dt[i] += stretch_vec
            
        # Dissipation
        do_dt[i] -= nu * omegas[i]
        
    # Transform to tau-domain: dX/dtau = (dX/dt) * (dt/dtau)
    dn_dtau = dn_dt * dt_dtau
    do_dtau = do_dt * dt_dtau
    
    return np.concatenate([dn_dtau.flatten(), do_dtau.flatten(), [dt_dtau]])

def run_sundman_audit():
    print(f"--- SUNDMAN TIME-LIFTING AUDIT (Camlin 2025) ---")
    print(f"{'Tau':<10} | {'Physical Time T':<15} | {'Enstrophy Z':<15}")
    print("-" * 55)
    
    nu = 0.01 # Low viscosity to encourage blow-up
    n_seg = 2
    # Extreme singular configuration: Near-parallel filaments, very close
    n0 = np.array([[0, 0, 0], [0.05, 0, 0]], dtype=float)
    o0 = np.array([[0, 10, 0], [0.1, -10, 0]], dtype=float)
    y0 = np.concatenate([n0.flatten(), o0.flatten(), [0.0]])
    
    # Run in tau-domain for a long time
    sol = solve_ivp(vortex_dynamics_sundman, [0, 100], y0, args=(n_seg, nu), 
                    method='RK45', rtol=1e-8)
    
    for i in range(0, len(sol.t), len(sol.t)//10):
        state = sol.y[:, i]
        oms = state[3*n_seg:6*n_seg].reshape((n_seg, 3))
        Z = np.sum(np.linalg.norm(oms, axis=1)**2)
        t = state[-1]
        print(f"{sol.t[i]:<10.1f} | {t:<15.6f} | {Z:<15.4f}")

if __name__ == "__main__":
    run_sundman_audit()
