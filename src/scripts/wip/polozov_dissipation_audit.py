import numpy as np
from scipy.integrate import solve_ivp

def vortex_dynamics(t, state, n_segments, nu, emergent_factor=0.0):
    # state: [x1, y1, z1, ..., ox1, oy1, oz1, ...]
    nodes = state[:3*n_segments].reshape((n_segments, 3))
    omegas = state[3*n_segments:].reshape((n_segments, 3))
    
    dn = np.zeros_like(nodes)
    do = np.zeros_like(omegas)
    
    sigma = 0.01 # Much smaller core for intensity
    
    for i in range(n_segments):
        # 1. Biot-Savart Velocity & Strain (Simplification: Segments move with velocity)
        vel = np.zeros(3)
        strain = np.zeros((3, 3))
        for j in range(n_segments):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r2 = np.dot(r_vec, r_vec) + sigma**2
            r = np.sqrt(r2)
            
            # Velocity: (1/4pi) * (omega_j x r) / r^3
            v_term = (1.0 / (4.0 * np.pi)) * np.cross(omegas[j], r_vec) / (r**3)
            vel += v_term
            
            # Strain: (3/4pi) * (r (x) (omega_j x r) + cross product terms) / r^5
            # Simplified Stretching term directly:
            # d(omega_i)/dt = S_ij . omega_i
            # We use the dot-cross-dot form: S_ij . omega_i = (3/4pi) * (omega_i.r) * (omega_j x r) / r^5
            stretch_vec = (3.0 / (4.0 * np.pi)) * np.dot(omegas[i], r_vec) * np.cross(omegas[j], r_vec) / (r**5)
            do[i] += stretch_vec
            
        dn[i] = vel
        
        # 2. Emergent Nonlinear Dissipation (Polozov 2025)
        # Standard: -nu * omega
        # Emergent: -nu * (1 + beta * |omega|^2) * omega
        mag_o = np.linalg.norm(omegas[i])
        dissipation = -nu * (1.0 + emergent_factor * mag_o**2) * omegas[i]
        do[i] += dissipation
        
    return np.concatenate([dn.flatten(), do.flatten()])

def run_polozov_audit():
    print(f"--- POLOZOV EMERGENT DISSIPATION AUDIT (2025) ---")
    print(f"{'Beta (Emergent)':<15} | {'Max Enstrophy':<15} | {'Extinction Time'}")
    print("-" * 55)
    
    nu = 0.1
    n_seg = 2
    # Start with a stretching configuration (Two filaments attacking each other)
    n0 = np.array([[0, 0, 0], [0.2, 0, 0]], dtype=float)
    o0 = np.array([[0, 1, 0], [0, -1, 0]], dtype=float) * 100.0 # Extreme initial vorticity
    y0 = np.concatenate([n0.flatten(), o0.flatten()])
    
    for beta in [0.0, 0.1, 0.5, 1.0]:
        sol = solve_ivp(vortex_dynamics, [0, 10], y0, args=(n_seg, nu, beta), 
                        method='RK45', rtol=1e-6)
        
        # Compute max enstrophy Z = sum |omega|^2
        z_vals = []
        for state in sol.y.T:
            oms = state[3*n_seg:].reshape((n_seg, 3))
            z_vals.append(np.sum(np.linalg.norm(oms, axis=1)**2))
            
        max_z = max(z_vals)
        extinction = sol.t[np.where(np.array(z_vals) < 0.1)[0][0]] if any(np.array(z_vals) < 0.1) else ">10"
        
        print(f"{beta:<15.2f} | {max_z:<15.4f} | {extinction}")

if __name__ == "__main__":
    run_polozov_audit()
