import numpy as np
from scipy.integrate import solve_ivp

def restricted_euler_ode(t, A_flat, mode='A', nu=0.01):
    """
    dA/dt = -A^2 - H + nu*laplacian(A)
    A is 3x3 velocity gradient tensor, trace(A)=0
    """
    A = A_flat.reshape((3, 3))
    
    # Trace(A^2) is part of the local pressure term P_local = -trace(A^2)
    # The isotropic part of the pressure Hessian ensures trace(dA/dt) = 0
    A2 = A @ A
    tr_A2 = np.trace(A2)
    
    # H_iso = -(1/3) * trace(A^2) * I
    H_iso = (1/3) * tr_A2 * np.eye(3)
    
    if mode == 'A':
        # Restricted Euler: H_aniso = 0
        H = H_iso
    else:
        # Mode B: Pressure Hessian closure (Simplified Tetrad/Chevillard-Meneveau proxy)
        # H_aniso is modeled to oppose the stretching
        # Here we use a term that relaxes A towards intermediate alignment
        H_aniso = 0.5 * (A2 - H_iso) # Rough proxy for the non-local pressure Hessian
        H = H_iso + H_aniso
        
    dA = -A2 + H - nu * A # Viscous damping
    return dA.flatten()

def run_pressure_hessian_audit(n_trials=100, t_span=(0, 10)):
    results = {'A': [], 'B': []}
    
    print(f"--- PRESSURE HESSIAN AUDIT: RESTRICTED EULER MODEL ---")
    
    for mode in ['A', 'B']:
        blow_ups = 0
        alignments = [] # dot product of omega and alpha-eigenvector
        
        for _ in range(n_trials):
            # 1. Random Trace-free initial A
            A0 = np.random.normal(0, 1, (3, 3))
            A0 -= np.trace(A0)/3 * np.eye(3)
            
            # 2. Integrate
            try:
                sol = solve_ivp(restricted_euler_ode, t_span, A0.flatten(), args=(mode,), method='RK45')
                
                if not sol.success or np.any(np.abs(sol.y) > 1e10):
                    blow_ups += 1
                else:
                    # Analyze final alignment
                    Af = sol.y[:, -1].reshape((3, 3))
                    S = 0.5 * (Af + Af.T) # Strain tensor
                    w, v = np.linalg.eigh(S)
                    # Omega is the antisymmetric part of A
                    vorticity_vec = np.array([Af[2,1]-Af[1,2], Af[0,2]-Af[2,0], Af[1,0]-Af[0,1]])
                    if np.linalg.norm(vorticity_vec) > 1e-9:
                        vorticity_vec /= np.linalg.norm(vorticity_vec)
                        # Alignment with alpha (v[:, 2])
                        alignments.append(np.abs(np.dot(vorticity_vec, v[:, 2])))
            except:
                blow_ups += 1
                
        results[mode] = (blow_ups, np.mean(alignments) if alignments else 0)
        print(f"Mode {mode}: Blow-ups = {blow_ups}/{n_trials} | Mean Alpha-Alignment = {results[mode][1]:.4f}")

if __name__ == "__main__":
    run_pressure_hessian_audit()
