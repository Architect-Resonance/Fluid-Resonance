import numpy as np
from scipy.integrate import solve_ivp

def restricted_euler_with_nonlocal_pressure(t, y, nu=0.01, nonlocal_gain=0.5):
    """
    3x3 Matrix ODE for velocity gradient A:
    dA/dt = -A^2 - H - nu*A
    where H is the pressure Hessian.
    Non-local model: H = trace(A^2) * (1/3 * I) + nonlocal_gain * (A^2_anisotropic)
    """
    A = y.reshape((3, 3))
    
    # Incompressibility: ensure trace(A) remains 0
    tr_A = np.trace(A)
    if np.abs(tr_A) > 1e-10:
        A = A - (tr_A/3.0) * np.eye(3)
        
    A2 = np.dot(A, A)
    tr_A2 = np.trace(A2)
    
    # Isotropic pressure (Enforces trace-free dA/dt)
    H_iso = (tr_A2 / 3.0) * np.eye(3)
    
    # Non-local Anisotropic pressure (The coordinator dimension)
    A2_aniso = A2 - H_iso
    # Nonlinear coordination: stronger pressure reaction at high intensities
    H_nonlocal = nonlocal_gain * (tr_A2 / (tr_A2 + 100.0)) * A2_aniso
    
    H = H_iso + H_nonlocal
    
    dAdt = -A2 - H - nu * A
    
    return dAdt.flatten()

def run_pressure_coordinator_audit():
    print(f"--- PRESSURE COORDINATOR AUDIT (Non-local Dimension) ---")
    
    # Initial condition: Highly intense skewed gradient
    np.random.seed(42) # Reproducibility
    A0 = np.random.normal(0, 50, (3, 3))
    A0 -= np.trace(A0)/3.0 * np.eye(3)
    y0 = A0.flatten()
    
    t_span = (0, 0.1) # Short time because of blow-up potential
    t_eval = np.linspace(0, 0.1, 100)
    
    # Mode 1: Local only (nonlocal_gain = 0)
    sol_local = solve_ivp(restricted_euler_with_nonlocal_pressure, t_span, y0, 
                          args=(0.01, 0.0), t_eval=t_eval)
    
    # Mode 2: Non-local Coordinator (nonlocal_gain = 0.5)
    sol_nonlocal = solve_ivp(restricted_euler_with_nonlocal_pressure, t_span, y0, 
                            args=(0.01, 0.9), t_eval=t_eval)

    # Compute Z = tr(A^T A)
    def compute_z(sol):
        z = []
        if not sol.success: return [np.nan] * len(t_eval)
        for i in range(sol.y.shape[1]):
            A = sol.y[:, i].reshape((3, 3))
            z.append(np.trace(np.dot(A.T, A)))
        return np.array(z)
    
    z_local = compute_z(sol_local)
    z_nonlocal = compute_z(sol_nonlocal)
    
    print(f"{'Time':<10} | {'Z_local':<15} | {'Z_nonlocal':<15}")
    print("-" * 45)
    max_idx = min(len(z_local), len(z_nonlocal))
    for i in range(0, max_idx, 20):
        print(f"{t_eval[i]:<10.3f} | {z_local[i]:<15.2f} | {z_nonlocal[i]:<15.2f}")
    
    print("-" * 45)
    if z_nonlocal[-1] < z_local[-1]:
        reduction = (1 - z_nonlocal[-1]/z_local[-1]) * 100
        print(f"RESULT: Non-local Pressure reduction: {reduction:.2f}%")
    else:
        print(f"RESULT: No depletion effect found.")

if __name__ == "__main__":
    run_pressure_coordinator_audit()
