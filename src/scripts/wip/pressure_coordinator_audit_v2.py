import numpy as np
from scipy.integrate import solve_ivp

def restricted_euler_with_nonlocal_pressure(t, y, nu=0.01, nonlocal_gain=0.5):
    A = y.reshape((3, 3))
    tr_A = np.trace(A)
    if np.abs(tr_A) > 1e-10:
        A = A - (tr_A/3.0) * np.eye(3)
    A2 = np.dot(A, A)
    tr_A2 = np.trace(A2)
    
    # Pressure is the trace of A^2
    H_iso = (tr_A2 / 3.0) * np.eye(3)
    
    # Non-local coordination: Pressure reacts to the anisotropic intensity
    A2_aniso = A2 - H_iso
    # Stabilization factor: Non-locality scales with global enstrophy
    H_nonlocal = nonlocal_gain * (tr_A2 / (tr_A2 + 10.0)) * A2_aniso
    
    dAdt = -A2 - (H_iso + H_nonlocal) - nu * A
    
    # Blow-up detector: if any element is too large, return zero to stop
    if np.any(np.abs(y) > 1e10):
        return np.zeros_like(y)
    
    return dAdt.flatten()

def run_stabilization_audit():
    print(f"--- STABILIZATION AUDIT: COORDINATOR DIMENSION ---")
    
    np.random.seed(42)
    # Start with a highly unstable configuration
    A0 = np.random.normal(0, 100, (3, 3))
    A0 -= np.trace(A0)/3.0 * np.eye(3)
    y0 = A0.flatten()
    
    t_eval = np.linspace(0, 0.05, 50)
    
    # 1. Local (Blow-up expected)
    sol_local = solve_ivp(restricted_euler_with_nonlocal_pressure, (0, 0.05), y0, 
                          args=(0.01, 0.0), t_eval=t_eval, method='RK45')
    
    # 2. Non-local (Stabilization expected)
    sol_nonlocal = solve_ivp(restricted_euler_with_nonlocal_pressure, (0, 0.05), y0, 
                            args=(0.01, 1.0), t_eval=t_eval, method='RK45')

    print(f"{'Time':<10} | {'Status Local':<15} | {'Status Non-local':<15}")
    print("-" * 50)
    
    for i in range(len(t_eval)):
        z_l = "BLOW-UP" if (i >= len(sol_local.t) or np.any(np.isnan(sol_local.y[:,i]))) else f"{np.trace(sol_local.y[:,i].reshape((3,3))):.2f}"
        z_n = "BLOW-UP" if (i >= len(sol_nonlocal.t) or np.any(np.isnan(sol_nonlocal.y[:,i]))) else f"{np.trace(sol_nonlocal.y[:,i].reshape((3,3))):.2f}"
        
        # We actually want Enstrophy tr(A^T A)
        if z_l != "BLOW-UP":
            A = sol_local.y[:,i].reshape((3,3)); z_l = f"{np.trace(np.dot(A.T, A)):.1e}"
        if z_n != "BLOW-UP":
            A = sol_nonlocal.y[:,i].reshape((3,3)); z_n = f"{np.trace(np.dot(A.T, A)):.1e}"
            
        if i % 10 == 0:
            print(f"{t_eval[i]:<10.3f} | {z_l:<15} | {z_n:<15}")

    print("-" * 50)
    if "BLOW-UP" in z_l and "BLOW-UP" not in z_n:
        print("RESULT: Verified. Non-local Pressure acts as a Regularization Dimension.")

if __name__ == "__main__":
    run_stabilization_audit()
