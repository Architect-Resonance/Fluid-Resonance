import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def audit_aliasing_diff():
    print(f"--- S51: ALIASING INSTABILITY AUDIT ---")
    print(f"{'Step':<10} | {'Z (Dealias)':<15} | {'Z (No Dealias)'}")
    print("-" * 55)
    
    N = 32 # Lower res to see aliasing faster
    nu = 0.0 # Zero viscosity for maximum stress
    
    solver_y = SpectralNSSolver(N=N, nu=nu, use_dealias=True)
    solver_n = SpectralNSSolver(N=N, nu=nu, use_dealias=False)
    
    solver_y.dt = 0.002
    solver_n.dt = 0.002
    
    # Use identical random field
    u_hat = np.random.randn(3, N, N, N) + 1j * np.random.randn(3, N, N, N)
    solver_y.u_hat = u_hat.copy()
    solver_n.u_hat = u_hat.copy()
    
    for i in range(100):
        solver_y.step_rk4()
        solver_n.step_rk4()
        
        if i % 20 == 0:
            _, zy = solver_y.compute_diagnostics()
            _, zn = solver_n.compute_diagnostics()
            print(f"{i:<10} | {zy:<15.4f} | {zn:<15.4f}")
            
        if np.isnan(zy) or np.isnan(zn):
            print("NaN detected.")
            break

    print("-" * 55)
    print("INSIGHT: If Z (No Dealias) explodes while Z (Dealias) stays stable,")
    print("then the 'Blow-up' is likely an aliasing artifact.")

if __name__ == "__main__":
    audit_aliasing_diff()
